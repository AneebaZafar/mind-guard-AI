from flask import Flask, render_template, request, send_file
import sqlite3
import os
import io
from datetime import datetime, timedelta
from collections import Counter

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import matplotlib.pyplot as plt
import pandas as pd

# --- NLTK Setup ---
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('vader_lexicon', download_dir=nltk_data_path)

# --- Initialize Sentiment Analyzer ---
sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

DB_NAME = "mindguard.db"

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  sentiment TEXT,
                  emotions TEXT,
                  response TEXT,
                  timestamp TEXT,
                  mood_rating TEXT)''')
                
    conn.commit()
    conn.close()

init_db()

# Emotion keywords
emotion_dict = {
     "Joy": ["happy", "good", "great", "joyful", "excited", "love", "fantastic", "amazing", "cheerful"],
    "Sadness": ["sad","not", "down", "bad", "not good", "unhappy", "depressed", "unwell", "gloomy"],
    "Anger": ["angry", "mad", "frustrated", "irritated", "annoyed", "furious"],
    "Fear": ["afraid", "scared", "nervous", "worried", "fearful", "anxious", "overwhelmed"],
    "Surprise": ["surprised", "shocked", "amazed", "astonished"],
    "Anxiety" : ["anxious", "nervous", "worried", "overwhelmed"]
}

# --- Emoji to numeric score mapping ---
emoji_mood_map = {
    "😄": 1.0,   # Very Happy
    "🙂": 0.5,   # Happy
    "😐": 0.0,   # Neutral
    "😔": -0.5,  # Sad
    "😢": -1.0,  # Very Sad
    "😡": -1.0,  # Angry
    "😨": -0.7   # Anxious
}


def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
        tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(tok) for tok in tokens]



# --- Detect emotions with better negation handling ---
def detect_emotions(text):
    words = preprocess(text)
    detected = []

    for i, word in enumerate(words):
        for emotion, keywords in emotion_dict.items():
            for kw in keywords:
                if word == kw:
                    # Check for negation immediately before
                    if i > 0 and words[i-1] in ["not", "no", "never"]:
                        # Flip emotion: if positive keyword after negation, count as negative
                        if emotion == "Joy":
                            detected.append("Sadness")
                        elif emotion == "Sadness":
                            detected.append("Joy")
                        # you can add other flips if needed
                        continue
                    detected.append(emotion)
                    break

    # Remove duplicates
    return list(set(detected))


# --- Smarter sentiment calculation ---
def calculate_sentiment(text, mood_rating=None):
    vader_score = sid.polarity_scores(text)['compound']
    emoji_score = emoji_mood_map.get(mood_rating, 0) if mood_rating else 0

    # Give priority to negative emotions detected in text
    emotions = detect_emotions(text)
    if "Sadness" in emotions or "Anger" in emotions or "Anxiety" in emotions:
        combined_score = vader_score  # ignore emoji if text shows negative emotions
    else:
        combined_score = 0.7 * vader_score + 0.3 * emoji_score  # mix normally

    # Convert numeric score to sentiment
    if combined_score > 0.2:
        return "Positive"
    elif combined_score < -0.2:
        return "Negative"
    else:
        return "Neutral"
    
# Generate supportive message
def generate_supportive_message(sentiment, emotions):
    if sentiment == "Negative" or "Sadness" in emotions or "Anxiety" in emotions:
        return "It seems you had a tough day. Try taking a short walk or doing some deep breaths."
    elif sentiment == "Positive" or "Joy" in emotions:
        return "Awesome! Keep enjoying these happy moments!"
    else:
        return "Thanks for sharing! Remember to take small breaks and stay mindful."

# Full analysis function
def analyze_text(text, mood_rating=None):
    emotions = detect_emotions(text)
    sentiment = calculate_sentiment(text, mood_rating)
    response = generate_supportive_message(sentiment, emotions)
    return sentiment, emotions, response

def get_personalized_tips(emotions):
    tips = []
    if "Anxiety" in emotions or "Fear" in emotions:
        tips += ["Try a 5-minute deep breathing exercise", "Write down your worries to clear your mind", "Take a short walk to relax"]
    if "Sadness" in emotions:
        tips += ["Listen to your favorite uplifting music", "Call or text a friend for support", "Write down three things you are grateful for"]
    if "Anger" in emotions:
        tips += ["Take a short break and count to ten", "Try physical activity like a quick walk or stretch", "Write down what made you angry to vent safely"]
    if "Joy" in emotions:
        tips += ["Keep a journal of happy moments", "Share your joy with someone", "Celebrate small victories today"]
    if not tips:
        tips = ["Take small breaks and practice mindfulness."]
    return tips

def generate_mood_card(text, sentiment, emotions, response, filename, tips=None, bg_image_path=None):
    # --- Prepare fonts ---
    try:
        title_font = ImageFont.truetype("static/fonts/Poppins-Bold.ttf", 40)
        subtitle_font = ImageFont.truetype("static/fonts/Poppins-Bold.ttf", 28)
        small_font = ImageFont.truetype("static/fonts/Poppins-Regular.ttf", 20)
    except OSError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # --- Prepare wrapped text lengths ---
    max_width_chars = 60  # chars per line
    wrapped_text = textwrap.wrap(text, width=max_width_chars)
    wrapped_response = textwrap.wrap(response, width=max_width_chars)
    wrapped_tips = []
    if tips:
        for tip in tips:
            wrapped_tips.extend(textwrap.wrap(f"• {tip}", width=max_width_chars))

    # --- Calculate required image height dynamically ---
    base_height = 600
    line_height = 25
    extra_height = (len(wrapped_text) + len(wrapped_response) + len(wrapped_tips)) * line_height
    img_height = base_height + extra_height
    img_width = 800

    # --- Load background image or default color ---
    if bg_image_path and os.path.exists(bg_image_path):
        bg = Image.open(bg_image_path).convert("RGB")
        bg = bg.resize((img_width, img_height))
        img = bg.filter(ImageFilter.GaussianBlur(2))
    else:
        img = Image.new('RGB', (img_width, img_height), color=(245, 245, 250))

    draw = ImageDraw.Draw(img)

    # --- Sentiment Box ---
    sentiment_colors = {"Positive": "#00C49A", "Negative": "#FF6B6B", "Neutral": "#FFD93D"}
    color = sentiment_colors.get(sentiment, "#AAAAAA")
    draw.rectangle([20, 20, img_width-20, 90], fill=color)
    draw.text((30, 30), f"Sentiment: {sentiment}", fill="white", font=title_font)

    # --- Emotions ---
    draw.text((30, 100), f"Emotions: {', '.join(emotions)}", fill="#333333", font=subtitle_font)

    # --- User Entry ---
    y = 140
    draw.text((30, y), "Your Entry:", fill="#333333", font=subtitle_font)
    y += 40
    for line in wrapped_text:
        draw.text((30, y), line, fill="#111111", font=small_font)
        y += line_height

    # --- Supportive Message ---
    y += 20
    draw.text((30, y), "Supportive Message:", fill="#333333", font=subtitle_font)
    y += 40
    for line in wrapped_response:
        draw.text((30, y), line, fill="#111111", font=small_font)
        y += line_height

    # --- Personalized Coping Tips ---
    if wrapped_tips:
        y += 20
        draw.text((30, y), "Coping Tips:", fill="#333333", font=subtitle_font)
        y += 40
        for line in wrapped_tips:
            draw.text((30, y), line, fill="#111111", font=small_font)
            y += line_height

    # --- Save Image ---
    os.makedirs("static/cards", exist_ok=True)
    path = f"static/cards/{filename}"
    img.save(path)

    return path


def calculate_streak():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
        dates = c.fetchall()

    if not dates:
        return 0

    from datetime import datetime, timedelta

    # Convert to just date (ignore time)
    unique_dates = sorted(set([datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S").date() for d in dates]), reverse=True)
    today = datetime.now().date()

    streak = 0
    current_day = today

    for entry_date in unique_dates:
        if entry_date == current_day:
            streak += 1
            current_day -= timedelta(days=1)
        elif entry_date < current_day:
            break

    return streak

def mood_alert(num_days=5):
    """
    Alert the user if their mood is truly declining or consistently negative.
    """
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        # Fetch oldest first for proper trend analysis
        c.execute(f"""
            SELECT timestamp, sentiment 
            FROM entries 
            ORDER BY timestamp ASC 
            LIMIT {num_days}
        """)
        entries = c.fetchall()

    if not entries or len(entries) < 2:
        return None  # Not enough data to analyze trend

    # Map sentiment to numeric score
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    scores = [sentiment_map.get(s[1], 0) for s in entries]

    # Check for declining trend (oldest → newest)
    declining = all(earlier > later for earlier, later in zip(scores, scores[1:]))

    # Check for consistently negative mood
    consistently_negative = all(score < 0 for score in scores)

    if declining or consistently_negative:
        return "⚠️ Your recent mood trend is low. Consider taking a break, journaling, or practicing mindfulness!"

    return None

def weekly_insight():
    """
    Analyze past week's entries and provide insights.
    Returns a summary string.
    """
    import pandas as pd
    from datetime import datetime, timedelta

    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, sentiment, emotions FROM entries ORDER BY timestamp DESC", conn
        )

    if df.empty:
        return "No entries yet to generate weekly insights."

    # Convert timestamp to datetime and extract weekday
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['weekday'] = df['timestamp'].dt.day_name()  # Monday, Tuesday, etc.

    # Map sentiment to numeric
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)

    # Most positive & negative day
    day_sentiment = df.groupby('weekday')['sentiment_score'].mean()
    most_positive_day = day_sentiment.idxmax()
    most_negative_day = day_sentiment.idxmin()

    # Peak emotion per weekday
    df['emotions_list'] = df['emotions'].fillna("").apply(lambda x: x.split(',') if x else [])
    df_exploded = df.explode('emotions_list')
    emotion_counts = df_exploded.groupby('weekday')['emotions_list'].apply(
        lambda x: x.value_counts().idxmax() if not x.empty else None
    )

    # Build summary lines
    summary_lines = ["📊 Weekly Insights:"]
    summary_lines.append(f"- Most positive day: {most_positive_day}")
    summary_lines.append(f"- Lowest mood day: {most_negative_day}")
    summary_lines.append("- Peak emotions per day:")

    for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
        emo = emotion_counts.get(day, None)
        if emo:
            summary_lines.append(f"  • {day}: {emo}")

    return "\n".join(summary_lines)


app = Flask(__name__)
# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/entry", methods=["GET", "POST"])
def entry():
    if request.method == "POST":
        text = request.form['mood_text']
        mood_rating = request.form.get('mood_rating', '😐')  # default to neutral
        print("Mood rating submitted:", mood_rating)
        # --- Analyze text + emoji ---
        sentiment, emotions, response_msg = analyze_text(text, mood_rating)
        tips = get_personalized_tips(emotions)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Generate shareable mood card ---
        card_filename = f"moodcard_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        card_path = generate_mood_card(
            text, sentiment, emotions, response_msg, card_filename,
            tips= tips,
            bg_image_path="static/bg1.jpg"
        )

        # --- Save to DB ---
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO entries (text, sentiment, emotions, response, timestamp, mood_rating) VALUES (?, ?, ?, ?, ?, ?)",
                (text, sentiment, ",".join(emotions), response_msg, timestamp, mood_rating)
            )

        return render_template(
            "analysis.html",
            text=text,
            sentiment=sentiment,
            emotions=emotions,
            response=response_msg,
            card_path=card_path,
            mood_rating=mood_rating
        )

    return render_template("entry.html")

@app.route("/dashboard")
def dashboard():
    # Fetch entries
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT text, sentiment, emotions, timestamp, mood_rating FROM entries ORDER BY timestamp DESC")
        data = c.fetchall() 

    alert_msg = mood_alert(num_days=5)
    streak = calculate_streak()
    insight_summary = weekly_insight()
    # Plot mood trend
    df = pd.DataFrame(data, columns=["text", "sentiment", "emotions", "timestamp", "mood_rating"])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        sentiment_colors = {"Positive": "#00C49A", "Negative": "#FF6B6B", "Neutral": "#FFD93D"}
        df['color'] = df['sentiment'].map(sentiment_colors)

        plt.figure(figsize=(12,4))
        plt.scatter(df['timestamp'], [1]*len(df), c=df['color'], s=100, alpha=0.9, edgecolors='k')
        plt.plot(df['timestamp'], [1]*len(df), color="#6C63FF", alpha=0.5, linewidth=2)
        plt.yticks([])
        plt.title("Mood Trend Over Time", fontsize=16, weight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("static/mood_trend.png", transparent=True)
        plt.close()

    # Count most frequent emotions
    from collections import Counter
    all_emotions = []
    for entry in data:
        if entry[2]:  # emotions field
            all_emotions += entry[2].split(',')
    freq_emotions = Counter(all_emotions)

    # Plot emotions frequency
    if freq_emotions:
        plt.figure(figsize=(8,4))
        plt.bar(freq_emotions.keys(), freq_emotions.values(), color="#6C63FF", alpha=0.8)
        plt.title("Most Frequent Emotions", fontsize=14, weight='bold')
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("static/emotions_trend.png", transparent=True)
        plt.close()

    return render_template("dashboard.html", entries=data, streak=streak, alert_msg=alert_msg, insight_summary=insight_summary)

# --- CSV Export ---
@app.route("/export_csv")
def export_csv():
    # Get entries
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(
            "SELECT text, sentiment, emotions, timestamp FROM entries ORDER BY id DESC", conn
        )
    
    # Ensure timestamp is string
    df['timestamp'] = df['timestamp'].astype(str)
    
    # Create CSV in memory
    mem = io.StringIO()
    df.to_csv(mem, index=False)  # let pandas handle writing properly
    mem.seek(0)
    
    return send_file(
        io.BytesIO(mem.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='mood_history.csv'
    )

if __name__ == "__main__":
    app.run(debug=True)
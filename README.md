# Mind Guard AI

Mind Guard AI is an intelligent mood tracking and mental wellness assistant that analyzes user text and emoji inputs to detect emotions and sentiment, providing supportive messages, personalized coping tips, and visual mood trends.

---

## 🚀 Features

- **Sentiment Analysis**: Classifies text as Positive, Neutral, or Negative using **VADER Sentiment Analyzer**.
- **Emotion Detection**: Detects emotions like Joy, Sadness, Anger, Fear, Surprise, and Anxiety using keyword-based NLP with **NLTK**.
- **Emoji Mood Mapping**: Combines emojis with text sentiment to calculate a **combined mood score**.
- **Supportive Messages**: Generates personalized supportive messages based on detected sentiment and emotions.
- **Coping Tips**: Suggests practical tips to improve mood based on emotional analysis.
- **Mood Cards**: Creates visually appealing shareable images summarizing entries, sentiment, emotions, and tips.
- **Trend Analysis & Insights**: Tracks user mood over time, visualizing mood trends and emotion frequencies using **Matplotlib** and **Pandas**.
- **Data Persistence**: Stores user entries in **SQLite database** for long-term tracking.
- **Export CSV**: Allows users to download their mood history for personal use.

## 🛠 Technologies & AI Models

- **Python 3**  
- **Flask** – Web framework  
- **SQLite** – Database  
- **NLTK** – Natural Language Toolkit for text processing  
- **VADER Sentiment Analysis** – Sentiment classification  
- **Matplotlib & Pandas** – Data visualization  
- **Pillow (PIL)** – Mood card generation  

**AI Models & Techniques:**
- **VADER** for sentiment analysis  
- **Keyword-based NLP** for emotion detection  
- **Emoji mapping** for mood scoring  
- **Rule-based supportive message generator**  

---

## 💡 How It Works

1. User enters text and selects a mood emoji.  
2. The system preprocesses the text (tokenization, lemmatization) and detects emotions.  
3. Sentiment is calculated combining text sentiment (VADER) and emoji score.  
4. Supportive messages and personalized coping tips are generated.  
5. Mood card image is created summarizing the entry.  
6. Entries are stored in SQLite and can be visualized in trends and exported as CSV.  

---

## 🎥 Demo Video

- **Duration:** 8–12 minutes  
- **Sections:**  
  1. Problem explanation  
  2. Code walkthrough  
  3. Live execution  
  4. Challenges faced and debugging  

---

## ⚡ Future Improvements

- Integrate **GPT or other LLM** for dynamic response generation.  
- Implement **real-time chat interface** with mood feedback.  
- Add **mobile app support** with notifications and reminders.  

---

## 📜 License

This project is made for educational and hackathon purposes.

---

## 📂 Project Structure

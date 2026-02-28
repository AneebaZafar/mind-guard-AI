import sqlite3
from datetime import datetime, timedelta

DB_NAME = "mindguard.db"

# Create 5 entries with strictly declining mood
base_time = datetime.now()
entries = [
    ('Feeling great!', 'Positive', 'Joy', 'Awesome! Keep enjoying these happy moments!', (base_time - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"), '😄'),
    ('Feeling okay', 'Neutral', 'Joy', 'Thanks for sharing! Remember to take small breaks.', (base_time - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"), '😐'),
    ('A bit sad today', 'Negative', 'Sadness', 'It seems you had a tough day.', (base_time - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"), '😔'),
    ('Feeling worse', 'Negative', 'Sadness', 'It seems you had a tough day.', (base_time - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"), '😢'),
    ('Really bad day', 'Negative', 'Sadness', 'It seems you had a tough day.', (base_time - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"), '😢'),
]

# Insert into database
conn = sqlite3.connect(DB_NAME)
c = conn.cursor()
for e in entries:
    c.execute("INSERT INTO entries (text, sentiment, emotions, response, timestamp, mood_rating) VALUES (?, ?, ?, ?, ?, ?)", e)
conn.commit()
conn.close()
print("Test entries inserted!")
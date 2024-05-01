import sqlite3

conn = sqlite3.connect('sentiment_analysis.db')

cur = conn.cursor()
# Text table
cur.execute('''
CREATE TABLE IF NOT EXISTS text_analysis (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    prediction TEXT NOT NULL
)
''')
# Audio table
cur.execute('''
CREATE TABLE IF NOT EXISTS audio_analysis (
    id INTEGER PRIMARY KEY,
    audio TEXT NOT NULL,
    prediction TEXT NOT NULL
)
''')

# Commit changes and close the connection
conn.commit()
conn.close()


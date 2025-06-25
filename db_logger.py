# db_logger.py
import sqlite3

def init_db():
    conn = sqlite3.connect("dialogue_log.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            user_text TEXT,
            reply_text TEXT,
            text_emotion TEXT,
            voice_emotion TEXT,
            intent TEXT,
            suggestion TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_log(timestamp, user_text, reply_text, text_emotion, voice_emotion, intent, suggestion):
    conn = sqlite3.connect("dialogue_log.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO logs (
            timestamp, 
            user_text, 
            reply_text, 
            text_emotion, 
            voice_emotion,  
            intent, 
            suggestion
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, user_text, reply_text, text_emotion, voice_emotion, intent, suggestion))
    conn.commit()
    conn.close()
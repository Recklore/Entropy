import os
import sqlite3
import pandas as pd


QUESTIONS_FILE_PATH = "./data/SL/sl_questions.csv"

DATABASE_DIR = "./data/assessment_database"
DATABASE_PATH = DATABASE_DIR + "/Questions.db"

os.makedirs(DATABASE_DIR, exist_ok=True)

questions_df = pd.read_csv(QUESTIONS_FILE_PATH)

with sqlite3.connect(DATABASE_PATH) as conn:
    cur = conn.cursor()

    cur.execute(
        """CREATE TABLE IF NOT EXISTS questions (
        question_number INTEGER PRIMARY KEY,
        skill_name TEXT,
        question_text TEXT,
        option_1 TEXT, option_2 TEXT,
        option_3 TEXT, option_4 TEXT,
        correct_option INTEGER
        )"""
    )

    for _, row in questions_df.iterrows():
        cur.execute(
            """INSERT INTO questions 
               (question_number, skill_name, question_text, option_1, option_2, option_3, option_4, correct_option)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(row["question_number"]),
                row["skill_name"],
                row["question_text"],
                row["option_1"],
                row["option_2"],
                row["option_3"],
                row["option_4"],
                int(row["correct_option"]),
            ),
        )

    conn.commit()

print("Assessment Database Created")

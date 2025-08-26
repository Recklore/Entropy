import sqlite3
import random
import ollama
from typing import Annotated
from pydantic import BaseModel, Field
from lightrag import QueryParam


class Questions(BaseModel):
    class Question(BaseModel):
        question_text: str
        options: Annotated[list[str], Field(min_items=4, max_items=4)]
        correct_answer: int
        skill_name: str

    questions: list[Question]


async def retrieve_content(rag, skill_name):

    prompt = f"""
    Provide a comprehensive and technically precise explanation of '{skill_name}', suitable for a university student studying statistical learning. Use the retrieved context to structure your answer into the following sections:

    1.  **Fundamental Concept & Purpose**:
        - Clearly define what '{skill_name}' is.
        - Explain its primary purpose and why it is important in the context of statistical or machine learning.

    2.  **Technical Breakdown & Mechanism**:
        - Describe the core mechanism or process.
        - For a model (e.g., Linear Regression), explain how it is estimated or trained.
        - For a concept (e.g., Bias-Variance Tradeoff), explain its components and their relationship.
        - For a method (e.g., K-Fold Cross Validation), explain the step-by-step procedure.

    3.  **Application & Interpretation**:
        - Discuss the practical applications. When and where is this used?
        - Explain how to interpret the results or outcomes associated with '{skill_name}'. This can include model metrics, diagnostic plots, or the implications of a concept.

    4.  **Challenges & Key Considerations**:
        - Outline common challenges, limitations, or potential pitfalls.
        - Discuss any important assumptions, trade-offs, or nuances to be aware of when applying or considering '{skill_name}'.

    Please synthesize a clear, well-structured, and complete response based on the information provided.
    """

    try:
        response = await rag.aquery(
            prompt,
            param=QueryParam(
                mode="mix",
                top_k=10,
                response_type="Multiple Paragraphs",
            ),
        )

        return response
    except Exception as e:
        return f"Sorry, I was unable to retrieve the content for this skill due to error: {e}"


def generate_assessment(
    skill_name=None,
    model_name="mistral:7b-instruct",
    database_path="./data/assessment_database/Questions.db",
    num_q=None,
):

    with sqlite3.connect(database_path) as conn:
        cur = conn.cursor()

        if skill_name:
            cur.execute(
                "SELECT question_text, option_1, option_2, option_3, option_4, correct_option"
                " FROM questions WHERE skill_name = ? ORDER BY RANDOM() LIMIT 2",
                (skill_name,),
            )

            rows = cur.fetchall()

            if not rows:
                return f"No questions found for skill: {skill_name}"

            prompt = (
                f"Here are some example questions for the skill '{skill_name}':\n\n"
                + ",".join(str(q) for q in rows)
                + f"\n\nUsing the above as examples, generate {num_q if num_q else "10"} new questions for the same skill"
            )

        else:
            cur.execute("SELECT DISTINCT skill_name FROM questions")
            all_skills = [row[0] for row in cur.fetchall()]

            chosen_skills = random.sample(all_skills, (num_q if num_q else 20))

            questions = []
            for skill in chosen_skills:
                cur.execute(
                    "SELECT question_text, option_1, option_2, option_3, option_4, correct_option"
                    " FROM questions WHERE skill_name=? LIMIT 1 OFFSET ?",
                    (skill, random.randint(0, 4)),
                )
                row = cur.fetchone()

                if row:
                    questions.append((skill,) + row)

            prompt = (
                f"Here is one example question for each of {num_q if num_q else "20"} different skills:\n\n"
                + ",".join(f"{question}" for question in questions)
                + f"\n\nUsing the above as examples, generate {num_q if num_q else "20"} new diverse questions across these skills"
            )
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=Questions.model_json_schema(),
        options={"temperature": 0.4},
    )

    return response["message"]["content"]

import json
import sqlite3
import random
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
    You are an expert AI Tutor specializing in statistical learning and machine learning. Your task is to generate a comprehensive, technically precise, and well-structured educational guide on the topic of '{skill_name}'.

    The target audience is a university student, so the explanation must be clear, rigorous, and build strong intuition.

    Use the provided context to generate the response, but synthesize it into the following structure. **The entire output must be in Markdown format.**

    ---

    ## {skill_name}: A Comprehensive Guide

    ### 1. Fundamental Concept & Purpose
    - **Definition:** Clearly define what '{skill_name}' is in the context of statistical learning.
    - **Purpose and Importance:** Explain its primary goal. Why is it used? What fundamental problem does it solve or what question does it answer?
    - **Intuitive Analogy:** Provide a simple, real-world analogy to help build intuition around the core idea.

    ### 2. Core Mathematical and Algorithmic Details
    - **Mechanism:** Describe the core mathematical or algorithmic process. Use LaTeX for all mathematical notations, equations, and symbols (e.g., enclose inline math in `$` and display equations in `$$`).
    - **Step-by-Step Example:** If applicable (e.g., for a method like K-Fold CV or an algorithm like Gradient Descent), provide a simple, step-by-step numerical example to illustrate the process.
    - **Key Parameters:** Explain any important parameters, hyperparameters, or components involved.

    ### 3. Practical Application & Interpretation
    - **Use Cases:** Discuss common practical applications and real-world scenarios where '{skill_name}' is applied.
    - **Interpreting Results:** Explain how to interpret the outcomes. This could include model coefficients, performance metrics (e.g., $R^2$, AUC, F1-score), diagnostic plots (e.g., residual plots), or the implications of a concept like the bias-variance tradeoff.

    ### 4. Assumptions, Strengths, and Limitations
    - **Key Assumptions:** List and explain the critical assumptions that must hold for '{skill_name}' to be applied correctly and effectively.
    - **Strengths:** What are the main advantages of using this method or concept?
    - **Limitations & Pitfalls:** Outline common challenges, limitations, or potential pitfalls. When might it perform poorly or give misleading results?

    ---

    Ensure the final output is a single, coherent, and well-formatted Markdown document that directly follows this structure.
    """

    try:
        response = await rag.aquery(
            prompt,
            param=QueryParam(
                mode="mix",
                top_k=10,
                response_type="Multiple Paragraphs",
                enable_rerank=False,
            ),
        )

        return response
    except Exception as e:
        return f"Sorry, I was unable to retrieve the content for this skill due to error: {e}"


def generate_assessment(
    skill_name=None,
    client=None,
    model_name="llama-3.3-70b-versatile",
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

    if client is None:
        raise ValueError("Groq client is required for assessment generation.")

    system_prompt = (
        "You are a precise JSON generator. Return ONLY valid JSON with the following schema: "
        f"{Questions.model_json_schema()}. Do not include markdown or commentary."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    payload = json.loads(content)
    Questions.model_validate(payload)
    return json.dumps(payload)

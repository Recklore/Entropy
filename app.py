import streamlit as st
import os
import json
import time
import random
import sqlite3
import asyncio
import torch
import pandas as pd
import torch.nn as nn
from google import genai
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from groq import Groq

from src.RAG.retrieve import generate_assessment, retrieve_content

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)

EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")

DKT_PATH = "./models/DKT_model.pt"
DQN_PATH = "./models/DQN_agent.pt"
LIGHTRAG_WORKING_DIR = "./data/lightrag_database"
SKILLS_PATH = "./data/SL/sl_skills.csv"


@st.cache_resource
def load_models(DKTplus_path="./models/DKT_model.pt", DQN_path="./models/DQN_agent.pt"):
    if os.path.exists(DKTplus_path) and os.path.exists(DQN_path):
        dkt = torch.jit.load(DKTplus_path, map_location=DEVICE)
        dqn = torch.jit.load(DQN_path, map_location=DEVICE)
        dkt.eval()
        dqn.eval()
        return dkt, dqn
    else:
        st.error("Models path invalid")
        return None, None


@st.cache_resource
def load_skills(path=SKILLS_PATH):
    skill_to_index = pd.read_csv(path, header=None, skiprows=1)
    skill_to_index = {skill: i + 1 for i, skill in enumerate(skill_to_index[1].squeeze().tolist())}
    index_to_skill = {v: k for k, v in skill_to_index.items()}
    return skill_to_index, index_to_skill


def embed_texts(texts, client, model_name):
    if isinstance(texts, str):
        texts = [texts]
    response = client.models.embed_content(model=model_name, contents=texts)
    return [embedding.values for embedding in response.embeddings]


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    if history_messages is None:
        history_messages = []
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


@st.cache_resource
def initailise_rag_sync(working_dir, llm_model, embed_model, embed_dim):
    asyncio.set_event_loop(st.session_state.loop)
    return st.session_state.loop.run_until_complete(
        initailise_rag(working_dir, llm_model, embed_model, embed_dim)
    )


async def initailise_rag(working_dir, llm_model, embed_model, embed_dim):
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(embed_dim),
            max_token_size=8192,
            func=lambda texts: embed_texts(texts, GEMINI_CLIENT, embed_model),
        ),
        rerank_model_func=None,
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def display_dashboard(rag, skill_to_index):
    st.header("Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Explore Topics")
            st.write("Choose a topic to study at your own pace.")

            topics = {
                "Foundations of Statistical Learning": [
                    "Bias-Variance Decomposition",
                    "Overfitting vs Underfitting",
                    "K-Fold Cross Validation",
                    "Fairness in ML",
                ],
                "Linear Models": [
                    "Simple Linear Regression",
                    "Multiple Linear Regression",
                    "Interaction Terms",
                    "Collinearity and VIF",
                ],
                "Advanced Regression and Regularization": [
                    "Ridge Regression and Lasso",
                    "Piecewise Polynomials",
                    "Natural and Smoothing Splines",
                ],
                "Tree-Based Methods and Ensembles": [
                    "Bagging and Model Averaging",
                    "Bootstrap Methods",
                ],
                "Unsupervised and Prototype Methods": ["Curse of Dimensionality"],
            }

            for topic, skills in topics.items():
                with st.expander(topic):
                    for skill in skills:
                        if st.button(f"Study: {skill}", key=f"study_{skill}"):
                            st.session_state.view = "studying"
                            st.session_state.selected_skill = skill
                            st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("Personalized Learning Path")
            st.write("Take an initial assessment to identify your knowledge gaps and receive a tailored learning plan.")
            if st.button("🚀 Start Initial Assessment"):
                st.session_state.view = "assessment"
                st.session_state.assessment_started = False  # Reset for assessment flow
                st.rerun()


def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
    <style>
        @keyframes glow {
            0% { box-shadow: 0 0 3px #BF00FF, 0 0 5px #BF00FF, 0 0 8px #BF00FF; }
            50% { box-shadow: 0 0 10px #BF00FF, 0 0 15px #BF00FF, 0 0 20px #BF00FF; }
            100% { box-shadow: 0 0 3px #BF00FF, 0 0 5px #BF00FF, 0 0 8px #BF00FF; }
        }
        .start-button-container .stButton>button {
            animation: glow 1.5s infinite;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 12px 18px;
            border: 1px solid #BF00FF;
            background-color: transparent;
            color: #BF00FF;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #BF00FF;
            color: white;
        }
        .stButton>button:disabled {
            border-color: #555;
            color: #555;
        }
        .current-question-button button {
            border: 2px solid #BF00FF !important; /* Purple border for current question */
            box-shadow: 0 0 8px #BF00FF;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --- Session State Initialization ---
    if "student_name" not in st.session_state:
        st.session_state.student_name = ""
    if "view" not in st.session_state:
        st.session_state.view = "start"  # start, login, dashboard, studying, assessment
    if "assessment_started" not in st.session_state:
        st.session_state.assessment_started = False
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "student_answers" not in st.session_state:
        st.session_state.student_answers = []
    if "question_status" not in st.session_state:
        st.session_state.question_status = []  # 'answered', 'skipped', 'not_visited'
    if "q_history" not in st.session_state:
        st.session_state.q_history = []
    if "r_history" not in st.session_state:
        st.session_state.r_history = []
    if "t_history" not in st.session_state:
        st.session_state.t_history = []
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "show_content" not in st.session_state:
        st.session_state.show_content = False
    if "selected_skill" not in st.session_state:
        st.session_state.selected_skill = ""
    if "content" not in st.session_state:
        st.session_state.content = ""
    if "mastery" not in st.session_state:
        st.session_state.mastery = None
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()

    # --- Load Models and Data ---
    dkt, dqn = load_models(DKT_PATH, DQN_PATH)
    skill_to_index, index_to_skill = load_skills(SKILLS_PATH)
    rag = initailise_rag_sync(LIGHTRAG_WORKING_DIR, GROQ_MODEL_NAME, GEMINI_EMBED_MODEL, EMBEDDING_DIM)

    # --- Sidebar ---
    if st.session_state.mastery is not None:
        st.sidebar.title("🎓 Skill Mastery")
        mastery_df = pd.DataFrame(
            {
                "Skill": [index_to_skill[i + 1] for i in range(len(st.session_state.mastery))],
                "Mastery": st.session_state.mastery.cpu().numpy(),
            }
        )
        st.sidebar.bar_chart(mastery_df.set_index("Skill"))

    # --- Main App Logic ---
    if st.session_state.view == "start":
        st.markdown(
            """
            <div style="text-align: center; padding-top: 5rem;">
                <h1 style="font-size: 3.5rem;">Welcome to Entropy-Learning</h1>
                <p style="font-size: 1.2rem; max-width: 600px; margin: auto; margin-bottom: 2rem;">
                    A personalized platform designed to help you master supervised learning. 
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _, col2, _ = st.columns([2, 1, 2])
        with col2:
            st.markdown('<div class="start-button-container">', unsafe_allow_html=True)
            if st.button("Start Learning", use_container_width=True):
                st.session_state.view = "login"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.view == "login":
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.header("Enter Your Name")
            st.session_state.student_name = st.text_input("Please enter your name to begin:", key="name_input")
            if st.button("Continue", use_container_width=True):
                if st.session_state.student_name:
                    st.session_state.view = "dashboard"
                    st.rerun()
                else:
                    st.error("Please enter your name.")

    elif st.session_state.view == "dashboard":
        st.write(f"Hello, {st.session_state.student_name}!")
        display_dashboard(rag, skill_to_index)

    elif st.session_state.view == "studying":
        st.header(f"Studying: {st.session_state.selected_skill}")
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.view = "dashboard"
            st.rerun()
        with st.spinner("Retrieving content..."):
            asyncio.set_event_loop(st.session_state.loop)
            content = st.session_state.loop.run_until_complete(retrieve_content(rag, st.session_state.selected_skill))
            st.markdown(content)

    elif st.session_state.view == "assessment":
        if not st.session_state.assessment_started:
            st.header("Initial Assessment")
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.view = "dashboard"
                st.rerun()
            with st.spinner("Generating your first assessment..."):
                first_assessment = json.loads(
                    generate_assessment(client=GROQ_CLIENT, model_name=GROQ_MODEL_NAME, num_q=5)
                )["questions"]
                st.session_state.questions = first_assessment
                st.session_state.student_answers = [None] * len(first_assessment)
                st.session_state.question_status = ["not_visited"] * len(first_assessment)
                st.session_state.assessment_started = True
                st.rerun()

        if st.session_state.questions:
            main_col, nav_col = st.columns([3, 1])
            with main_col:
                st.header("Initial Assessment")
            with nav_col:
                st.header("Questions")

            if st.button("⬅️ Back to Dashboard"):
                st.session_state.view = "dashboard"
                st.session_state.assessment_started = False
                st.session_state.questions = []
                st.session_state.student_answers = []
                st.session_state.question_status = []
                st.rerun()

            with nav_col:
                q_cols = st.columns(5)
                for i in range(len(st.session_state.questions)):
                    col = q_cols[i % 5]
                    status = st.session_state.question_status[i]
                    label = f"Q{i+1}"
                    button_type = "secondary"
                    if status == "answered":
                        button_type = "primary"

                    if i == st.session_state.current_question_index:
                        with col:
                            st.markdown('<div class="current-question-button">', unsafe_allow_html=True)
                            if st.button(
                                label,
                                key=f"nav_{i}",
                                on_click=lambda i=i: st.session_state.update(current_question_index=i),
                                type=button_type,
                                use_container_width=True,
                            ):
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        if col.button(
                            label,
                            key=f"nav_{i}",
                            on_click=lambda i=i: st.session_state.update(current_question_index=i),
                            type=button_type,
                            use_container_width=True,
                        ):
                            st.rerun()
                st.write("---")
                if st.button("Finish Assessment", use_container_width=True, type="primary"):
                    st.session_state.view = "results"
                    st.rerun()

            with main_col:
                idx = st.session_state.current_question_index
                question = st.session_state.questions[idx]
                st.write(f"**Question {idx + 1} of {len(st.session_state.questions)}**")
                st.write(f"_{question['skill_name']}_")
                st.subheader(question["question_text"])
                options = question["options"]
                option_labels = [f"{chr(97+i)}. {opt}" for i, opt in enumerate(options)]
                current_answer = st.session_state.student_answers[idx]
                student_answer_index = st.radio(
                    "Your answer:",
                    range(len(options)),
                    format_func=lambda x: option_labels[x],
                    index=current_answer,
                    key=f"q_{idx}",
                )
                if student_answer_index != current_answer:
                    st.session_state.student_answers[idx] = student_answer_index
                    st.session_state.question_status[idx] = "answered"
                    st.rerun()

                st.write("<br>", unsafe_allow_html=True)
                left_nav, right_nav = st.columns([1, 1])
                with left_nav:
                    if st.button("Clear Answer", use_container_width=True):
                        st.session_state.student_answers[idx] = None
                        st.session_state.question_status[idx] = "not_visited"
                        st.rerun()
                with right_nav:
                    nav_buttons = st.columns(3)
                    if nav_buttons[0].button("⬅️ Previous", use_container_width=True, disabled=idx == 0):
                        st.session_state.current_question_index -= 1
                        st.rerun()
                    if nav_buttons[1].button("Skip", use_container_width=True):
                        st.session_state.question_status[idx] = "skipped"
                        if idx < len(st.session_state.questions) - 1:
                            st.session_state.current_question_index += 1
                            st.rerun()
                    if nav_buttons[2].button(
                        "Next ➡️",
                        use_container_width=True,
                        disabled=idx == len(st.session_state.questions) - 1,
                    ):
                        st.session_state.current_question_index += 1
                        st.rerun()

    elif st.session_state.view == "results":
        st.header("Assessment Results")
        with st.spinner("Analyzing results and recommending a skill..."):
            for i, answer_idx in enumerate(st.session_state.student_answers):
                if answer_idx is not None:
                    question = st.session_state.questions[i]
                    question_skill = int(skill_to_index.get(question["skill_name"]))
                    is_correct = float((answer_idx + 1) == question["correct_answer"])
                    st.session_state.q_history.append(question_skill)
                    st.session_state.r_history.append(is_correct)
                    st.session_state.t_history.append(10.0)
            if not st.session_state.q_history:
                st.warning("You did not answer any questions. Please go back to the dashboard.")
                if st.button("Back to Dashboard"):
                    st.session_state.view = "dashboard"
                    st.rerun()
                return
            q_tensor = torch.tensor([st.session_state.q_history], dtype=torch.long).to(DEVICE)
            r_tensor = torch.tensor([st.session_state.r_history], dtype=torch.long).to(DEVICE)
            t_tensor = torch.tensor([st.session_state.t_history], dtype=torch.float).to(DEVICE)
            with torch.no_grad():
                student_masteries = torch.sigmoid(dkt(q_tensor, r_tensor, t_tensor)[0, -1])
                st.session_state.mastery = student_masteries
                selected_skill_index = torch.argmax(dqn(student_masteries.unsqueeze(0)), dim=1).item()
                st.session_state.selected_skill = index_to_skill[selected_skill_index]
            st.success(
                f"Based on your assessment, the recommended skill for you is: **{st.session_state.selected_skill}**"
            )
        st.header(f"Recommended Content: {st.session_state.selected_skill}")
        with st.spinner("Retrieving content..."):
            asyncio.set_event_loop(st.session_state.loop)
            content = st.session_state.loop.run_until_complete(retrieve_content(rag, st.session_state.selected_skill))
            st.session_state.content = content
        st.markdown(st.session_state.content)
        if st.button("Start Skill Assessment"):
            with st.spinner(f"Generating assessment for {st.session_state.selected_skill}..."):
                skill_assessment = json.loads(
                    generate_assessment(
                        client=GROQ_CLIENT,
                        model_name=GROQ_MODEL_NAME,
                        skill_name=st.session_state.selected_skill,
                        num_q=2,
                    )
                )["questions"]
                st.session_state.questions = skill_assessment
                st.session_state.student_answers = [None] * len(skill_assessment)
                st.session_state.question_status = ["not_visited"] * len(skill_assessment)
                st.session_state.current_question_index = 0
                st.session_state.view = "assessment"
                st.session_state.assessment_started = True
                st.rerun()


if __name__ == "__main__":
    main()

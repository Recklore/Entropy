import os
import ast
import time
import random
import ollama
import sqlite3
import asyncio
import torch
import pandas as pd
import torch.nn as nn
from dotenv import load_dotenv
from pydantic import BaseModel, conlist
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from src.RAG.retrieve import generate_assessment, retrieve_content

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLEAR_COMMAND = "cls" if os.name == "nt" else "clear"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

OLLAMA_HOST = os.getenv("LLM_BINDING_HOST")
RETRIEVAL_LLM_MODEL = os.getenv("RETRIEVAL_LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")

DKT_PATH = "./models/DKT_model.pt"
DQN_PATH = "./models/DQN_agent.pt"
LIGHTRAG_WORKING_DIR = "./data/lightrag_database"
SKILLS_PATH = "./data/SL/sl_skills.csv"


def load_models(DKTplus_path="./models/DKT_model.pt", DQN_path="./models/DQN_agent/pt"):

    if os.path.exists(DKTplus_path) and os.path.exists(DQN_path):
        dkt = torch.jit.load(DKTplus_path, map_location=DEVICE)
        dqn = torch.jit.load(DQN_path, map_location=DEVICE)

        dkt.eval()
        dqn.eval()

        return dkt, dqn

    else:
        print("Models path invalid")

        return None


async def initailise_rag(working_dir, llm_model, embed_model, embed_dim, ollama_host):

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={
            "host": ollama_host,
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(embed_dim),
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embed_model,
                host=ollama_host,
            ),
        ),
        rerank_model_func=None,
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def load_skills(path=SKILLS_PATH):
    skill_to_index = pd.read_csv(SKILLS_PATH, header=None, skiprows=1)
    skill_to_index = {skill: i + 1 for i, skill in enumerate(skill_to_index[1].squeeze().tolist())}
    index_to_skill = {v: k for k, v in skill_to_index.items()}

    return skill_to_index, index_to_skill


def countdown(sec=10):
    for t in range(sec, 0, -1):
        os.system(CLEAR_COMMAND)
        print(f"Assessment starts in: {t}s", end="", flush=True)
        time.sleep(0.999)
    os.system(CLEAR_COMMAND)


async def main():

    os.system(CLEAR_COMMAND)
    print("Welcome to Entropi-Learning\n\n")

    print("Loading models...")
    dkt, dqn = load_models(DKT_PATH, DQN_PATH)

    print("Initailising RAG mechanism...")
    rag = await initailise_rag(LIGHTRAG_WORKING_DIR, RETRIEVAL_LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM, OLLAMA_HOST)

    print("Loading skill list...")
    skill_to_index, index_to_skill = load_skills(SKILLS_PATH)

    student_name = input("\nPlease enter your name: ")

    os.system(CLEAR_COMMAND)
    print(f"Generating {student_name}'s first assessment...", end="", flush=True)
    first_assestment = ast.literal_eval(generate_assessment(num_q=5))["questions"]

    q, r, t = [], [], []
    countdown(5)
    for i, question in enumerate(first_assestment, 1):

        print(f"Q{i}. {question["question_text"]}")
        print(
            f"a. {question["options"][0]}\n"
            f"b. {question["options"][1]}\n"
            f"c. {question["options"][2]}\n"
            f"d. {question["options"][3]}\n\n"
        )

        start_time = time.time()

        student_answer = input("Your answer(a, b, c, d): ").lower()
        while True:
            if student_answer not in ["a", "b", "c", "d"]:
                student_answer = input("Invalid choice, try again: ")
            else:
                break

        end_time = time.time()

        question_skill = int(skill_to_index.get(question["skill_name"]))

        if not question_skill:
            q.append(int(0))
            r.append(float(0))
            t.append(float(0))
        else:
            q.append(int(question_skill))
            r.append(float((ord(student_answer) - 96) == question["correct_answer"]))
            t.append(float(round(end_time - start_time, 3)))

        os.system(CLEAR_COMMAND)

    while True:
        print("Processing student response...")
        q_tensor = torch.tensor([q], dtype=torch.long).to(DEVICE)
        r_tensor = torch.tensor([r], dtype=torch.long).to(DEVICE)
        t_tensor = torch.tensor([t], dtype=torch.float).to(DEVICE)

        q.clear()
        r.clear()
        t.clear()

        os.system(CLEAR_COMMAND)
        print("Getting skill mastries...")
        with torch.no_grad():
            student_masteries = torch.sigmoid(dkt(q_tensor, r_tensor, t_tensor)[0, -1])

        os.system(CLEAR_COMMAND)
        print("Prediction best next skill to study...")
        with torch.no_grad():
            selected_skill = index_to_skill[torch.argmax(dqn(student_masteries.unsqueeze(0)), dim=1).item()]

        os.system(CLEAR_COMMAND)
        print(f"Dear {student_name}, the best next skill to you is {selected_skill}\n\n")
        while True:
            student_response = input("Do you want to continue (yes or no): ").lower()
            if student_response not in ["yes", "no"]:
                student_response = input("Invalid choice, try again: ")
            elif student_response == "no":
                os._exit(0)
            elif student_response == "yes":
                break

        os.system(CLEAR_COMMAND)
        print("Getting skill content...", end="", flush=True)
        content = await retrieve_content(rag, selected_skill)
        os.system(CLEAR_COMMAND)

        print(content)

        while True:
            student_response = input("Enter 'ready' to move to the assessment: ").lower()
            if student_response != "ready":
                student_response = input("Invalid choice, try again: ")
            else:
                break

        os.system(CLEAR_COMMAND)
        print(f"Generating {student_name}'s assessment for skill {selected_skill}...", end="", flush=True)
        skill_assestment = ast.literal_eval(generate_assessment(skill_name=selected_skill, num_q=2))["questions"]

        countdown(5)
        for i, question in enumerate(skill_assestment, 1):

            print(f"Q{i}. {question["question_text"]}")
            print(
                f"a. {question["options"][0]}\n"
                f"b. {question["options"][1]}\n"
                f"c. {question["options"][2]}\n"
                f"d. {question["options"][3]}\n\n"
            )

            start_time = time.time()

            student_answer = input("Your answer(a, b, c, d): ").lower()
            while True:
                if student_answer not in ["a", "b", "c", "d"]:
                    student_answer = input("Invalid choice, try again: ")
                else:
                    break

            end_time = time.time()

            q.append(int(skill_to_index.get(question["skill_name"])))
            r.append(float((ord(student_answer) - 96) == question["correct_answer"]))
            t.append(float(round(end_time - start_time, 3)))

            os.system(CLEAR_COMMAND)


if __name__ == "__main__":
    asyncio.run(main())

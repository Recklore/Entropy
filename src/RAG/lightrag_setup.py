import os
import re
import asyncio
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from google import genai
from groq import Groq


load_dotenv()

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")


GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)

WORKING_DIR = "./data/lightrag_database"
MD_FILE_PATH = "./data/SL/sl.md"


def load_document(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            document = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found")

    parts = re.split(r"(Chapter\s+\d+:)", document)
    if len(parts) > 1:
        chapters = [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
        if parts[0].strip():
            chapters.insert(0, parts[0].strip())
        return [d.strip() for d in chapters if d.strip()]

    return [document.strip()]


async def rag_setup():
    try:
        print("Initialising LightRAG")

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

        def embed_texts(texts, client, model_name):
            if isinstance(texts, str):
                texts = [texts]
            response = client.models.embed_content(model=model_name, contents=texts)
            return [embedding.values for embedding in response.embeddings]

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            llm_model_name=GROQ_MODEL_NAME,
            embedding_func=EmbeddingFunc(
                embedding_dim=int(EMBEDDING_DIM),
                max_token_size=8192,
                func=lambda texts: embed_texts(texts, GEMINI_CLIENT, GEMINI_EMBED_MODEL),
            ),
            graph_storage="Neo4JStorage",
            vector_storage="FaissVectorDBStorage",
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        chapters = load_document(MD_FILE_PATH)

        for i, doc in enumerate(chapters, 1):
            print(f"Ingesting {i}/{len(chapters)}")
            await rag.ainsert(doc)

    finally:
        if rag:
            await rag.finalize_storages()


os.makedirs(WORKING_DIR, exist_ok=True)
asyncio.run(rag_setup())

import os
import re
import asyncio
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status


load_dotenv()

os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")


LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

OLLAMA_HOST = os.getenv("LLM_BINDING_HOST")

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

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            llm_model_name=LLM_MODEL,
            llm_model_kwargs={
                "host": OLLAMA_HOST,
                "options": {"num_ctx": 8192},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=int(EMBEDDING_DIM),
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model=EMBEDDING_MODEL,
                    host=OLLAMA_HOST,
                ),
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

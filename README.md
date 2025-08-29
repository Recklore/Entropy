
# Entropi-Learning: An AI-Powered Personalized Learning System

This repository contains the source code for an AI-powered personalized learning system that dynamically adapts to a student's knowledge level. The system integrates a student knowledge model (DKTplus), a reinforcement learning-based policy agent (DQN), and a retrieval-augmented generation (RAG) component (LightRAG) to recommend learning materials and assessments tailored to individual student needs.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Data & Preprocessing](#data--preprocessing)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Assumptions](#assumptions)

## Overview

The core of this project is a closed-loop system that models a student's knowledge, recommends the next best skill to learn, provides relevant educational content, and assesses the student's understanding. This cycle allows for continuous adaptation and personalization of the learning path.

The system is composed of three main AI components:
1.  **DKTplus (Deep Knowledge Tracing Plus)**: A recurrent neural network (LSTM-based) that models the student's knowledge state over time. It takes a sequence of student interactions (questions, answers, and response times) and predicts their mastery level for a set of skills.
2.  **DQN (Deep Q-Network)**: A reinforcement learning agent that acts as the policy for the learning system. It takes the student's current knowledge mastery (from DKTplus) as input and decides which skill the student should focus on next to maximize their learning gain.
3.  **LightRAG**: A retrieval-augmented generation system that provides educational content for the skill recommended by the DQN agent. It uses a knowledge graph and vector database to retrieve relevant information and a large language model (LLM) to generate explanations and assessments.

## Architecture

The system's architecture is divided into two main processes: the Training Pipeline, where the models are created, and the Inference Pipeline, which runs the live interactive learning session.

### Training Pipeline

The training process is sequential and prepares all the necessary components for the application.

1.  **Data Preparation**:
    -   Raw data from `data/ASSISTments/` and `data/SL/` is processed by the `src/data_preprocessing.ipynb` notebook.
    -   This produces `ASSISTments_processed_data.json`, which contains structured student interaction sequences.

2.  **DKTplus Model Training**:
    -   `src/DKTplus/train.py` uses the processed data to train the knowledge tracing model.
    -   The output is the trained model file: `models/DKT_model.pt`.

3.  **DQN Agent Training**:
    -   `src/DQN_agent/train.py` uses the trained DKTplus model to simulate a student environment and train the reinforcement learning agent.
    -   The output is the trained policy agent: `models/DQN_agent.pt`.

4.  **RAG Knowledge Base Setup**:
    -   `src/RAG/lightrag_setup.py` ingests the educational content from `data/SL/sl.md`.
    -   This builds the knowledge graph and vector database used for content retrieval, stored in `data/lightrag_database/`.

### Inference Pipeline

The inference pipeline represents the live, interactive learning loop that the user experiences when running `src/main.py`.

**Flow:** `Student Interaction` -> `DKTplus` -> `DQN` -> `LightRAG` -> `Student Interaction`

1.  **Student Interaction & Assessment**: The user starts a session and takes an assessment. Their interactions (questions answered, correctness, time taken) are collected.

2.  **Knowledge Tracing (DKTplus)**: The interaction history is fed into the loaded DKTplus model to produce a "mastery vector," which represents the student's current knowledge state across all skills.

3.  **Skill Recommendation (DQN)**: The mastery vector is passed to the DQN agent, which recommends the optimal next skill for the student to learn.

4.  **Content Retrieval (LightRAG)**: The recommended skill is used to query the LightRAG system, which retrieves and presents relevant learning content to the student.

5.  **Loop**: After studying the content, the user is assessed on the new skill. Their performance is added to their interaction history, and the cycle repeats from step 2.

## Components

### DKTplus (Deep Knowledge Tracing)
-   **File**: `src/DKTplus/model.py`
-   **Class**: `DKTplus(nn.Module)`
-   **Description**: This module implements the DKTplus model using an LSTM network. It takes sequences of student interactions (question, response, time) and outputs a vector representing the student's mastery of each skill.
-   **Key Functions**:
    -   `forward(self, q, r, t)`: The forward pass for the model.

### DQN (Deep Q-Network)
-   **File**: `src/DQN_agent/dqn.py`
-   **Class**: `DQN(nn.Module)`
-   **Description**: This module implements the DQN, which is a simple multi-layer perceptron that takes the student's mastery vector as input and outputs Q-values for each skill (action).
-   **Key Functions**:
    -   `forward(self, x)`: The forward pass for the DQN.

### Student Environment
-   **File**: `src/DQN_agent/student_env.py`
-   **Class**: `student_env`
-   **Description**: This class simulates a student's learning process. It uses the DKTplus model to update the student's knowledge state based on the actions (skills) chosen by the DQN agent.
-   **Key Functions**:
    -   `step(self, action)`: Simulates one step in the learning process.
    -   `reset(self)`: Resets the student's knowledge state.

### LightRAG
-   **File**: `src/RAG/lightrag_setup.py`
-   **Description**: This script sets up the LightRAG system. It loads the educational content from a Markdown file (`data/SL/sl.md`), builds a knowledge graph and a vector database, and initializes the RAG pipeline.
-   **File**: `src/RAG/retrieve.py`
-   **Description**: This module contains functions for retrieving content and generating assessments using the LightRAG system.
-   **Key Functions**:
    -   `retrieve_content(rag, skill_name)`: Retrieves educational content for a given skill.
    -   `generate_assessment(...)`: Generates assessment questions for a given skill or a set of skills.

### Main Application
-   **File**: `src/main.py`
-   **Description**: This is the main entry point of the application. It loads the trained models, initializes the RAG system, and runs the interactive learning loop with the user.

## Data & Preprocessing

The project uses two main data sources:

1.  **ASSISTments Dataset**:
    -   **Files**: `data/ASSISTments/`
    -   **Description**: This dataset contains student interaction data from the ASSISTments online tutoring platform. It is used to train the DKTplus model.
    -   **Preprocessing**: The `src/data_preprocessing.ipynb` notebook processes the raw ASSISTments data into sequences of questions, responses, and timings, which are then saved in `data/ASSISTments/ASSISTments_processed_data.json`.

2.  **Statistical Learning (SL) Dataset**:
    -   **Files**: `data/SL/`
    -   **Description**: This dataset contains the educational content for the LightRAG system, in the form of a Markdown file (`sl.md`), as well as skill maps and questions.

### Expected Inputs and Outputs
-   **DKTplus Input**:
    -   `q`: A tensor of question/skill indices. Shape: `(batch_size, sequence_length)`
    -   `r`: A tensor of student responses (0 for incorrect, 1 for correct). Shape: `(batch_size, sequence_length)`
    -   `t`: A tensor of response times. Shape: `(batch_size, sequence_length)`
-   **DKTplus Output**: A tensor of predicted mastery levels for each skill. Shape: `(batch_size, sequence_length, num_skills)`
-   **DQN Input**: A student mastery vector. Shape: `(1, num_skills)`
-   **DQN Output**: Q-values for each skill. Shape: `(1, num_skills)`

## Requirements

The following Python packages are required to run the project. They are listed in `requirements.txt`.

```
torch
sentence-transformers
scikit-learn
pydantic
numpy
pandas
lightrag-hku[ollama,neo4j]
python-dotenv
faiss-cpu
```

## Installation

1.  **Clone the repository**:
    ```shell
    git clone https://github.com/Recklore/Entropy.git
    cd Entropy
    ```

2.  **Set up external services**:
    This project requires Neo4j and Ollama to be running.
    -   **Neo4j**: Start a Neo4j database.
    -   **Ollama**: Ensure the Ollama service is running and the required models are pulled.
        ```shell
        ollama pull mistral:7b-instruct
        ollama pull bgm-m3:567m
        ```

3.  **Create a Python virtual environment and install dependencies**:
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    **Note on PyTorch**: The `requirements.txt` file lists `torch`. You may need to install a specific version of PyTorch that is compatible with your CUDA version for GPU support. See the [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions.

4.  **Set up environment variables**:
    Create a `.env` file in the root of the project and add the following variables:
    ```
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"
    LLM_BINDING_HOST="http://localhost:11434"
    RETRIEVAL_LLM_MODEL="mistral:7b-instruct"
    EMBEDDING_MODEL="nomic-embed-text"
    EMBEDDING_DIM="768"
    ```

5.  **Prepare the data and knowledge base**:
    -   Run the data preprocessing notebook to prepare the DKTplus training data:
        Open and run all cells in `src/data_preprocessing.ipynb`.
    -   Set up the LightRAG knowledge base:
        ```shell
        python src/RAG/lightrag_setup.py
        ```

## Quickstart

To run the interactive learning application, execute the `main.py` script:

```shell
python src/main.py
```

The application will guide you through an initial assessment, and then enter a loop of recommending skills, providing content, and assessing your knowledge.

## Training

The training process is divided into two stages: training the DKTplus model and then training the DQN agent.

### 1. DKTplus Training

-   **Script**: `src/DKTplus/train.py`
-   **Command**:
    ```shell
    python src/DKTplus/train.py
    ```
-   **Hyperparameters**: The hyperparameters for the DKTplus model are defined at the top of the `train.py` script.
    -   `BATCH_SIZE = 64`
    -   `EPOCHS = 25`
    -   `NUM_C = 44` (Number of skills)
    -   `LEARNING_RATE = 0.001`
    -   `EMB_SIZE = 128`
    -   `HIDDEN_SIZE = 256`
-   **Output**: The trained DKTplus model is saved to `models/DKT_model.pt`.

### 2. DQN Training

-   **Script**: `src/DQN_agent/train.py`
-   **Command**:
    ```shell
    python src/DQN_agent/train.py
    ```
-   **Hyperparameters**: The hyperparameters for the DQN agent are defined at the top of the `train.py` script.
    -   `NUM_EPISODES = 600`
    -   `TARGET_UPDATE_EPISODES = 20`
    -   `MAX_STEPS = 200`
    -   `BATCH_SIZE = 128`
    -   `LEARNING_RATE = 0.0001`
-   **Output**: The trained DQN agent is saved to `models/DQN_agent.pt`.

## Inference

The `src/main.py` script demonstrates how to load the trained models and run the full inference pipeline. The core logic is as follows:

1.  **Load Models**: The `load_models` function loads the `DKT_model.pt` and `DQN_agent.pt` files.
2.  **Get Student Mastery**: The DKTplus model is used to get the student's mastery vector from their interaction history.
3.  **Recommend Skill**: The DQN agent takes the mastery vector and recommends the next skill to learn.
4.  **Retrieve Content**: The `retrieve_content` function from `src/RAG/retrieve.py` is called to get the learning material for the recommended skill.
5.  **Assess Student**: The `generate_assessment` function is used to create questions to assess the student's understanding of the new skill.

## Configuration

The main configuration for the application is managed through environment variables in the `.env` file.

-   `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Credentials for the Neo4j database.
-   `LLM_BINDING_HOST`: The host for the Ollama LLM service.
-   `RETRIEVAL_LLM_MODEL`: The LLM model to use for retrieval and generation.
-   `EMBEDDING_MODEL`: The model to use for text embeddings.
-   `EMBEDDING_DIM`: The dimension of the embeddings.

Model paths and other constants are defined at the top of the respective Python scripts (`src/main.py`, `src/DKTplus/train.py`, etc.).

## Troubleshooting

-   **GPU/CPU Issues**: The system will automatically use a GPU if `torch.cuda.is_available()` is true. If you encounter CUDA errors, ensure your PyTorch installation matches your CUDA version, or force CPU usage by setting `DEVICE = "cpu"`.
-   **Missing Data Paths**: If you get `FileNotFoundError`, make sure you have run the `data_preprocessing.ipynb` notebook and the `lightrag_setup.py` script.
-   **Ollama/Neo4j Connection Errors**: Ensure that the Ollama and Neo4j services are running and that the credentials in your `.env` file are correct.
-   **RAG Index Issues**: If the RAG system is not returning good results, you may need to rebuild the index by deleting the contents of the `data/lightrag_database` directory and re-running `src/RAG/lightrag_setup.py`.

## References

-   **DKTplus**: The implementation is based on the principles of Deep Knowledge Tracing.
-   **DQN**: The DQN agent is based on the work by Mnih et al., "Playing Atari with Deep Reinforcement Learning".
-   **LightRAG**: A lightweight and modular RAG framework.
-   **ASSISTments Dataset**: A public dataset for research in student learning.

## Assumptions

Here are the assumptions I made while generating this README:

-   The project requires Python 3.
-   The user has `git` and `python` installed.
-   The user is familiar with setting up services like Neo4j and Ollama.
-   No formal test suite or CI/CD pipeline exists.
-   The project does not have a `LICENSE` file.
-   The primary goal of the user is to run the interactive learning application.

# 🧠 AtomMem: Agentic Memory for LLM Agents

**AtomMem** is a long-term memory system designed for personalized LLM agents. It organizes continuous dialogue history around atomic facts, incrementally constructs event memories and temporal user profiles, and retrieves highly relevant evidence through a dynamic associative memory graph.

---

## ✨ Key Features

* 🧩 **Tri-partite Memory Architecture**: Maintains three distinct memory views: standalone **Atomic Facts**, clustered **Event Memories**, and evolving **Temporal User Profiles**.
* 🕸️ **Associative Graph Retrieval**: Utilizes a localized memory graph (Entities, Events, and Dialogue Turns) and applies Random Walk with Restart (RWR) to surface implicit contextual evidence.
* ⚡ **Plug-and-Play Pipeline**: Easily ingest raw conversations or pre-extracted facts. Fully compatible with OpenAI-like API endpoints for fact extraction and downstream QA generation.

---

## 🏗️ System Architecture

At QA time, AtomMem retrieves a high-precision seed set, activates a localized fact graph, and propagates activation scores to identify the optimal context.

## 📂 Repository Layout

```text
.
├── atommem_core/              # Public pipeline components
├── data/                      # Raw samples and Fact Executor SFT data
├── prompts/                   # LLM prompts used by memory construction and QA
├── scripts/                   # Execution and evaluation scripts
├── src/                       # Core storage, retrieval, LLM, and embedding modules
├── config.py                  # Environment-driven settings
├── .env.example
└── requirements.txt
```

## 🚀 Getting Started

1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configuration
Create your environment file and configure your LLM / Embedding endpoints. config.py contains no API keys; all credentials must be supplied via the environment.

```bash
cp .env.example .env
```

Edit `.env`:

```bash
ATOMMEM_API_KEY=your_api_key_here
ATOMMEM_API_BASE=[https://api.openai.com/v1](https://api.openai.com/v1)
ATOMMEM_LLM_MODEL=gpt-4o-mini
ATOMMEM_EMBEDDING_MODEL=all-MiniLM-L6-v2

# (Optional) If using a separate Fact Executor endpoint
ATOMMEM_FACT_EXECUTOR_API_KEY=your_fact_executor_key
ATOMMEM_FACT_EXECUTOR_API_BASE=https://your-fact-executor-endpoint/v1
ATOMMEM_FACT_EXECUTOR_MODEL=your_fact_executor_model
```

## 📊 LoCoMo Benchmark Evaluation
To evaluate AtomMem on the LoCoMo benchmark (requires raw samples in data/split_samples/ and pre-extracted facts in data/locomo_preextracted_facts/):

* **Reproduce Paper Results:** Use the provided pre-extracted facts in `data/locomo_preextracted_facts/` alongside raw samples in `data/split_samples/` to directly reproduce our reported metrics.
* **End-to-End Evaluation:** Run the full pipeline starting from raw dialogues. We provide `data/SFT_training_data.json` to help you fine-tune your own Fact Executor model for this purpose.

```bash
python scripts/evaluate_locomo.py
```

This script builds independent memory banks per conversation and evaluates categories 1-4 by default. Detailed question-level predictions and aggregate metrics (F1, BLEU-1, Recall@10, J-Score), along with total token usage, are exported to runs/locomo_eval/.


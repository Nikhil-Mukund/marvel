# MARVEL: A Multi-Agent Research Validator and Enabler using LLMs

[![DOI](https://zenodo.org/badge/1121643405.svg)](https://doi.org/10.5281/zenodo.18156826)

**Associated paper:**  
N. Mukund *et al.*, *MARVEL: A Multi-Agent Research Validator and Enabler using LLMs*, arXiv:2601.03436  
https://arxiv.org/abs/2601.03436

MARVEL [ligogpt.mit.edu/marvel](https://ligogpt.mit.edu/marvel) is a locally deployable, open-source framework for domain-aware question answering and assisted research.

It combines a fast path for straightforward queries with a **DeepSearch** mode that integrates retrieval‑augmented generation (RAG) and Monte‑Carlo Tree Search to explore complementary sub‑queries and synthesise evidence without duplication. It draws on a curated semantic index of literature, internal/public documents, and (optionally) targeted web searches, with stable citation tracking throughout.

Developed by: Nikhil Mukund, Yifang Luo, Fan Zhang, Erik Katsavounidis and Lisa Barsotti, with support from the MIT Kavli Institute for Astrophysics and Space Research & LIGO Laboratory, and the NSF AI Institute for Artificial Intelligence and Fundamental Interactions.

## Code documentation (auto-generated)

An automatically generated, browsable overview of the MARVEL source code is available at:

https://deepwiki.com/Nikhil-Mukund/marvel

This documentation is generated directly from the GitHub repository using an automated code analysis tool, and is intended to help developers and contributors quickly navigate the codebase, understand module structure, and trace key execution paths.  
For authoritative details, the source code in this repository remains the primary reference.


---

## Repository layout (high level)

- `marvel.py` — main Streamlit app
- `config/` — YAML configuration (models, retrieval, data, server, prompts, …)
- `RAG_DataSets/` — your local corpora, grouped by dataset type (e.g. `DocPDF`, `LatexData`, …)
- `faiss/` — persisted FAISS vector stores (`./faiss/<DatasetName>/`)
- `environments/` — conda environment definitions (Linux/macOS/Windows)
- `scripts/setup/setup_conda_env.py` — helper for creating the conda env
- `scripts/evaluation/datasets/` — evaluation datasets + schema docs (see [`scripts/evaluation/datasets/README.md`](scripts/evaluation/datasets/README.md))

---

## Quickstart

1. **Create the conda environment**
2. **Configure inference**
   - Local LLMs with **Ollama** (recommended)
   - Optional cloud LLM inference via **Groq**
   - Optional web search via **Tavily**
3. **Select/build your FAISS vector store** via `config/data.yaml`
4. **Launch the Streamlit app**

---

## Conda environment setup (Linux / macOS / Windows)

Environment definitions live in `./environments/`:

- **Linux**: `environments/environment.yml`
- **macOS**: `environments/environment_mac.yml`
- **Windows**: `environments/environment_windows.yml`

A helper script is provided at:

```text
scripts/setup/setup_conda_env.py
```

### Option A: Use the helper script (recommended)

The helper script runs:

```bash
conda env create -f environment.yml --prefix ~/miniconda/envs/marvel
```

So it expects an `environment.yml` **in your current working directory**.

From the repo root:

**Linux**
```bash
cp environments/environment.yml environment.yml
python scripts/setup/setup_conda_env.py
conda activate ~/miniconda/envs/marvel
```

**macOS**
```bash
cp environments/environment_mac.yml environment.yml
python scripts/setup/setup_conda_env.py
conda activate ~/miniconda/envs/marvel
```

**Windows**
```powershell
copy environments\environment_windows.yml environment.yml
python scripts\setup\setup_conda_env.py
conda activate $HOME\miniconda\envs\marvel
```

> Note: If your conda installation is not under `~/miniconda/` (common: `~/miniconda3/`), either:
> - edit `scripts/setup/setup_conda_env.py` to point at your preferred env directory, **or**
> - use Option B below.

### Option B: Create the env manually (more portable)

This avoids hard-coding an env path:

**Linux**
```bash
conda env create -f environments/environment.yml -n marvel
conda activate marvel
```

**macOS**
```bash
conda env create -f environments/environment_mac.yml -n marvel
conda activate marvel
```

**Windows**
```powershell
conda env create -f environments\environment_windows.yml -n marvel
conda activate marvel
```

---

## Local LLM inference with Ollama

MARVEL uses LangChain’s Ollama integration. You need:

1. Ollama installed and running
2. The model weights pulled locally
3. `config/models.yaml` pointing to those model names

### 1) Install Ollama

Install Ollama for your OS using the official installer (macOS, Linux, Windows). After install, confirm:

```bash
ollama --version
ollama serve
```

> On many systems Ollama runs as a background service automatically; `ollama serve` is only needed if it is not already running.

### 2) Pull the models (from `config/models.yaml`)

Your `config/models.yaml` indicates:

```yaml
primary_llm_model: gemma3:27b-it-qat
secondary_llm_model: qwen2.5:latest
tertiary_llm_model: mistral-nemo:latest
```

Pull them once:

```bash
ollama pull gemma3:27b-it-qat
ollama pull qwen2.5:latest
ollama pull mistral-nemo:latest
```

### 3) Ensure the Ollama base URL is correct

MARVEL reads the Ollama server URL from `config/server.yaml` (typically `http://localhost:11434`). If you run Ollama elsewhere, update that value.

---

## Optional: Cloud LLM inference via Groq

If you want cloud inference (instead of / in addition to Ollama):

1. Create a Groq account at **groq.com**
2. Generate an API key in your Groq dashboard
3. Add it to `config/retrieval.yaml`

In `config/retrieval.yaml`, set:

```yaml
GROQ_API_KEY: "YOUR_GROQ_API_KEY"
enable_groq_inference_for_public_docs: true
```

> In `marvel.py`, the app exports `GROQ_API_KEY` into the environment at runtime, based on the value in `config/retrieval.yaml`.

---

## Optional: Web search via Tavily

To enable web search:

1. Create an account at **tavily.com**
2. Generate an API key
3. Add it to `config/retrieval.yaml`

In `config/retrieval.yaml`, set:

```yaml
TAVILY_API_KEY: "YOUR_TAVILY_API_KEY"
enable_tavily_web_search: true
```

(Exact flag names may vary by config version; the key point is that MARVEL’s Tavily retriever needs the API key.)

---

## Selecting and building FAISS vector stores

MARVEL stores your source documents under `./RAG_DataSets/<DatasetName>/` and stores the corresponding FAISS vector database under `./faiss/<DatasetName>/`.

Which dataset is active is controlled by **`config/data.yaml`**:

```yaml
datasets: "DocPDF"     # or "LatexData", "TextData", "JSONLData", "ALL_V2", ...
```

### Important: embedding lifecycle

- If `./faiss/<DatasetName>/` **exists**, MARVEL will **load** it.
- If `./faiss/<DatasetName>/` **does not exist**, MARVEL will **create** it by embedding the data found in `./RAG_DataSets/<DatasetName>/`.

If you add new files and want to regenerate embeddings, delete the persisted vector store folder first:

```bash
rm -rf ./faiss/<DatasetName>
```

> This deletion is only needed if you want to regenerate embeddings. If nothing new was added, do not delete.

### PDF extraction method (DocPDF)

In `config/data.yaml` you can choose the PDF extraction backend:

```yaml
pdf_extraction_method: surya   # or "tesseract"
```

This value is used when building `DocPDF` vector stores.

---

## RAG dataset types

MARVEL’s dataset selection is case‑sensitive (it is used to form folder paths like `./RAG_DataSets/<datasets>/`).

### DocPDF

**Goal:** ingest PDFs.

1. Put PDFs in:

   ```text
   ./RAG_DataSets/DocPDF/
   ```

2. (Optional) delete the existing vectorstore to re-embed:

   ```bash
   rm -rf ./faiss/DocPDF
   ```

3. Select the dataset in `config/data.yaml`:

   ```yaml
   datasets: "DocPDF"
   ```

4. Launch MARVEL (see below). If `./faiss/DocPDF/` is missing, it will be created automatically.

---

### LatexData

**Goal:** ingest LaTeX sources (e.g., paper source trees / `.tex` content).

1. Put LaTeX source files or folders in:

   ```text
   ./RAG_DataSets/LatexData/
   ```

2. (Optional) delete the existing vectorstore to re-embed:

   ```bash
   rm -rf ./faiss/LatexData
   ```

3. Select the dataset in `config/data.yaml`:

   ```yaml
   datasets: "LatexData"
   ```

4. Launch MARVEL.

---

### TextData

**Goal:** ingest plain text (notes, markdown, logs, etc).

1. Put text files in:

   ```text
   ./RAG_DataSets/TextData/
   ```

2. (Optional) delete the existing vectorstore to re-embed:

   ```bash
   rm -rf ./faiss/TextData
   ```

3. Select the dataset in `config/data.yaml`:

   ```yaml
   datasets: "TextData"
   ```

4. Launch MARVEL.

---

### JSONLData

**Goal:** ingest JSONL corpora (one JSON object per line).

1. Put `.jsonl` files in:

   ```text
   ./RAG_DataSets/JSONLData/
   ```

2. (Optional) delete the existing vectorstore to re-embed:

   ```bash
   rm -rf ./faiss/JSONLData
   ```

3. Select the dataset in `config/data.yaml`:

   ```yaml
   datasets: "JSONLData"
   ```

4. Launch MARVEL.

> If your JSONL schema is custom, you may need to adapt the JSONL extractor in `libs/extractors.py`.

---

## Combining multiple vector stores with ALL_V2

If you want to run RAG over *multiple* datasets at once (e.g., PDFs + LaTeX + text + JSONL), MARVEL provides a combined store:

```yaml
datasets: "ALL_V2"
```

### How ALL_V2 works (in `marvel.py`)

When `datasets: "ALL_V2"`, MARVEL will **merge existing FAISS vectorstores** listed in `config/data.yaml` under `faiss_vector_store_merges`, and then persist the merged store to:

```text
./faiss/ALL_V2/
```

### Steps to build / rebuild ALL_V2

1. Make sure each underlying store exists (build them once):
   - `./faiss/DocPDF/`
   - `./faiss/LatexData/`
   - `./faiss/TextData/`
   - `./faiss/JSONLData/`
   - (and any other datasets you include)

2. Configure the merge list in `config/data.yaml`:

   ```yaml
   datasets: "ALL_V2"

   faiss_vector_store_merges:
     - TextData
     - DocPDF
     - LatexData
     - JSONLData
     # - AudioTextData   # include if you use it
   ```

3. If you want to regenerate the merged store, delete it first:

   ```bash
   rm -rf ./faiss/ALL_V2
   ```

4. Launch MARVEL. The merged store will be created and saved.

---

## Adding a new dataset type

To add a new input type (e.g., `DocText`, `CSVData`, etc.):

1. Create a folder:

   ```text
   ./RAG_DataSets/<NewDataSetName>/
   ```

2. Implement an extractor in `libs/extractors.py` (convert files → LangChain `Document`s).

3. Add a new branch in `marvel.py` where datasets are handled (search for `RAG_DataSet_directory` and the `elif os.path.normpath(...)` chain).

4. Add the dataset name to `config/data.yaml`:
   - `vec_store_options`
   - and (optionally) `faiss_vector_store_merges` if you want it included in `ALL_V2`.

---

## Evaluation datasets (RAGAS JSONL)

If you want to reproduce or inspect the benchmark results, MARVEL includes **per‑sample RAGAS metrics** and the
corresponding QA examples + model answers under:

- `scripts/evaluation/datasets/`  (see [`scripts/evaluation/datasets/README.md`](scripts/evaluation/datasets/README.md) for full details)

### What’s in this folder

All evaluation files are **newline‑delimited JSON** (`.jsonl`) with **one JSON object per line** (UTF‑8). Each
record includes the QA sample (question, context, reference/ground‑truth answer), the baseline model answer(s),
MARVEL answer(s), and RAGAS metrics (typically floats in **[0, 1]**; higher is better). Some metric values may be
`null` if they could not be computed.

### Files included

**Full sets (Baseline vs MARVEL‑Standard)**  
- `eval_ArxivData_gpt4o-mini_vs_MARVEL-Standard.jsonl` — **N = 910** *(ArXiv‑derived set; in this repo the same corpus is typically referred to as `LatexData`)*  
- `eval_LogbookData_gpt4o-mini_vs_MARVEL-Standard.jsonl` — **N = 696**

**DeepSearch subset (Baseline vs MARVEL‑Standard vs MARVEL‑DeepSearch)**  
- `eval_ArxivData_gpt4o-mini_vs_MARVEL-Standard_vs_MARVEL-DeepSearch.jsonl` — **N = 168**  
- `eval_LogbookData_gpt4o-mini_vs_MARVEL-Standard_vs_MARVEL-DeepSearch.jsonl` — **N = 135**

### Quick loading example

```python
import pandas as pd

df = pd.read_json(
    "scripts/evaluation/datasets/eval_ArxivData_gpt4o-mini_vs_MARVEL-Standard.jsonl",
    lines=True
)
print(df.columns)
```


## Running MARVEL

### Launch Streamlit App

From the repository root (with the conda environment activated):

```bash
streamlit run marvel.py
```

### Development workflow (fast iteration)

For fast access, LLMs and vector databases are loaded and kept in Streamlit session state.

Typical loop:

1. Launch the service.
2. Edit code in your editor.
3. Refresh the Streamlit webpage to re-run the app.

> If you change **`config/data.yaml`** (e.g., switch datasets) or you delete/rebuild FAISS folders, you may need to restart Streamlit (or clear the session) to force re-loading the vectorstore.

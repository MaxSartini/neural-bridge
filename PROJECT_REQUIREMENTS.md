# Project Requirements: Enterprise-Grade Future Predictor

## 1. Project Goal
Transform the baseline social-media simulator (Neural Bridge) into a deterministic, high-accuracy **Quantitative Market Prediction Engine**. The system utilizes multi-agent simulation specifically as a computational probability environment to argue forward-looking datasets based on strictly enforced historical market math.

## 2. Core Architecture Pipeline

### 2.1 The Data Processing Layer (The Quant Desk)
*   **Agnostic Ingestion:** The engine must accept arbitrary time-series data (e.g., CSV files) but dynamically find the primary trajectory columns without hard-coded financial definitions (enabling weather, commodity, or crypto predictions natively).
*   **Absolute Math Derivation:** Must incorporate standard qualitative data science libraries (`pandas`, `numpy`) to process the full breadth of the input file and extract absolute mathematical factual truths:
    *   Historical Z-Scores (Standard Deviations from the aggregate mean).
    *   Dynamic Support and Resistance minimum/maximums.
    *   Volatility expansion/contraction regimes.
    *   Short vs. Long-term rolling means.
*   **VRAM Limit Enforcement:** The raw data dump provided to localized LLM contexts within Neo4j graph schemas *must explicitly be capped* (e.g., Last 30 dataset rows maximum) to prevent "Lost in the Middle" token hallucination.

### 2.2 Persona Prompts & Graph Guardrails
*   **Anti-Hallucination Restrictions:** The generation rules for individual and group personas must be hard-coded to reject the invention of contextual history (company news, fictional balance sheets) not explicitly mathematically derived in the `STATISTICAL DATA REPORT`.
*   **Adversarial Modeling:** Agent personas (e.g., Institutional Quants, Contrarians, Retail Traders) must be given distinct cognitive biases to force aggressive, multi-faceted debate paths.

### 2.3 Forward Projection Engine (The Simulator)
*   **Temporal Anchoring:** Agents natively operate without a sense of date progression. The simulation execution loop must mechanically inject a "Simulation Clock" into every agent's context memory per iteration (e.g., "Round 14 of Forward Projection Window"). Support precise date anchoring (e.g., predicting a 30-day outlook).
*   **Rolling Context Mechanism:** Agents cannot view the entirety of the database continually. The core loops must summarize and compress the previous round's data iteratively to prevent contextual limits from exceeding standard 8k constraints.

### 2.4 Consensus Distillation (The Output)
*   **Post-Simulation Extraction:** The system requires an isolated, non-blocking script/process to run upon simulation termination. 
*   **Formatting Enforcement:** The extraction hook must read the localized database output (SQLite event logging), parse the accumulated multi-agent predictions, and prompt the LLM to output exclusively a verified JSON structure (e.g., 16-point JSE predictive analysis schema) to an external file for automated trading usage.

## 3. Hardware & Software Bounds
*   **Orchestration Environment:** The pipeline must survive independent execution within background threading environments (Flask, Uvicorn, Unix `nohup`) targeting M-Series Apple Silicon architectures via LM Studio proxying (Port 1234 → LLM API, Port 1235 → embeddings, Port 5001 → REST, Port 7474/7687 → Neo4j browser/Bolt, Port 3000 or 5173 → Vite client).
*   **Crash Continuity:** Strict implementation of standard un-handled exception skips (e.g. `try/except` mapping over numerical zero-division edge cases for flatlined Z-scores) to ensure the master `run_parallel_simulation.py` process never breaks on bad node ingestion.

## 4. Full Dependency Stack

### 4.1 Backend (Python 3.11+, declared in `backend/requirements.txt` and `backend/pyproject.toml`)

**Core web framework**
*   `flask` >= 3.0 — Application factory + blueprints (`/api/graph`, `/api/simulation`, `/api/report`, `/api/scrape`).
*   `flask-cors` >= 6.0 — CORS for the Vite dev server.

**LLM / inference clients (all OpenAI-compatible, pointed at local routes)**
*   `openai` >= 1.0 — Unified client for LM Studio + Ollama (`/v1/chat/completions`, `/v1/embeddings`).
*   `httpx` >= 0.27 — Service health pre-flight in `run.py` (`LM Studio LLM`, `LM Studio Embeddings`, `Neo4j` HTTP).
*   `requests` >= 2.28 — Ollama embedding fallbacks and miscellaneous HTTP calls.

**Graph database**
*   `neo4j` >= 5.15 — Bolt driver, sessions, exception types.

**Multi-agent simulation**
*   `camel-oasis` == 0.2.5 — OASIS social-platform simulator (`ActionType`, `LLMAction`, `ManualAction`, `generate_reddit_agent_graph`).
*   `camel-ai` == 0.2.78 — `ModelFactory`, `ModelPlatformType`, `BaseMessage`, `OpenAIBackendRole`. **Must** receive explicit `url=` / `api_key=` so it does not silently hit `api.openai.com`.

**Quantitative / data processing**
*   `numpy` >= 1.26 — Z-scores, rolling statistics, native-Python conversion.
*   `pandas` >= 2.1 — CSV / XLSX ingestion (`file_parser.py`), time-series consolidation (`market_data_consolidator.py`), graph CSV stats (`api/graph.py`).
*   `openpyxl` >= 3.1 — Pandas backend for `.xlsx` uploads.

**File parsing**
*   `PyMuPDF` (`fitz`) >= 1.24 — PDF text extraction.
*   `Pillow` >= 10.0 — Image decoding for the screenshot vision pipeline.
*   `charset-normalizer` >= 3.0, `chardet` >= 5.0 — Two-stage encoding detection for non-UTF-8 text.

**Utilities**
*   `python-dotenv` >= 1.0 — `.env` loading from project root.
*   `pydantic` >= 2.0 — Validation for ontology nodes and structured LLM output.

**Dev / optional**
*   `pytest` >= 8.0, `pytest-asyncio` >= 0.23 — Test suite (`backend/test_*.py`, root `test_oasis.py`).
*   `pipreqs` >= 0.5 — Dependency auditing.

**Standard library on the hot path** (no install needed, but load-bearing)
`asyncio`, `multiprocessing`, `threading`, `queue`, `signal`, `atexit`, `subprocess`, `sqlite3` (OASIS event log), `dataclasses`, `enum`, `pathlib`, `logging` (with `RotatingFileHandler`), `uuid`, `base64`, `tempfile`, `shutil`, `glob`, `csv`, `re`, `math`, `json`, `io`, `traceback`, `functools`.

### 4.2 Frontend (Node.js 18+, declared in `frontend/package.json`)

*   `vue` ^3.5 — SPA with Composition API.
*   `vue-router` ^4.6 — Client-side routing across `Process`, `MainView`, `SimulationRunView`, `ReportView`, `InteractionView`.
*   `axios` ^1.13 — API client (`baseURL` `:5001`, 5-minute timeout, retry helper) in `src/api/index.js`.
*   `d3` ^7.9 — Graph visualisation in `GraphPanel.vue` and step views.
*   `vite` ^7.2 + `@vitejs/plugin-vue` ^6.0 — Dev server on `:3000`, proxies `/api` → `:5001`.
*   Root `concurrently` ^9.1 — Boots backend + frontend together via `npm run dev`.

### 4.3 Infrastructure / external services

*   **Neo4j 5.15+ Community** with the APOC plugin (Bolt `:7687`, browser `:7474`). Heap configured 512m → 2g in `docker-compose.yml`.
*   **LM Studio** (preferred on Apple Silicon) — LLM on `:1234`, embeddings on `:1235`. Pinned model: `gemma-4-26b-a4b-it`. Embedding model: `nomic-embed-text-v1.5` (Q4_K_M GGUF, 768-dim).
*   **Ollama** (Docker default) — LLM + embeddings on `:11434`. Used by `docker-compose.yml`.
*   **Docker / docker-compose** — services: `neural_bridge`, `neural_bridge-neo4j`, `neural_bridge-ollama`.
*   **`uv`** >= 0.9 — Python lock-file install path (`backend/uv.lock`); used by the Dockerfile and `npm run backend`.

### 4.4 Project module map (source-of-truth modules in this repo)

*   **API blueprints** (`backend/app/api/`): `graph`, `simulation`, `report`, `scrape`.
*   **Services** (`backend/app/services/`): `ontology_generator`, `graph_builder`, `text_processor`, `entity_reader`, `oasis_profile_generator`, `simulation_manager`, `simulation_config_generator`, `simulation_runner`, `simulation_ipc`, `graph_memory_updater`, `graph_tools`, `market_data_consolidator`, `screenshot_processor`, `report_agent`.
*   **Storage** (`backend/app/storage/`): `neo4j_storage`, `neo4j_schema`, `graph_storage`, `embedding_service`, `ner_extractor`, `search_service`.
*   **Utilities** (`backend/app/utils/`): `file_parser`, `llm_client`, `logger`.
*   **Models** (`backend/app/models/`): `project`, `task`.
*   **Simulation scripts** (`backend/scripts/`): `run_parallel_simulation.py` (local Neural Bridge loop), compatibility adapter scripts, `extract_consensus.py` (Oracle distillation), `action_logger.py`, `test_profile_format.py`.
*   **Frontend src** (`frontend/src/`): `views/` (Home, MainView, Process, SimulationView, SimulationRunView, ReportView, InteractionView), `components/` (Step1GraphBuild → Step5Interaction, GraphPanel, HistoryDatabase), `api/` (`index.js`, `graph.js`, `simulation.js`, `report.js`), `router/`, `store/pendingUpload.js`.

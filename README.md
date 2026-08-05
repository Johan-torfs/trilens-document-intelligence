# TriLens Document Intelligence

TriLens is a local multimodal document intelligence system. Upload document images or PDFs, search them with natural language, and run experimental analysis — all without sending data to an external service.

**Stack at a glance:**

| Layer               | Technology                             |
| ------------------- | -------------------------------------- |
| Visual embeddings   | SigLIP (via Hugging Face Transformers) |
| Text embeddings     | Sentence Transformers                  |
| OCR                 | DocTR                                  |
| Vector store        | Qdrant                                 |
| Metadata store      | SQLite                                 |
| Analysis (optional) | OpenFlamingo                           |
| Backend API         | FastAPI                                |
| Frontend            | Next.js                                |
| Prototype UI        | Streamlit                              |

Search uses **hybrid ranking**: SigLIP similarity, sentence-level text similarity and full-text score are combined into a single final score.

> TriLens is a portfolio and learning project. Model output may be inaccurate and must not be used as legal, financial or identity advice.

---

## Screenshots

### Unified dashboard

The Next.js interface combines document upload and semantic search on a single page.

![TriLens dashboard with upload and search forms](docs/screenshots/01-dashboard.png)

### Search results

Results include the document image, document type and individual ranking signals.

![TriLens hybrid document search results](docs/screenshots/02-search-results.png)

### Inline document analysis

A result can be analysed directly on the same page. OpenFlamingo is optional and can be disabled.

![TriLens inline document analysis](docs/screenshots/03-inline-analysis.png)

### Upload result

Uploaded documents show embedding model, OCR model and processing outcome.

![TriLens document indexing result](docs/screenshots/04-upload-result.png)

---

## Installation and setup

### Prerequisites

- Python 3.12
- Node.js 20 or newer and npm
- Docker (for Qdrant)

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/Johan-torfs/trilens-document-intelligence.git
cd trilens-document-intelligence
python -m venv .venv
source .venv/bin/activate
```

### 2 — Install Python dependencies

```bash
pip install -e .
```

PyTorch is not on PyPI with a universal wheel. If the default `pip install` does not pull a matching build for your hardware, install it manually first:

```bash
# CPU-only example
pip install torch==2.2.* torchvision==0.17.* --index-url https://download.pytorch.org/whl/cpu
```

See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) for the correct index URL for your platform and CUDA version.

OpenFlamingo is not on PyPI. Install it from source if you want experimental analysis:

```bash
pip install git+https://github.com/mlfoundations/open_flamingo.git@<commit>
```

The application runs fully without it.

### 3 — Install the frontend

```bash
cd frontend
npm ci
cd ..
```

### 4 — Start Qdrant

```bash
docker compose up -d
```

This starts Qdrant on `http://localhost:6333` using a named Docker volume for persistence.

### 5 — Configure environment variables

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

The defaults work out of the box for local development. The key variables:

**`.env`**

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=trilens_vectors_v1
TRILENS_OPEN_FLAMINGO_ENABLED=false
TRILENS_CORS_ORIGINS=http://localhost:3000
```

**`frontend/.env.local`**

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### 6 — Run the backend

```bash
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

API available at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

### 7 — Run the frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

### Streamlit prototype (optional)

The original prototype UI is still available. Streamlit is not in the default dependencies — install it first:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

---

## How it works

### Document indexing pipeline

When a document is uploaded:

1. **Validation** — MIME type, extension and image integrity are checked.
2. **Preparation** — EXIF orientation is corrected, the image is converted to RGB and resized.
3. **Visual embedding** — SigLIP encodes the image into a vector and stores it in Qdrant.
4. **Text embedding** — a Sentence Transformer encodes document metadata for text-based retrieval.
5. **OCR** — DocTR extracts document text, stored in SQLite.
6. **Classification** — if no document type is provided, the type is inferred automatically (see below).
7. **Deduplication** — a SHA-256 checksum prevents re-indexing the same file.

PDF documents are split into individual pages; each page is embedded and stored separately.

### Document classification

When no document type is selected at upload time, TriLens classifies the document automatically using two signals:

- **Visual (SigLIP zero-shot)** — the document image is embedded with SigLIP and compared against a precomputed embedding for each candidate type's text prompt (e.g. "a business invoice"). Cosine similarities are softmax-normalised across all types.
- **Lexical** — the OCR text is scanned for type-specific keywords (e.g. `invoice no`, `total due`). Hit counts are softmax-normalised across all types.

The two signals are fused with a fixed weight:

$$\text{score}_{\text{type}} = 0.60 \times \text{visual}_{\text{type}} + 0.40 \times \text{lexical}_{\text{type}}$$

The type with the highest combined score is assigned. If the winning score is below the confidence threshold (default 0.40), the document is labelled `unknown`.

The user can always override the auto-detected type by selecting one from the dropdown before uploading.

**Supported types:** `invoice`, `purchase_order`, `receipt`, `delivery_note`, `application_form`, `identity_card`, `contract`, `letter`, `report`, `bank_statement`, `pay_slip`, `quotation`, `certificate`, `tax_document`.

### Hybrid search

A query goes through the following steps:

1. SigLIP encodes the query text and retrieves a candidate pool from Qdrant (3× the requested top-k, capped at 100).
2. A Sentence Transformer computes text similarity between the query and indexed document text.
3. Full-text search provides an additional FTS signal.
4. The final score is a weighted combination:

$$\text{score} = 0.60 \times \text{visual} + 0.30 \times \text{text} + 0.10 \times \text{fts}$$

When no text or FTS signal is available for a candidate, the final score equals the visual score.

5. Candidates are re-ranked by final score and the top-k are returned.

### Optional analysis

When OpenFlamingo is enabled, a selected search result can be queried with a natural language question. The model runs locally. It is slow on CPU and may produce incorrect or hallucinated output.

If OpenFlamingo is disabled or unavailable, a fallback answer is returned from stored OCR text.

### Storage

| Store            | Contents                                                       |
| ---------------- | -------------------------------------------------------------- |
| SQLite           | Document records, processing status, OCR text, model artifacts |
| Qdrant           | Visual and text embedding vectors                              |
| Local filesystem | Uploaded document images                                       |

Runtime files live under `data/runtime/` and are not committed to Git.

---

## API reference

| Method | Endpoint                       | Description                        |
| ------ | ------------------------------ | ---------------------------------- |
| `GET`  | `/api/health`                  | Returns API status                 |
| `POST` | `/api/documents`               | Uploads and indexes a document     |
| `POST` | `/api/search`                  | Searches indexed documents         |
| `GET`  | `/api/documents/{id}/image`    | Returns the stored document image  |
| `POST` | `/api/documents/{id}/analysis` | Runs analysis on a single document |

### Search request

```json
{
  "query": "invoice with several product rows",
  "top_k": 5,
  "document_type": "invoice"
}
```

`document_type` is optional. If omitted the type is inferred automatically. Supported values: `invoice`, `purchase_order`, `receipt`, `delivery_note`, `application_form`, `identity_card`, `contract`, `letter`, `report`, `bank_statement`, `pay_slip`, `quotation`, `certificate`, `tax_document`.

### Search response (abbreviated)

```json
{
  "ranking_mode": "hybrid",
  "results": [
    {
      "document_id": "...",
      "rank": 1,
      "final_score": 0.84,
      "visual_score": 0.79,
      "text_score": 0.91,
      "fts_score": 0.6,
      "document_type": "invoice",
      "image_url": "/api/documents/.../image"
    }
  ]
}
```

---

## Project structure

```text
trilens-document-intelligence/
├── app/
│   ├── api/            # FastAPI routes and dependencies
│   ├── domain/         # Pydantic domain models
│   ├── preprocessing/  # Image validation, transforms, pipeline
│   ├── repositories/   # SQLite and Qdrant adapters
│   ├── services/       # Application services and pipeline
│   ├── strategies/     # Model strategy implementations
│   ├── ui/             # Streamlit pages
│   └── bootstrap.py    # Dependency wiring
├── frontend/           # Next.js interface
├── tests/              # Python test suite
├── scripts/            # Dataset download utilities
├── data/
│   ├── external/       # Downloaded dataset images (not in Git)
│   └── runtime/        # SQLite, Qdrant uploads (not in Git)
├── docs/screenshots/
├── docker-compose.yaml
├── pyproject.toml
└── requirements.txt
```

---

## Tests

```bash
# Python tests
.venv/bin/pytest

# Frontend lint and build
cd frontend
npm run lint
npm run build
```

Model-heavy integrations are mocked; the test suite does not download model weights.

---

## Dataset

A small dataset of external document images is used for local development and evaluation. The images are not committed to Git.

| Source    | Count | Document types                                      |
| --------- | ----- | --------------------------------------------------- |
| CORD      | 10    | Receipts                                            |
| FUNSD     | 10    | Scanned forms                                       |
| DocLayNet | 30    | Financial reports, articles, laws, manuals, patents |

To download:

```bash
# FUNSD requires a manual download first:
# https://guillaumejaume.github.io/FUNSD/download
python -m scripts.dataset.fetch_external_datasets \
  --funsd-archive ~/Downloads/dataset.zip
```

CORD and DocLayNet are fetched automatically. Dataset-specific licence terms apply.

---

## Continuous integration

GitHub Actions runs on every push to `main` and on pull requests:

- Python test suite (Python 3.12)
- `npm ci` and Next.js lint
- Next.js production build

Model downloads are disabled in CI.

---

## Known limitations

- The evaluation dataset is small; retrieval quality varies by document type and query phrasing.
- SigLIP is not an OCR system — queries targeting exact document text may not match as expected.
- OpenFlamingo is slow on CPU and may hallucinate or repeat output.
- The current OpenFlamingo configuration may exceed 8 GB of GPU memory.
- Processing is synchronous; there is no background job queue.
- No authentication or rate limiting.
- Not intended for production use.

### Classification

- **Scores are uncalibrated.** Combined classification scores rarely exceed 40% in practice. The `confidence` value in the API response reflects the raw fused score, not a calibrated probability.
- **English only.** The lexical scorer matches against English keywords. Non-English document text produces no lexical signal; classification falls back to visual scoring alone.

### Search scoring

- **Weights are placeholders.** The hybrid ranking weights (0.60 visual / 0.30 text / 0.10 FTS) and score calibration thresholds have not been empirically validated. Final scores should not be treated as absolute confidence values.

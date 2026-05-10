# RBAC RAG Internal Chatbot

An internal AI chatbot for a FinTech organization that combines:

- Role-Based Access Control (RBAC)
- Retrieval-Augmented Generation (RAG)
- Department-specific document access
- Secure, context-rich answers for employees and leadership

This project is designed to solve a common enterprise problem: teams across Finance, Marketing, HR, Engineering, and Leadership need fast access to the right information without exposing data they should not see.

## Problem Statement

The company has communication delays and data silos across departments. These delays slow down decision-making, project execution, and strategic planning.

The goal of this project is to build a chatbot that:

1. Authenticates the user.
2. Identifies the user’s role.
3. Retrieves only the documents allowed for that role.
4. Generates a response using relevant context.
5. Returns the answer with source references.

## Current Status

This repository currently contains the starter backend and sample department documents.

Implemented so far:

- FastAPI app scaffold
- JWT-based authentication (`/token`) with hashed passwords
- Role assignment and reusable RBAC guards
- Department-level access matrix (including `administrator` and `employee`)
- Document ingestion pipeline (`POST /ingest`) for markdown, text, and CSV
- Chunking + vector indexing with metadata (`department`, `source`, `allowed_roles`)
- RBAC-filtered retrieval endpoint (`POST /retrieve`)
- Sample data files for Finance, HR, Marketing, Engineering, and General use

Not yet implemented:

- Automated tests

## Roles And Access

The chatbot should enforce access based on the following roles:

| Role | Allowed Access |
|---|---|
| Administrator | Ingest, delete documents, create/delete users, change roles, and full access to all company data |
| Finance Team | Financial reports, marketing expenses, equipment costs, reimbursements, and related finance documents |
| Marketing Team | Campaign performance data, customer feedback, and sales metrics |
| HR Team | Employee data, attendance records, payroll, and performance reviews |
| Engineering Department | Technical architecture, development processes, and operational guidelines |
| C-Level Executives | Full access to all company data |
| Employee Level | General company information only, such as policies, events, and FAQs |

## Project Structure

```text
.
├── app
│   ├── main.py
│   ├── schemas/
│   ├── services/
│   └── utils/
├── resources
│   ├── data
│   │   ├── engineering/
│   │   ├── finance/
│   │   ├── general/
│   │   ├── hr/
│   │   └── marketing/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Data Sources

The project includes sample source documents in `resources/data`:

- `resources/data/finance/`
- `resources/data/marketing/`
- `resources/data/hr/`
- `resources/data/engineering/`
- `resources/data/general/`

These files are intended to be ingested into a vector store and tagged with metadata such as:

- department
- source file
- access roles
- document type

## How The System Should Work

The intended workflow is:

1. A user logs in with valid credentials.
2. The backend assigns the user a role.
3. The chatbot receives a natural language question.
4. The app checks whether the user is allowed to access the requested content.
5. The retriever searches only within permitted documents.
6. The LLM generates a response using retrieved context.
7. The response includes source references so the user can verify the answer.

## Local User Accounts

The current app uses a hardcoded in-memory user database in `app/main.py` with hashed passwords.

Example users:

- `Tony` / `password123` / `engineering`
- `Ram` / `admin123` / `administrator`
- `Bruce` / `securepass` / `marketing`
- `Sam` / `financepass` / `finance`
- `Peter` / `pete123` / `engineering`
- `Sid` / `sidpass123` / `marketing`
- `Natasha` / `hrpass123` / `hr`
- `Morgan` / `execpass123` / `c_level`
- `Eve` / `employeepass` / `employee`

## API Endpoints

The backend currently exposes these endpoints:

### `POST /token`
Authenticates with username and password, then returns a bearer token.

### `GET /login`
Protected endpoint that validates the bearer token and returns welcome info.

### `GET /me`
Returns the current authenticated user and role.

### `GET /permissions`
Returns role-based department access for the current user.

### `GET /access/{department}`
Checks whether the current user can access a given department.

### `GET /test`
A protected endpoint used to confirm authentication is working.

### `POST /ingest`
Queues a background ingest job that indexes `resources/data`.
Allowed roles: `administrator`.

### `POST /ingest/sync`
Runs ingestion immediately for administrator workflows.

### `GET /ingest/jobs`
Lists background ingest jobs.

### `GET /ingest/jobs/{job_id}`
Returns the status of a specific ingest job.

### `GET /admin/users`
Lists all users for administrator management.

### `POST /admin/users`
Creates a new user in PostgreSQL with a hashed password.

### `DELETE /admin/users/{username}`
Deletes a user, but blocks administrators from deleting their own account.

### `PATCH /admin/users/{username}/role`
Changes a user role.

### `GET /admin/documents`
Lists all documents available to the RAG pipeline.

### `POST /admin/documents/upload`
Uploads a new document into the document store and rebuilds the vector index.

### `GET /health`
Returns a simple readiness signal and the current indexed chunk count.

### `GET /metrics`
Returns lightweight operational metrics such as uptime, ingest job counts, and rate-limit settings.

### `POST /retrieve`
Performs RBAC-filtered semantic retrieval for the authenticated user.

### `POST /chat`
Authenticated chat endpoint: retrieves role-filtered context and generates an answer with citations.

## Step-By-Step Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag_with_rbac_chatbot
```

### 2. Create and activate a virtual environment

If you use `uv`:

```bash
uv venv rbac_chatbot_env
source rbac_chatbot_env/bin/activate
```

If you prefer `python -m venv`:

```bash
python3 -m venv rbac_chatbot_env
source rbac_chatbot_env/bin/activate
```

### 3. Install dependencies

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
pip install -e .
```

### 4. Run the FastAPI server

```bash
uv run fastapi dev app/main.py
```

If you prefer `uvicorn`:

```bash
uvicorn app.main:app --reload
```

### 5. Run the Streamlit UI

In a new terminal:

```bash
uv run streamlit run streamlit_app.py
```

Open:

- `http://localhost:8501`

### 5b. Run with Docker

If you prefer containers:

```bash
docker compose up --build
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8501`

### 6. Test authentication

Open the API docs:

- `http://127.0.0.1:8000/docs`

Then:

1. Use `POST /token` with form data: `username=<your_user>` and `password=<your_password>`.
2. Click `Authorize` in Swagger and paste `Bearer <access_token>`.
3. Call `POST /ingest` once (using the `administrator` account).
4. Poll `GET /ingest/jobs/{job_id}` until the job reaches `completed`.
5. Call protected endpoints: `GET /login`, `GET /me`, `GET /permissions`, `GET /access/finance`, `POST /retrieve`, `POST /chat`.

### 7. Test via Streamlit

1. Log in using one of the sample users.
2. Use the `Admin` view to manage users, upload documents, delete documents, and run ingestion as `administrator`.
3. Switch back to `Chat` and ask questions to verify `Sources` and `Citations` are shown.

## LLM Configuration

`/chat` can call an LLM if you configure environment variables.

OpenAI (recommended for quick start):

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `LLM_PROVIDER` (optional, default: `openai`)
- `LLM_TIMEOUT_S` (optional, default: `30`)

If `OPENAI_API_KEY` is not set, `/chat` still works but uses a simple local fallback response.

## Storage Configuration

The project now separates relational metadata from the raw document files:

- PostgreSQL stores users plus document references and metadata only.
- Raw document files are copied into a separate document-store filesystem path.
- Chroma stores embeddings and chunk metadata in its own container in Docker.

Config variables:

- `DATABASE_URL` points to PostgreSQL in production and Docker, or SQLite for local tests if needed.
- `DOCUMENT_STORE_PATH` controls where raw document files are stored.
- `CHROMA_HOST` and `CHROMA_PORT` point the API at a separate Chroma server when it is running as its own container.
- `CHROMA_SSL` enables HTTPS for the Chroma connection when needed.
- `VECTOR_DB_PATH` remains available for local embedded Chroma fallback mode.
- `VECTOR_COLLECTION` (optional, default: `rbac_docs`)

After changing embedding model/provider or vector store config, run `POST /ingest` again to rebuild the index.

## Security And Ops

- `SECRET_KEY` sets the JWT signing secret and should be a long random value in production.
- `AUDIT_LOG_PATH` controls where audit events are written as JSONL.
- `REQUEST_BODY_MAX_BYTES` sets the maximum request body size allowed by the API middleware.
- `RATE_LIMIT_WINDOW_SECONDS` and `RATE_LIMIT_MAX_REQUESTS` control the simple IP-based throttle.
- `INGEST_MAX_ATTEMPTS` and `INGEST_RETRY_BACKOFF_SECONDS` control background ingest retries.
- `DATABASE_URL` configures PostgreSQL for users and resource document storage.
- `DOCUMENT_STORE_PATH` controls the filesystem location where raw documents are copied.

## Docker Deployment

The repository includes:

- [`Dockerfile`](Dockerfile) for API and Streamlit targets
- [`docker-compose.yml`](docker-compose.yml) for local deployment with PostgreSQL, API, and Streamlit
- [`.dockerignore`](.dockerignore) to keep secrets, caches, and generated files out of the image context

Use `docker compose up --build` to start PostgreSQL, Chroma, the API, and the Streamlit UI with persistent volumes for the database, document store, vector store, and audit log.

## Recommended Next Development Steps

1. Add integration tests for authentication, RBAC deny/allow behavior, retrieval filtering, and citation responses.
2. Add conversation persistence and audit logs for enterprise traceability.
3. Add production deployment concerns: background ingestion jobs, retries, and monitoring.

## Suggested Tech Stack Expansion

The README currently aligns with the starter stack, but the full solution will likely also need:

- `pydantic` for typed request and response models
- `langchain` or `llamaindex` for RAG orchestration
- `qdrant`, `chroma`, or `pinecone` for vector search
- `sentence-transformers` or an embedding model provider
- `streamlit` for the UI
- `pytest` for automated tests

## Development Notes

- Keep department documents separated by access level.
- Store source metadata with every chunk.
- Prefer explicit role checks before retrieval.
- Always return the source document name or path in the chatbot response.
- Add tests for both allowed and denied access paths.

## Project Goal

The final system should allow employees to ask natural language questions and receive:

- correct answers
- only the data they are permitted to see
- a traceable source reference
- a clean chat experience for internal use

## License

Add your preferred license before publishing the project publicly.

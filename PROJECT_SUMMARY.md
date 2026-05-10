# Project Summary

## Project Overview
This is an internal AI chatbot for a FinTech company. It combines retrieval-augmented generation with role-based access control so employees can ask questions and get grounded answers from company documents without seeing data they are not allowed to view.

It solves a common enterprise problem: knowledge is split across Finance, HR, Marketing, Engineering, General, and leadership. The system reduces search time and sharing risk by filtering content by role and returning answers with sources.

The intended users are administrators, department users, and general employees. Administrators manage users, documents, and ingestion. The supported roles are `administrator`, `finance`, `marketing`, `hr`, `engineering`, `c_level`, and `employee`.

## How It Works: End-To-End Workflow
1. The user logs in through the Streamlit frontend with a username and password.
2. FastAPI verifies the credentials and returns a JWT access token.
3. The token carries the user’s username and role for later requests.
4. When a question is asked, the backend checks whether a department is implied or selected.
5. The retriever searches Chroma for matching chunks that are allowed for that role.
6. Access control is enforced on both the requested department and the returned chunks.
7. The backend builds a prompt that tells the LLM to answer only from retrieved context.
8. The API returns the answer, source paths, retrieved chunks, and citations to the UI.

## Architecture & Design
The frontend is a Streamlit app with login, chat, and an admin console. The backend is a FastAPI service that handles authentication, RBAC, ingestion, retrieval, document management, health, metrics, and chat.

PostgreSQL stores users and document metadata. A filesystem document store keeps raw uploaded files. Chroma stores embeddings and chunk metadata. These services communicate over HTTP through the FastAPI layer.

Ingestion works in two stages. First, source documents are copied into the document store and recorded in PostgreSQL with department, file type, checksum, and source path. Second, the RAG service chunks the documents, embeds the chunks, and writes them into Chroma with access metadata. The current implementation tracks pending, new, and modified documents so unchanged files are skipped.

## Tools & Technologies
- **FastAPI**: backend API for auth, chat, retrieval, admin, and ingest endpoints.
- **Streamlit**: frontend for login, chat, document upload, and admin controls.
- **PostgreSQL**: stores users and document metadata.
- **SQLAlchemy**: ORM and database session layer.
- **ChromaDB**: vector store for embeddings and semantic retrieval.
- **OpenAI API**: optional LLM and embedding provider when keys are set.
- **httpx**: HTTP client used by the model wrappers.
- **python-jose**: JWT creation and verification.
- **passlib**: password hashing and verification.
- **python-dotenv**: environment variable loading.
- **requests**: API client used by the Streamlit frontend.
- **pytest**: automated integration testing.
- **uv / uv.lock**: dependency management and reproducible environment setup.

## Key Features
- Role-based access control across chat and retrieval.
- Department-scoped retrieval so users only see permitted content.
- Grounded answers with citations and source paths.
- Admin controls for creating users, changing roles, deleting users, uploading documents, deleting documents, and triggering ingestion.
- Incremental ingest tracking that highlights new or modified documents and skips unchanged files.
- Health, metrics, audit logging, and basic rate limiting.
- Local fallback behavior when OpenAI credentials are not configured.

## Project Structure
- `app/main.py`: FastAPI app, routes, auth, chat flow, and ingest orchestration.
- `app/services/`: persistence layer, RAG pipeline, embeddings, LLM, and user data.
- `streamlit_app.py`: frontend UI, admin console, and API helpers.
- `resources/data/`: sample department documents.
- `resources/document_store/`: runtime storage for uploaded documents and bootstrap copies.
- `tests/`: integration tests for login, permissions, ingestion, admin actions, retrieval, and citations.
- `docker-compose.yml`: local multi-container setup for PostgreSQL, Chroma, API, and UI.
- `pyproject.toml`: project metadata and dependencies.


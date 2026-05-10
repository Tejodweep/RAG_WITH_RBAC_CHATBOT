import importlib
import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient


def _get_token(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/token", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "access_token" in data
    return data["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _start_ingest(client: TestClient, token: str) -> str:
    resp = client.post("/ingest", headers=_auth_headers(token))
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert "job_id" in data
    return data["job_id"]


def _wait_for_ingest(client: TestClient, token: str, job_id: str) -> dict:
    for _ in range(50):
        resp = client.get(f"/ingest/jobs/{job_id}", headers=_auth_headers(token))
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] in {"completed", "failed"}:
            return data
        time.sleep(0.05)
    raise AssertionError("ingest job did not finish in time")


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    # Use a temp persistent Chroma path so tests don't touch local data.
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
        os.environ["VECTOR_DB_PATH"] = tmp
        os.environ["VECTOR_COLLECTION"] = "test_docs"
        os.environ["DOCUMENT_STORE_PATH"] = f"{tmp}/documents"
        os.environ["CHROMA_HOST"] = ""
        os.environ["CHROMA_PORT"] = ""
        os.environ["CHROMA_SSL"] = "false"

        # Ensure tests never call external OpenAI services.
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["EMBED_PROVIDER"] = "local"
        os.environ["LLM_PROVIDER"] = "local"

        import app.services.postgres_store as postgres_store
        import app.main as main

        importlib.reload(postgres_store)
        importlib.reload(main)
        with TestClient(main.app) as client:
            yield client


def test_token_and_me(client: TestClient) -> None:
    token = _get_token(client, "Tony", "password123")
    resp = client.get("/me", headers=_auth_headers(token))
    assert resp.status_code == 200
    assert resp.json()["username"] == "Tony"
    assert resp.json()["role"] == "engineering"


def test_permissions_endpoint(client: TestClient) -> None:
    token = _get_token(client, "Sam", "financepass")
    resp = client.get("/permissions", headers=_auth_headers(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "finance"
    assert "finance" in data["departments_allowed"]
    assert "general" in data["departments_allowed"]


def test_department_access_allow_and_deny(client: TestClient) -> None:
    employee_token = _get_token(client, "Eve", "employeepass")
    resp = client.get("/access/general", headers=_auth_headers(employee_token))
    assert resp.status_code == 200
    resp = client.get("/access/finance", headers=_auth_headers(employee_token))
    assert resp.status_code == 403


def test_ingest_requires_privileged_role(client: TestClient) -> None:
    employee_token = _get_token(client, "Eve", "employeepass")
    resp = client.post("/ingest", headers=_auth_headers(employee_token))
    assert resp.status_code == 403

    engineering_token = _get_token(client, "Tony", "password123")
    resp = client.post("/ingest", headers=_auth_headers(engineering_token))
    assert resp.status_code == 403

    c_level_token = _get_token(client, "Morgan", "execpass123")
    resp = client.get("/access/finance", headers=_auth_headers(c_level_token))
    assert resp.status_code == 200
    resp = client.get("/access/engineering", headers=_auth_headers(c_level_token))
    assert resp.status_code == 200
    resp = client.post("/ingest", headers=_auth_headers(c_level_token))
    assert resp.status_code == 403

    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"
    assert data["files_indexed"] > 0
    assert data["chunks_indexed"] > 0

    import app.services.postgres_store as postgres_store

    documents = postgres_store.list_resource_documents()
    assert documents
    first_document = documents[0]
    assert Path(first_document.source_path).exists()
    assert first_document.source_path.startswith(os.environ["DOCUMENT_STORE_PATH"])


def test_ingest_status_tracks_pending_documents(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")

    status_resp = client.get("/ingest/status", headers=_auth_headers(admin_token))
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["has_pending_documents"] is True
    assert status_data["pending_documents"] > 0

    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    status_resp = client.get("/ingest/status", headers=_auth_headers(admin_token))
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["has_pending_documents"] is False
    assert status_data["pending_documents"] == 0
    assert status_data["message"] == "No new documents to ingest."

    upload_resp = client.post(
        "/admin/documents/upload",
        headers=_auth_headers(admin_token),
        files={"file": ("new_general_note.txt", b"fresh general note", "text/plain")},
        data={"department": "general"},
    )
    assert upload_resp.status_code == 201, upload_resp.text

    status_resp = client.get("/ingest/status", headers=_auth_headers(admin_token))
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["has_pending_documents"] is True
    assert status_data["pending_documents"] == 1
    assert any(dept["department"] == "general" for dept in status_data["departments"])

    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    status_resp = client.get("/ingest/status", headers=_auth_headers(admin_token))
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["has_pending_documents"] is False
    assert status_data["pending_documents"] == 0


def test_admin_user_and_document_management(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")

    create_resp = client.post(
        "/admin/users",
        headers=_auth_headers(admin_token),
        json={"username": "Lara", "password": "secret123", "role": "marketing", "disabled": False},
    )
    assert create_resp.status_code == 201, create_resp.text
    assert create_resp.json()["username"] == "Lara"
    assert create_resp.json()["role"] == "marketing"

    update_resp = client.patch(
        "/admin/users/Lara/role",
        headers=_auth_headers(admin_token),
        json={"role": "c_level"},
    )
    assert update_resp.status_code == 200, update_resp.text
    assert update_resp.json()["role"] == "c_level"

    list_resp = client.get("/admin/users", headers=_auth_headers(admin_token))
    assert list_resp.status_code == 200
    assert any(user["username"] == "Lara" for user in list_resp.json())

    docs_resp = client.get("/admin/documents", headers=_auth_headers(admin_token))
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert docs

    upload_resp = client.post(
        "/admin/documents/upload",
        headers=_auth_headers(admin_token),
        files={"file": ("admin_note.txt", b"hello admin note", "text/plain")},
        data={"department": "general"},
    )
    assert upload_resp.status_code == 201, upload_resp.text

    delete_doc_resp = client.delete(f"/documents/{docs[0]['id']}", headers=_auth_headers(admin_token))
    assert delete_doc_resp.status_code == 200, delete_doc_resp.text
    assert delete_doc_resp.json()["deleted"] is True

    delete_self_resp = client.delete("/admin/users/Ram", headers=_auth_headers(admin_token))
    assert delete_self_resp.status_code == 403

    delete_user_resp = client.delete("/admin/users/Lara", headers=_auth_headers(admin_token))
    assert delete_user_resp.status_code == 200, delete_user_resp.text


def test_retrieve_filters_by_rbac(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    employee_token = _get_token(client, "Eve", "employeepass")
    # Employee should be denied finance department explicitly.
    resp = client.post(
        "/retrieve",
        headers=_auth_headers(employee_token),
        json={"query": "net income", "top_k": 3, "department": "finance"},
    )
    assert resp.status_code == 403

    # Employee should be allowed general.
    resp = client.post(
        "/retrieve",
        headers=_auth_headers(employee_token),
        json={"query": "leave policy", "top_k": 3, "department": "general"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "employee"
    assert isinstance(data["results"], list)

    # Exact employee-id lookups should favor the HR CSV row over semantically related prose.
    resp = client.post(
        "/retrieve",
        headers=_auth_headers(admin_token),
        json={"query": "What is the name of employee id FINEMP1006?", "top_k": 3},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["results"], data
    first = data["results"][0]
    assert first["source"].endswith("hr_data.csv"), first
    assert "FINEMP1006" in first["text"], first
    assert "Sara Sharma" in first["text"], first


def test_chat_returns_citations(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    finance_token = _get_token(client, "Sam", "financepass")
    resp = client.post(
        "/chat",
        headers=_auth_headers(finance_token),
        json={"message": "What is the net margin?", "top_k": 3, "department": "finance"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "answer" in data
    assert data["role"] == "finance"
    assert isinstance(data.get("sources"), list)
    assert isinstance(data.get("citations"), list)
    assert len(data["citations"]) > 0
    first = data["citations"][0]
    assert "id" in first and "source" in first and "chunk_id" in first


def test_chat_returns_fixed_fallback_for_unrelated_query(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    resp = client.post(
        "/chat",
        headers=_auth_headers(admin_token),
        json={"message": "What is the capital of France?", "top_k": 3},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "You don't have permission to access this information.\n   Please contact your administrator if you need access."
    assert data["sources"] == []
    assert data["context_chunks"] == []
    assert data["citations"] == []


def test_chat_extracts_single_field_without_dumping_record(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    resp = client.post(
        "/chat",
        headers=_auth_headers(admin_token),
        json={"message": "What is the email of Prisha Saxena?", "top_k": 4},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "prisha.saxena@fintechco.com" in data["answer"]
    assert "employee_id:" not in data["answer"].lower()
    assert "salary" not in data["answer"].lower()
    assert "manager_id" not in data["answer"].lower()


def test_chat_synthesizes_multi_chunk_answer(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    resp = client.post(
        "/chat",
        headers=_auth_headers(admin_token),
        json={"message": "What are the primary drivers of expense in 2024?", "top_k": 4, "department": "finance"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] != "I don't have enough detail on that. Could you clarify [specific gap]?"
    assert "Vendor Services" in data["answer"]
    assert "Software Subscriptions" in data["answer"]
    assert "Employee Benefits and HR Costs" in data["answer"] or "Employee Benefits & HR" in data["answer"]


def test_marketing_user_is_denied_hr_employee_record_queries(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    marketing_token = _get_token(client, "Bruce", "securepass")

    retrieve_resp = client.post(
        "/retrieve",
        headers=_auth_headers(marketing_token),
        json={"query": "What is the name of employee id FINEMP1006?", "top_k": 3},
    )
    assert retrieve_resp.status_code == 403
    assert "You don't have permission to access this information." in retrieve_resp.text

    chat_resp = client.post(
        "/chat",
        headers=_auth_headers(marketing_token),
        json={"message": "What is the name of employee id FINEMP1006?", "top_k": 3},
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["answer"] == "You don't have permission to access this information.\n   Please contact your administrator if you need access."
    assert chat_data["sources"] == []
    assert chat_data["context_chunks"] == []
    assert chat_data["citations"] == []


def test_hr_user_is_denied_finance_questions(client: TestClient) -> None:
    admin_token = _get_token(client, "Ram", "admin123")
    job_id = _start_ingest(client, admin_token)
    data = _wait_for_ingest(client, admin_token, job_id)
    assert data["status"] == "completed"

    hr_token = _get_token(client, "Natasha", "hrpass123")
    resp = client.post(
        "/chat",
        headers=_auth_headers(hr_token),
        json={"message": "What are the primary drivers of expense in 2024?", "top_k": 4},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "You don't have permission to access this information.\n   Please contact your administrator if you need access."
    assert data["sources"] == []
    assert data["context_chunks"] == []
    assert data["citations"] == []


def test_health_and_metrics(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    health_data = health.json()
    assert "status" in health_data
    assert "vector_store_ready" in health_data

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    metrics_data = metrics.json()
    assert "uptime_seconds" in metrics_data
    assert "ingest_jobs" in metrics_data

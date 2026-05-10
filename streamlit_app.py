import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEPARTMENT_OPTIONS = ["auto", "general", "finance", "marketing", "hr", "engineering"]
ROLE_OPTIONS = ["administrator", "finance", "marketing", "hr", "engineering", "c_level", "employee"]


def _get_auth_headers() -> Dict[str, str]:
    token = st.session_state.get("access_token", "")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _login(username: str, password: str) -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/token",
        data={"username": username, "password": password},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _fetch_me() -> Dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/me", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _fetch_permissions() -> Dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/permissions", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _ingest_docs() -> Dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}/ingest", headers=_get_auth_headers(), timeout=120)
    response.raise_for_status()
    return response.json()


def _fetch_ingest_status() -> Dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/ingest/status", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _fetch_admin_users() -> list[Dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/admin/users", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _create_admin_user(username: str, password: str, role: str, disabled: bool) -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/admin/users",
        headers={**_get_auth_headers(), "Content-Type": "application/json"},
        json={"username": username, "password": password, "role": role, "disabled": disabled},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _update_admin_user_role(username: str, role: str) -> Dict[str, Any]:
    response = requests.patch(
        f"{API_BASE_URL}/admin/users/{username}/role",
        headers={**_get_auth_headers(), "Content-Type": "application/json"},
        json={"role": role},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _delete_admin_user(username: str) -> Dict[str, Any]:
    response = requests.delete(f"{API_BASE_URL}/admin/users/{username}", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _fetch_documents() -> list[Dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/admin/documents", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _upload_document(uploaded_file, department: str) -> Dict[str, Any]:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), getattr(uploaded_file, "type", "text/plain"))}
    data = {"department": department}
    response = requests.post(
        f"{API_BASE_URL}/admin/documents/upload",
        headers=_get_auth_headers(),
        files=files,
        data=data,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _delete_document(document_id: int) -> Dict[str, Any]:
    response = requests.delete(f"{API_BASE_URL}/documents/{document_id}", headers=_get_auth_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def _poll_ingest_job(job_id: str, timeout_s: int = 120) -> Dict[str, Any]:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        response = requests.get(
            f"{API_BASE_URL}/ingest/jobs/{job_id}",
            headers=_get_auth_headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in {"completed", "failed"}:
            return payload
        time.sleep(1)
    raise TimeoutError("Ingest job did not finish in time")


def _chat(message: str, top_k: int, department: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message, "top_k": top_k}
    if department != "auto":
        payload["department"] = department
    response = requests.post(
        f"{API_BASE_URL}/chat",
        headers={**_get_auth_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _init_state() -> None:
    if "access_token" not in st.session_state:
        st.session_state.access_token = ""
    if "user" not in st.session_state:
        st.session_state.user = {}
    if "permissions" not in st.session_state:
        st.session_state.permissions = {}
    if "ingest_status" not in st.session_state:
        st.session_state.ingest_status = {}
    if "ingest_last_result" not in st.session_state:
        st.session_state.ingest_last_result = {}
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_view" not in st.session_state:
        st.session_state.active_view = "Chat"


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("RBAC Chatbot")
        st.caption(f"API: {API_BASE_URL}")

        if st.session_state.access_token:
            user = st.session_state.user
            st.success(f"Logged in as: {user.get('username', '-')}")
            st.write(f"Role: `{user.get('role', '-')}`")

            allowed = st.session_state.permissions.get("departments_allowed", [])
            if allowed:
                st.write("Allowed departments:")
                st.write(", ".join(allowed))

            ingest_status = {}
            if user.get("role") == "administrator":
                try:
                    ingest_status = _fetch_ingest_status()
                    st.session_state.ingest_status = ingest_status
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Failed to load ingest status: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Failed to load ingest status: {exc}")

            can_ingest = user.get("role") == "administrator" and bool(
                (ingest_status or st.session_state.get("ingest_status", {})).get("has_pending_documents")
            )
            if st.button("Ingest Documents", use_container_width=True, disabled=not can_ingest):
                try:
                    if not can_ingest:
                        st.info("No new documents to ingest.")
                    else:
                        result = _ingest_docs()
                        job_id = result.get("job_id")
                        if job_id:
                            st.info(f"Ingest job queued: `{job_id}`. Waiting for completion...")
                            final = _poll_ingest_job(job_id)
                            if final.get("status") == "completed":
                                st.session_state.ingest_last_result = {
                                    "status": "completed",
                                    "files_indexed": final.get("files_indexed", 0),
                                    "chunks_indexed": final.get("chunks_indexed", 0),
                                    "message": "Ingestion complete.",
                                }
                                st.success(
                                    f"Indexed {final.get('files_indexed')} files and {final.get('chunks_indexed')} chunks."
                                )
                            else:
                                st.session_state.ingest_last_result = {
                                    "status": "failed",
                                    "error": final.get("error", "unknown error"),
                                }
                                st.error(f"Ingest failed: {final.get('error', 'unknown error')}")
                        else:
                            st.warning("Ingest started, but no job id was returned.")
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Ingest failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Ingest failed: {exc}")
                except TimeoutError as exc:
                    st.error(str(exc))
            last_result = st.session_state.get("ingest_last_result", {})
            if last_result:
                st.caption("Last ingest result")
                if last_result.get("status") == "completed":
                    st.write(
                        f"Indexed {last_result.get('files_indexed', 0)} files and "
                        f"{last_result.get('chunks_indexed', 0)} chunks."
                    )
                else:
                    st.write(f"Last ingest failed: {last_result.get('error', 'unknown error')}")
            if user.get("role") == "administrator" and not can_ingest:
                st.caption("No new or modified documents are currently available for ingestion.")

            if user.get("role") == "administrator":
                st.divider()
                st.subheader("Admin Menu")
                st.session_state.active_view = st.radio(
                    "View",
                    options=["Chat", "Admin"],
                    index=0 if st.session_state.get("active_view") != "Admin" else 1,
                    label_visibility="collapsed",
                )
            else:
                st.session_state.active_view = "Chat"

            if st.button("Logout", use_container_width=True):
                st.session_state.access_token = ""
                st.session_state.user = {}
                st.session_state.permissions = {}
                st.session_state.ingest_status = {}
                st.session_state.ingest_last_result = {}
                st.session_state.history = []
                st.session_state.active_view = "Chat"
                st.rerun()
        else:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    try:
                        token_data = _login(username=username, password=password)
                        st.session_state.access_token = token_data["access_token"]
                        st.session_state.user = _fetch_me()
                        st.session_state.permissions = _fetch_permissions()
                        st.session_state.ingest_status = {}
                        st.session_state.ingest_last_result = {}
                        st.session_state.active_view = "Chat"
                        st.success("Login successful.")
                        st.rerun()
                    except requests.HTTPError as exc:
                        detail = exc.response.text if exc.response is not None else str(exc)
                        st.error(f"Login failed: {detail}")
                    except requests.RequestException as exc:
                        st.error(f"Login failed: {exc}")


def _render_chat() -> None:
    st.title("Internal RBAC RAG Chatbot")
    st.caption("Ask questions and get role-scoped answers with sources.")

    if not st.session_state.access_token:
        st.info("Log in from the sidebar to start chatting.")
        return

    allowed_departments: List[str] = st.session_state.permissions.get("departments_allowed", [])
    department_values = ["auto"] + [d for d in DEPARTMENT_OPTIONS[1:] if d in allowed_departments]
    selected_department = st.selectbox("Department Filter", options=department_values, index=0)
    top_k = st.slider("Top K Context Chunks", min_value=1, max_value=10, value=4, step=1)

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                st.markdown("**Sources**")
                for src in turn["sources"]:
                    st.write(f"- `{src}`")
            if turn["citations"]:
                with st.expander("Citations"):
                    st.json(turn["citations"])

    prompt = st.chat_input("Ask a question...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _chat(prompt, top_k=top_k, department=selected_department)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                citations = result.get("citations", [])
                st.write(answer)
                if sources:
                    st.markdown("**Sources**")
                    for src in sources:
                        st.write(f"- `{src}`")
                if citations:
                    with st.expander("Citations"):
                        st.json(citations)
                st.session_state.history.append(
                    {
                        "question": prompt,
                        "answer": answer,
                        "sources": sources,
                        "citations": citations,
                    }
                )
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Chat request failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Chat request failed: {exc}")


def _render_admin() -> None:
    st.title("Administrator Console")
    st.caption("Manage users, documents, ingestion, and role assignments.")

    if st.session_state.user.get("role") != "administrator":
        st.warning("Administrator access is required for this page.")
        return

    current_username = st.session_state.user.get("username", "")
    try:
        users = _fetch_admin_users()
        documents = _fetch_documents()
        ingest_status = _fetch_ingest_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Failed to load admin data: {detail}")
        return
    except requests.RequestException as exc:
        st.error(f"Failed to load admin data: {exc}")
        return

    user_tab, document_tab = st.tabs(["User Management", "Document Management"])

    with user_tab:
        st.subheader("Add User")
        with st.form("add_user_form", clear_on_submit=False):
            new_username = st.text_input("Username", key="admin_new_username")
            new_password = st.text_input("Password", type="password", key="admin_new_password")
            new_role = st.selectbox("Role", options=ROLE_OPTIONS, key="admin_new_role")
            new_disabled = st.checkbox("Disabled", value=False, key="admin_new_disabled")
            add_submitted = st.form_submit_button("Create User", use_container_width=True)
            if add_submitted:
                try:
                    _create_admin_user(new_username, new_password, new_role, new_disabled)
                    st.success(f"Created user '{new_username}'.")
                    st.rerun()
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Create user failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Create user failed: {exc}")

        st.subheader("Ingestion Queue")
        if ingest_status.get("has_pending_documents"):
            st.warning(ingest_status.get("message", "Pending document changes detected."))
            for department in ingest_status.get("departments", []):
                department_label = str(department.get("department", "")).title()
                new_documents = int(department.get("new_documents", 0))
                modified_documents = int(department.get("modified_documents", 0))
                pending_documents = int(department.get("pending_documents", 0))
                st.write(
                    f"- {department_label}: {pending_documents} pending "
                    f"({new_documents} new, {modified_documents} modified)"
                )
                documents_changed = department.get("documents", [])
                if documents_changed:
                    st.caption(", ".join(str(name) for name in documents_changed))
        else:
            st.info("No new documents to ingest.")

        st.subheader("User Dashboard")
        if users:
            st.dataframe(users, use_container_width=True, hide_index=True)
            manageable_users = [user["username"] for user in users if user["username"] != current_username]
            if manageable_users:
                selected_user = st.selectbox("Selected user", options=manageable_users, key="admin_selected_user")
                role_choice = st.selectbox("Assign role", options=ROLE_OPTIONS, key="admin_role_choice")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update Role", use_container_width=True, key="admin_update_role"):
                        try:
                            _update_admin_user_role(selected_user, role_choice)
                            st.success(f"Updated '{selected_user}' to role '{role_choice}'.")
                            st.rerun()
                        except requests.HTTPError as exc:
                            detail = exc.response.text if exc.response is not None else str(exc)
                            st.error(f"Role update failed: {detail}")
                        except requests.RequestException as exc:
                            st.error(f"Role update failed: {exc}")
                with col2:
                    if st.button("Delete User", use_container_width=True, key="admin_delete_user"):
                        try:
                            _delete_admin_user(selected_user)
                            st.success(f"Deleted user '{selected_user}'.")
                            st.rerun()
                        except requests.HTTPError as exc:
                            detail = exc.response.text if exc.response is not None else str(exc)
                            st.error(f"Delete user failed: {detail}")
                        except requests.RequestException as exc:
                            st.error(f"Delete user failed: {exc}")
            else:
                st.info("No deletable users available.")
        else:
            st.info("No users found.")

    with document_tab:
        st.subheader("Upload Document")
        upload_file = st.file_uploader("Choose a document", type=["md", "txt", "csv"], key="admin_upload_file")
        upload_department = st.selectbox(
            "Department",
            options=[d for d in DEPARTMENT_OPTIONS if d != "auto"],
            key="admin_upload_department",
        )
        if st.button("Upload Document", use_container_width=True, key="admin_upload_button"):
            if upload_file is None:
                st.error("Please choose a file first.")
            else:
                try:
                    result = _upload_document(upload_file, upload_department)
                    st.success(f"Uploaded document '{Path(result['source_path']).name}'.")
                    st.rerun()
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Upload failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")

        st.subheader("Document Dashboard")
        if documents:
            display_docs = [
                {
                    "id": doc["id"],
                    "filename": Path(doc["source_path"]).name,
                    "department": doc["department"],
                    "file_type": doc["file_type"],
                    "source_path": doc["source_path"],
                }
                for doc in documents
            ]
            st.dataframe(display_docs, use_container_width=True, hide_index=True)
            doc_lookup = {f"{doc['id']}: {Path(doc['source_path']).name} ({doc['department']})": doc["id"] for doc in documents}
            selected_doc_label = st.selectbox("Selected document", options=list(doc_lookup.keys()), key="admin_selected_doc")
            if st.button("Delete Document", use_container_width=True, key="admin_delete_doc"):
                try:
                    _delete_document(doc_lookup[selected_doc_label])
                    st.success("Document deleted.")
                    st.rerun()
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Delete failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Delete failed: {exc}")
        else:
            st.info("No documents found.")


def main() -> None:
    st.set_page_config(page_title="RBAC RAG Chatbot", layout="wide")
    _init_state()
    _render_sidebar()
    if st.session_state.get("active_view") == "Admin" and st.session_state.user.get("role") == "administrator":
        _render_admin()
    else:
        _render_chat()


if __name__ == "__main__":
    main()

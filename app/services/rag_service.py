import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import chromadb
from chromadb.api import Collection

from app.services.embedding_service import EmbeddingService
from app.services.postgres_store import (
    list_resource_documents,
    load_resource_document_content,
    sync_resource_documents_from_directory,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    source: str
    department: str


class RAGService:
    def __init__(
        self,
        role_department_access: Dict[str, Set[str]],
        embedder: EmbeddingService,
        vector_db_path: str | None = None,
        collection_name: str | None = None,
        chroma_host: str | None = None,
        chroma_port: int = 8000,
        chroma_ssl: bool = False,
    ) -> None:
        self.role_department_access = role_department_access
        self.embedder = embedder
        self.collection_name = collection_name or os.getenv("VECTOR_COLLECTION", "rbac_docs")
        self.min_relevance_score = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.55"))
        self.client = self._create_client(
            vector_db_path=vector_db_path,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            chroma_ssl=chroma_ssl,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _create_client(
        self,
        vector_db_path: str | None,
        chroma_host: str | None,
        chroma_port: int,
        chroma_ssl: bool,
    ):
        if chroma_host:
            return chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=chroma_ssl)

        db_path = vector_db_path or os.getenv("VECTOR_DB_PATH", "resources/vectorstore/chroma")
        return chromadb.PersistentClient(path=db_path)

    @property
    def chunk_count(self) -> int:
        return int(self.collection.count())

    def ingest_directory(self, base_data_dir: Path) -> Dict[str, int]:
        if not base_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {base_data_dir}")

        sync_resource_documents_from_directory(base_data_dir)
        return self.ingest_pending_documents()

    def rebuild_index(self) -> Dict[str, int]:
        self._reset_collection()
        chunks_indexed = 0
        documents = list_resource_documents()
        for document in documents:
            allowed_roles = self._allowed_roles_for_department(document.department)
            text_chunks = self._chunk_document_content(
                load_resource_document_content(document),
                file_type=document.file_type,
            )
            self._upsert_document_chunks(
                source_path=document.source_path,
                department=document.department,
                allowed_roles=allowed_roles,
                text_chunks=text_chunks,
                content_checksum=document.content_checksum,
            )
            chunks_indexed += len(text_chunks)
        return {"files_indexed": len(documents), "chunks_indexed": chunks_indexed}

    def ingest_pending_documents(self) -> Dict[str, int]:
        pending_documents = []
        for document in list_resource_documents():
            state = self._document_ingest_state(document)
            if state != "unchanged":
                pending_documents.append((document, state))

        files_indexed = 0
        chunks_indexed = 0
        for document, state in pending_documents:
            if state == "modified":
                self.delete_document_vectors(document.source_path)

            allowed_roles = self._allowed_roles_for_department(document.department)
            text_chunks = self._chunk_document_content(
                load_resource_document_content(document),
                file_type=document.file_type,
            )
            self._upsert_document_chunks(
                source_path=document.source_path,
                department=document.department,
                allowed_roles=allowed_roles,
                text_chunks=text_chunks,
                content_checksum=document.content_checksum,
            )
            files_indexed += 1
            chunks_indexed += len(text_chunks)

        return {"files_indexed": files_indexed, "chunks_indexed": chunks_indexed}

    def document_ingest_status(self) -> Dict[str, object]:
        totals = {
            "has_pending_documents": False,
            "pending_documents": 0,
            "new_documents": 0,
            "modified_documents": 0,
            "unchanged_documents": 0,
            "departments": [],
        }
        department_map: Dict[str, Dict[str, object]] = {}

        for document in list_resource_documents():
            state = self._document_ingest_state(document)
            if state == "unchanged":
                totals["unchanged_documents"] = int(totals["unchanged_documents"]) + 1
                continue

            totals["has_pending_documents"] = True
            totals["pending_documents"] = int(totals["pending_documents"]) + 1
            if state == "new":
                totals["new_documents"] = int(totals["new_documents"]) + 1
            else:
                totals["modified_documents"] = int(totals["modified_documents"]) + 1

            department_entry = department_map.setdefault(
                document.department,
                {
                    "department": document.department,
                    "new_documents": 0,
                    "modified_documents": 0,
                    "pending_documents": 0,
                    "documents": [],
                },
            )
            department_entry["pending_documents"] = int(department_entry["pending_documents"]) + 1
            department_entry[f"{state}_documents"] = int(department_entry[f"{state}_documents"]) + 1
            department_entry["documents"].append(Path(document.source_path).name)

        totals["departments"] = sorted(department_map.values(), key=lambda item: str(item["department"]))
        return totals

    def delete_document_vectors(self, source_path: str) -> None:
        self.collection.delete(where={"source": source_path})

    def retrieve(
        self,
        query: str,
        role: str,
        top_k: int = 4,
        department: str | None = None,
    ) -> List[RetrievedChunk]:
        if self.chunk_count == 0:
            return []

        preferred = self._exact_identifier_matches(query, role=role, department=department)
        preferred_ids = {chunk.chunk_id for chunk in preferred}

        query_vec = self.embedder.embed_query(query)
        where_filter: Dict[str, object] = {f"allow_{role}": True}
        if department:
            where_filter = {"$and": [where_filter, {"department": department}]}

        candidate_count = min(max(top_k * 5, 20), 50)

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=candidate_count,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        retrieved: List[tuple[float, float, RetrievedChunk]] = []
        for chunk_id, distance, text, metadata in zip(ids, distances, documents, metadatas):
            if str(chunk_id) in preferred_ids:
                continue
            vector_score = float(1.0 - distance)
            lexical_score = self._lexical_boost(query, str(text))
            combined_score = vector_score + lexical_score
            if combined_score < self.min_relevance_score:
                continue
            retrieved.append(
                (
                    combined_score,
                    vector_score,
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        score=vector_score,
                        text=str(text),
                        source=str((metadata or {}).get("source", "")),
                        department=str((metadata or {}).get("department", "")),
                    ),
                )
            )

        retrieved.sort(key=lambda item: (item[0], item[1]), reverse=True)
        combined = preferred + [item[2] for item in retrieved]
        return combined[:top_k]

    def _chunk_document_content(self, text: str, file_type: str) -> List[str]:
        if file_type.lower() == "csv":
            return self._chunk_csv_rows(text)
        return self._chunk_text(text)

    def _chunk_csv_rows(self, text: str) -> List[str]:
        rows = [line for line in text.splitlines() if line.strip()]
        if not rows:
            return []

        try:
            parsed_rows = list(csv.reader(rows))
        except csv.Error:
            return self._chunk_text(text)

        if not parsed_rows:
            return []

        header = [column.strip() for column in parsed_rows[0]]
        chunks: List[str] = []
        for row in parsed_rows[1:]:
            if not row:
                continue
            if header and len(row) == len(header):
                pairs = [f"{header[idx]}: {value.strip()}" for idx, value in enumerate(row)]
                chunks.append(" | ".join(pairs))
            else:
                chunks.append(" | ".join(part.strip() for part in row if part.strip()))
        return chunks or self._chunk_text(text)

    def _lexical_boost(self, query: str, text: str) -> float:
        query_lower = query.lower()
        text_lower = text.lower()

        boost = 0.0
        if query_lower in text_lower:
            boost += 2.0

        query_tokens = {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", query_lower)
            if token and token not in {"what", "is", "the", "name", "of", "employee", "id"}
        }
        if not query_tokens:
            return boost

        overlap = 0
        for token in query_tokens:
            if token in text_lower:
                overlap += 1
                if any(char.isdigit() for char in token):
                    boost += 5.0
                else:
                    boost += 0.5

        boost += min(overlap * 0.25, 1.0)
        return boost

    def _exact_identifier_matches(self, query: str, role: str, department: str | None) -> List[RetrievedChunk]:
        query_lower = query.lower()
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+", query)
            if any(char.isdigit() for char in token)
        ]
        if not tokens:
            return []

        matches: List[RetrievedChunk] = []
        for document in list_resource_documents():
            if department and document.department != department:
                continue
            if not self._allowed_roles_for_department(document.department) or role not in self._allowed_roles_for_department(
                document.department
            ):
                continue

            content = load_resource_document_content(document)
            chunks = self._chunk_document_content(content, document.file_type)
            for idx, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                if "employee id" in query_lower or "employee_id" in query_lower:
                    if all(chunk_lower.startswith(f"employee_id: {token}") for token in tokens):
                        matches.append(
                            RetrievedChunk(
                                chunk_id=f"{document.department}:{Path(document.source_path).name}:{idx}",
                                score=1.0,
                                text=chunk,
                                source=document.source_path,
                                department=document.department,
                            )
                        )
                    continue

                if all(token in chunk_lower for token in tokens):
                    matches.append(
                        RetrievedChunk(
                            chunk_id=f"{document.department}:{Path(document.source_path).name}:{idx}",
                            score=1.0,
                            text=chunk,
                            source=document.source_path,
                            department=document.department,
                        )
                    )

        return matches

    def _allowed_roles_for_department(self, department: str) -> Set[str]:
        return {
            role
            for role, allowed_departments in self.role_department_access.items()
            if department in allowed_departments
        }

    def _chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(cleaned)
        while start < text_len:
            end = min(text_len, start + chunk_size)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _upsert_document_chunks(
        self,
        source_path: str,
        department: str,
        allowed_roles: Set[str],
        text_chunks: List[str],
        content_checksum: str,
        batch_size: int = 64,
    ) -> None:
        if not text_chunks:
            return

        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i : i + batch_size]
            batch_embeddings = self.embedder.embed_texts(batch_chunks)
            ids: List[str] = []
            metadatas: List[dict] = []
            for offset, _ in enumerate(batch_chunks):
                idx = i + offset
                chunk_id = f"{department}:{Path(source_path).name}:{idx}"
                ids.append(chunk_id)
                metadata = {
                    "source": source_path,
                    "department": department,
                    "content_checksum": content_checksum,
                }
                for role_name in self.role_department_access:
                    metadata[f"allow_{role_name}"] = role_name in allowed_roles
                metadatas.append(metadata)
            self.collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

    def _document_ingest_state(self, document) -> str:
        current_chunks = self._chunk_document_content(
            load_resource_document_content(document),
            file_type=document.file_type,
        )
        existing = self.collection.get(where={"source": document.source_path}, include=["metadatas"])
        existing_metadatas = [metadata for metadata in (existing.get("metadatas") or []) if metadata]
        if not existing_metadatas:
            return "new"

        if len(existing_metadatas) != len(current_chunks):
            return "modified"

        if all(metadata.get("content_checksum") == document.content_checksum for metadata in existing_metadatas):
            return "unchanged"

        return "modified"

    def _reset_collection(self) -> Collection:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

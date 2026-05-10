from __future__ import annotations

import csv
import hashlib
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.users_data import DEFAULT_USERS, SUPPORTED_ROLES


Base = declarative_base()


def _default_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./resources/local.db")


def _document_store_root() -> Path:
    return Path(os.getenv("DOCUMENT_STORE_PATH", "resources/document_store"))


def _bootstrap_document_store_root() -> Path:
    return _document_store_root() / "bootstrap"


def _uploaded_document_store_root() -> Path:
    return _document_store_root() / "uploads"


def _create_engine(database_url: str | None = None) -> Engine:
    url = database_url or _default_database_url()
    engine_kwargs: dict[str, object] = {"future": True}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if url.endswith(":memory:"):
            engine_kwargs["poolclass"] = StaticPool
    return create_engine(url, **engine_kwargs)


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


class UserRecord(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    disabled = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ResourceDocument(Base):
    __tablename__ = "resource_documents"

    id = Column(Integer, primary_key=True)
    source_path = Column(String(500), unique=True, nullable=False, index=True)
    department = Column(String(50), nullable=False, index=True)
    file_type = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    content_checksum = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    Base.metadata.create_all(bind=engine)


def seed_demo_users(password_hasher) -> int:
    inserted = 0
    with session_scope() as session:
        for seed in DEFAULT_USERS:
            existing = session.execute(
                select(UserRecord).where(UserRecord.username == seed.username)
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    UserRecord(
                        username=seed.username,
                        hashed_password=password_hasher(seed.password),
                        role=seed.role,
                        disabled=False,
                    )
                )
                inserted += 1
    return inserted


def get_user_by_username(username: str) -> UserRecord | None:
    with get_session() as session:
        return session.execute(
            select(UserRecord).where(UserRecord.username == username)
        ).scalar_one_or_none()


def count_users() -> int:
    with get_session() as session:
        return session.query(UserRecord).count()


def count_resource_documents() -> int:
    with get_session() as session:
        return session.query(ResourceDocument).count()


def list_users() -> list[UserRecord]:
    with get_session() as session:
        return list(session.execute(select(UserRecord).order_by(UserRecord.username)).scalars().all())


def create_user(username: str, hashed_password: str, role: str, disabled: bool = False) -> UserRecord:
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role '{role}'")
    with session_scope() as session:
        existing = session.execute(select(UserRecord).where(UserRecord.username == username)).scalar_one_or_none()
        if existing is not None:
            raise ValueError(f"User '{username}' already exists")
        record = UserRecord(username=username, hashed_password=hashed_password, role=role, disabled=disabled)
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def update_user_role(username: str, role: str) -> UserRecord:
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role '{role}'")
    with session_scope() as session:
        record = session.execute(select(UserRecord).where(UserRecord.username == username)).scalar_one_or_none()
        if record is None:
            raise LookupError(f"User '{username}' not found")
        record.role = role
        session.flush()
        session.refresh(record)
        return record


def delete_user(username: str) -> bool:
    with session_scope() as session:
        record = session.execute(select(UserRecord).where(UserRecord.username == username)).scalar_one_or_none()
        if record is None:
            return False
        session.delete(record)
        return True


def ping_database() -> bool:
    with engine.connect() as connection:
        connection.exec_driver_sql("SELECT 1")
    return True


def _file_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_file_content(file_path: Path) -> str:
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            lines = [", ".join(row).strip() for row in reader if row]
            return "\n".join(lines)
    with file_path.open("r", encoding="utf-8") as text_file:
        return text_file.read()


def _document_store_path(base_data_dir: Path, file_path: Path, *, root: Path) -> Path:
    relative = file_path.relative_to(base_data_dir)
    return root / relative


def _copy_document_to_store(base_data_dir: Path, file_path: Path, *, root: Path) -> Path:
    stored_path = _document_store_path(base_data_dir, file_path, root=root)
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, stored_path)
    return stored_path


def _copy_uploaded_document_to_store(filename: str, content: bytes) -> Path:
    safe_name = Path(filename).name or "document"
    timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    stored_path = _uploaded_document_store_root() / f"{timestamp_ns}_{safe_name}"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)
    return stored_path


def load_resource_document_content(document: ResourceDocument) -> str:
    if document.content:
        return document.content
    with Path(document.source_path).open("r", encoding="utf-8") as text_file:
        return text_file.read()


def sync_resource_documents_from_directory(base_data_dir: Path) -> int:
    if not base_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {base_data_dir}")

    upserted = 0
    with session_scope() as session:
        bootstrap_prefix = str(_bootstrap_document_store_root())
        session.query(ResourceDocument).filter(ResourceDocument.source_path.like(f"{bootstrap_prefix}%")).delete(
            synchronize_session=False
        )
        shutil.rmtree(_bootstrap_document_store_root(), ignore_errors=True)
        for file_path in sorted(base_data_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt", ".csv"}:
                continue

            stored_path = _copy_document_to_store(base_data_dir, file_path, root=_bootstrap_document_store_root())
            relative = file_path.relative_to(base_data_dir)
            department = relative.parts[0]
            content = _read_file_content(file_path)
            checksum = _file_checksum(content)
            session.add(
                ResourceDocument(
                    source_path=str(stored_path),
                    department=department,
                    file_type=file_path.suffix.lower().lstrip("."),
                    content=content,
                    content_checksum=checksum,
                )
            )
            upserted += 1
    return upserted


def create_uploaded_resource_document(
    filename: str,
    department: str,
    content: bytes,
) -> ResourceDocument:
    stored_path = _copy_uploaded_document_to_store(filename, content)
    file_type = stored_path.suffix.lower().lstrip(".") or "txt"
    try:
        text_content = stored_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded document must be UTF-8 text, CSV, or markdown") from exc
    checksum = _file_checksum(text_content)
    with session_scope() as session:
        record = ResourceDocument(
            source_path=str(stored_path),
            department=department,
            file_type=file_type,
            content=text_content,
            content_checksum=checksum,
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def list_resource_documents() -> list[ResourceDocument]:
    with get_session() as session:
        return list(
            session.execute(select(ResourceDocument).order_by(ResourceDocument.department, ResourceDocument.source_path))
            .scalars()
            .all()
        )


def get_resource_document_by_id(document_id: int) -> ResourceDocument | None:
    with get_session() as session:
        return session.execute(
            select(ResourceDocument).where(ResourceDocument.id == document_id)
        ).scalar_one_or_none()


def delete_resource_document(document_id: int) -> ResourceDocument | None:
    with session_scope() as session:
        record = session.execute(
            select(ResourceDocument).where(ResourceDocument.id == document_id)
        ).scalar_one_or_none()
        if record is None:
            return None
        stored_path = Path(record.source_path)
        session.delete(record)
    try:
        stored_path.unlink(missing_ok=True)
        parent = stored_path.parent
        while parent != _document_store_root() and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass
    return record

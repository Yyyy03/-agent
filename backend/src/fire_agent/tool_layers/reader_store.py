from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ReaderDocumentStore:
    """Thread-local full-text cache shared by reader tools during one task."""

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def data(self) -> Dict[str, Dict[str, Any]]:
        data = getattr(self._local, "data", None)
        if data is None:
            data = {}
            self._local.data = data
        return data

    @property
    def source_index(self) -> Dict[str, str]:
        index = getattr(self._local, "source_index", None)
        if index is None:
            index = {}
            self._local.source_index = index
        return index

    def clear(self) -> None:
        self.data.clear()
        self.source_index.clear()
        self._local.next_id = 0

    def key(self, namespace: str, source: str) -> str:
        return f"{namespace}:{str(source or '').strip()}"

    def get(self, namespace: str, source: str) -> Optional[Dict[str, Any]]:
        document_key = self.source_index.get(self.key(namespace, source))
        return self.data.get(document_key or "")

    def get_key(self, key: str) -> Optional[Dict[str, Any]]:
        normalized = str(key or "").strip()
        return self.data.get(normalized) or self.data.get(self.source_index.get(normalized, ""))

    def _next_document_key(self, namespace: str) -> str:
        next_id = int(getattr(self._local, "next_id", 0)) + 1
        self._local.next_id = next_id
        return f"{namespace}:{next_id}"

    def put(
        self,
        namespace: str,
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_key = self.key(namespace, source)
        document_key = self.source_index.get(source_key) or self._next_document_key(namespace)
        record = {
            "key": document_key,
            "source_key": source_key,
            "namespace": namespace,
            "source": str(source or "").strip(),
            "content": content or "",
            "chars": len(content or ""),
            "metadata": dict(metadata or {}),
        }
        self.data[record["key"]] = record
        self.source_index[source_key] = record["key"]
        return record


READER_DOCUMENT_STORE = ReaderDocumentStore()

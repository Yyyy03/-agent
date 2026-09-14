from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


SEC_HOSTS = {"sec.gov", "www.sec.gov", "data.sec.gov"}


def is_sec_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url).strip())
    except Exception:
        return False
    host = (parsed.netloc or "").lower().split(":", 1)[0]
    return host in SEC_HOSTS or host.endswith(".sec.gov")


def sec_archive_path_cik(cik: Any) -> str:
    return str(int(str(cik).strip()))


def sec_accession_no_dashes(accession: Any) -> str:
    text = str(accession or "").strip()
    if not text:
        raise ValueError("Missing accession")
    return text.replace("-", "")


def sec_primary_document_url(cik: Any, accession: Any, primary_document: Any) -> str:
    doc = str(primary_document or "").strip()
    if not doc:
        raise ValueError("Missing primary_document")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{sec_archive_path_cik(cik)}/{sec_accession_no_dashes(accession)}/{doc}"
    )


def sec_accession_index_url(cik: Any, accession: Any) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{sec_archive_path_cik(cik)}/{sec_accession_no_dashes(accession)}/index.html"
    )


def is_sec_accession_landing_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    path = parsed.path.rstrip("/")
    if path.endswith(("/index.html", "/index.htm")):
        return True
    return bool(re.search(r"/Archives/edgar/data/\d+/\d+$", path))

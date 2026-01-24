# docstore/__init__.py
# -*- coding: utf-8 -*-
"""
docstore package exports.

讓你可以在 pages/Anya_Test.py 用：
from docstore import FileRow, build_indices_incremental, ... 以及 badges_markdown
"""

from .docstore import (
    FileRow,
    DocStore,
    build_file_row_from_bytes,
    build_indices_incremental,
    doc_list_payload,
    doc_search_payload,
    doc_get_fulltext_payload,
    HAS_UNSTRUCTURED_LOADERS,
    HAS_PYMUPDF,
    HAS_FLASHRANK,
    HAS_BM25,
    estimate_tokens_from_chars,
)

from .badges import badges_markdown

__all__ = [
    "FileRow",
    "DocStore",
    "build_file_row_from_bytes",
    "build_indices_incremental",
    "doc_list_payload",
    "doc_search_payload",
    "doc_get_fulltext_payload",
    "HAS_UNSTRUCTURED_LOADERS",
    "HAS_PYMUPDF",
    "HAS_FLASHRANK",
    "HAS_BM25",
    "estimate_tokens_from_chars",
    "badges_markdown",
]

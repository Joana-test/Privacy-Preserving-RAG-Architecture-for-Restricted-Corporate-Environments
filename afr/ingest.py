"""
Document ingestion: PDF and CSV extraction, chunking, rule-based tagging via
tagging.py, and FAISS indexing. Used by the interactive pipeline; the
evaluation scripts bypass this path and inject the pre-tagged test corpus
directly.

Taken from Namboothiri et al. (2026),
licensed under the Apache License, Version 2.0.
"""

import os
import csv
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from afr.tagging import ChunkMetadata, tag_chunk, get_tagging_summary

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_FILE_SIZE_MB = 15
MAX_FILES = 5
SUPPORTED_EXTENSIONS = {".pdf", ".csv"}


@dataclass
class IngestionResult:
    success: bool
    message: str
    chunks: List[ChunkMetadata]
    index: Optional[Any] = None
    tagging_summary: Optional[Dict] = None


class DocumentIngester:

    def __init__(
            self,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
            embed_model_name: str = DEFAULT_EMBED_MODEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name
        self._embed_model = None
        self._index = None
        self._chunks: List[ChunkMetadata] = []
        self._chunk_map: Dict[int, ChunkMetadata] = {}

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    @property
    def index(self) -> Optional[faiss.Index]:
        return self._index

    @property
    def chunks(self) -> List[ChunkMetadata]:
        return self._chunks

    def reset(self):
        self._index = None
        self._chunks = []
        self._chunk_map = {}

    def _generate_doc_id(self, filename: str, content: str) -> str:
        hash_input = f"{filename}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _extract_text_from_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _extract_text_from_csv(self, file_path: str) -> str:
        text_parts = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(8192)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)
                headers = reader.fieldnames or []

                if headers:
                    text_parts.append(f"CSV Headers: {', '.join(headers)}")
                    text_parts.append("")

                for row_num, row in enumerate(reader, 1):
                    row_text_parts = []
                    for key, value in row.items():
                        if value and str(value).strip():
                            row_text_parts.append(f"{key}: {value}")

                    if row_text_parts:
                        row_text = f"Row {row_num}: " + " | ".join(row_text_parts)
                        text_parts.append(row_text)

        except Exception as e:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

        return "\n".join(text_parts)

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                for sep in ["\n\n", "\n", ". ", " "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap if end < text_len else text_len

        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def _get_file_extension(self, file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower()

    def ingest_file(self, file_path: str) -> IngestionResult:
        filename = os.path.basename(file_path)
        ext = self._get_file_extension(file_path)

        if ext not in SUPPORTED_EXTENSIONS:
            return IngestionResult(
                success=False,
                message=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                chunks=[]
            )

        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return IngestionResult(
                    success=False,
                    message=f"File {filename} exceeds {MAX_FILE_SIZE_MB}MB limit",
                    chunks=[]
                )

            if ext == ".pdf":
                text = self._extract_text_from_pdf(file_path)
            elif ext == ".csv":
                text = self._extract_text_from_csv(file_path)
            else:
                return IngestionResult(
                    success=False,
                    message=f"Unsupported file type: {ext}",
                    chunks=[]
                )

            if not text.strip():
                return IngestionResult(
                    success=False,
                    message=f"No text extracted from {filename}",
                    chunks=[]
                )

            doc_id = self._generate_doc_id(filename, text)
            text_chunks = self._chunk_text(text)

            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{doc_id}_{i:04d}"
                chunk = tag_chunk(
                    text=chunk_text,
                    filename=filename,
                    doc_id=doc_id,
                    chunk_id=chunk_id
                )
                chunks.append(chunk)

            return IngestionResult(
                success=True,
                message=f"Ingested {len(chunks)} chunks from {filename}",
                chunks=chunks,
                tagging_summary=get_tagging_summary(chunks)
            )

        except Exception as e:
            return IngestionResult(
                success=False,
                message=f"Error processing {filename}: {str(e)}",
                chunks=[]
            )

    def ingest_pdf(self, file_path: str) -> IngestionResult:
        return self.ingest_file(file_path)

    def ingest_files(self, file_paths: List[str]) -> IngestionResult:
        if len(file_paths) > MAX_FILES:
            return IngestionResult(
                success=False,
                message=f"Maximum {MAX_FILES} files allowed",
                chunks=[]
            )

        all_chunks = []
        messages = []

        for file_path in file_paths:
            result = self.ingest_file(file_path)
            if result.success:
                all_chunks.extend(result.chunks)
                messages.append(result.message)
            else:
                messages.append(f"⚠️ {result.message}")

        if not all_chunks:
            return IngestionResult(
                success=False,
                message="No chunks extracted from any files",
                chunks=[]
            )

        texts = [chunk.text for chunk in all_chunks]
        embeddings = self._embed_texts(texts)

        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i].tolist()

        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        self._chunks = all_chunks
        self._chunk_map = {i: chunk for i, chunk in enumerate(all_chunks)}

        return IngestionResult(
            success=True,
            message="\n".join(messages),
            chunks=all_chunks,
            index=self._index,
            tagging_summary=get_tagging_summary(all_chunks)
        )

    def ingest_pdfs(self, file_paths: List[str]) -> IngestionResult:
        return self.ingest_files(file_paths)

    def search(self, query: str, k: int = 10) -> List[ChunkMetadata]:
        if self._index is None or not self._chunks:
            return []

        query_embedding = self._embed_texts([query])
        scores, indices = self._index.search(query_embedding, min(k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self._chunk_map:
                chunk = self._chunk_map[idx]
                chunk.score = float(score)
                results.append(chunk)

        return results

    def search_within_chunks(self, query: str, chunks: List[ChunkMetadata], k: int = 10) -> List[ChunkMetadata]:
        """
        Strict-AFR helper: rank ONLY within a pre-authorized chunk pool.

        Builds a temporary FAISS index over the provided chunks' stored embeddings,
        and returns the top-k chunks by similarity to the query embedding.
        """
        if not chunks:
            return []

        kept_chunks: List[ChunkMetadata] = []
        vecs: List[np.ndarray] = []
        for c in chunks:
            if getattr(c, "embedding", None) is None:
                continue
            try:
                v = np.asarray(c.embedding, dtype=np.float32)
            except Exception:
                continue
            if v.ndim != 1:
                continue
            kept_chunks.append(c)
            vecs.append(v)

        if not kept_chunks:
            return []

        emb = np.stack(vecs, axis=0).astype(np.float32, copy=False)
        dimension = emb.shape[1]
        tmp_index = faiss.IndexFlatIP(dimension)
        tmp_index.add(emb)

        query_embedding = self._embed_texts([query])
        scores, indices = tmp_index.search(query_embedding, min(k, len(kept_chunks)))

        results: List[ChunkMetadata] = []
        for score, local_idx in zip(scores[0], indices[0]):
            if local_idx < 0:
                continue
            chunk = kept_chunks[int(local_idx)]
            chunk.score = float(score)
            results.append(chunk)

        return results


_global_ingester: Optional[DocumentIngester] = None


def get_ingester() -> DocumentIngester:
    global _global_ingester
    if _global_ingester is None:
        _global_ingester = DocumentIngester()
    return _global_ingester


def reset_ingester():
    global _global_ingester
    if _global_ingester:
        _global_ingester.reset()
    _global_ingester = DocumentIngester()


def ingest_file(file_path: str) -> IngestionResult:
    return get_ingester().ingest_file(file_path)


def ingest_pdf(file_path: str) -> IngestionResult:
    return get_ingester().ingest_file(file_path)


def ingest_files(file_paths: List[str]) -> IngestionResult:
    return get_ingester().ingest_files(file_paths)


def ingest_pdfs(file_paths: List[str]) -> IngestionResult:
    return get_ingester().ingest_files(file_paths)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    ingester = DocumentIngester(chunk_size=chunk_size, chunk_overlap=overlap)
    return ingester._chunk_text(text)

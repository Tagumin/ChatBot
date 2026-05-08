import os
import sys
import time
import math
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass

try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama as OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    RERANKER_IMPORTED = True
except ImportError:
    RERANKER_IMPORTED = False

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
LLM_MODEL            = "deepseek-llm:7b" 
EMBED_MODEL          = "nomic-embed-text"
VECTOR_DB            = "./vectorstore"
COLLECTION_NAME      = "uu_indonesia"

TOP_K                = 6
FINAL_K              = 3

MAX_RETRIES          = 2
TEMPERATURE          = 0.2

MAX_DOC_LENGTH       = 500
MAX_TOTAL_CONTEXT    = 2500

CONFIDENCE_THRESHOLD = 0.45

ENABLE_RERANK        = False
RERANKER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ─────────────────────────────


# =========================
# UTILITIES
# =========================
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def clean_text(text: str) -> str:
    """Clean noise from text without removing article structure."""
    if not text:
        return ""

    text = re.sub(r'={5,}[^\n]*={5,}', '', text)
    text = re.sub(r'DOCUMENT\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def is_quality_content(text: str) -> bool:
    text = text.strip()
    if len(text) < 30:
        return False
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return len(words) >= 5


def truncate_content(text: str, max_len: int = MAX_DOC_LENGTH) -> str:
    """Trim text at the nearest sentence boundary."""
    text = text.strip()
    if len(text) <= max_len:
        return text

    truncated = text[:max_len]
    for delim in ['. ', '? ', '! ', ';\n', '.\n']:
        last_pos = truncated.rfind(delim)
        if last_pos > max_len * 0.5:
            return truncated[:last_pos + 1].strip()

    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.7:
        return truncated[:last_space].strip()

    return truncated.strip()


# =========================
# ARTICLE INFO EXTRACTOR
# =========================
def extract_article_info(text: str) -> Dict[str, str]:
    """
    Parse structure from chunk:
        Book One - General Rules
        Chapter I - Limits of ...
        Article 1
        (content)
    """
    info = {"book": "", "chapter": "", "article": "", "label": ""}

    book_m = re.search(r'(Book\s+\w+(?:\s*-\s*[^\n]+)?)', text, re.IGNORECASE)
    if book_m:
        info["book"] = book_m.group(1).strip()

    chap_m = re.search(r'(Chapter\s+\w+(?:\s*-\s*[^\n]+)?)', text, re.IGNORECASE)
    if chap_m:
        info["chapter"] = chap_m.group(1).strip()

    art_m = re.search(r'[Aa]rticle\s+(\d+)', text)
    if art_m:
        info["article"] = f"Article {art_m.group(1)}"

    parts = [p for p in [info["article"], info["chapter"], info["book"]] if p]
    info["label"] = " | ".join(parts) if parts else "unknown"

    return info


# =========================
# RETRIEVED DOC
# =========================
@dataclass
class RetrievedDoc:
    document: Document
    embedding_rank: int = 0
    bm25_rank: int = 0
    rerank_score: float = 0.0
    hybrid_score: float = 0.0
    bm25_score: float = 0.0

    @property
    def content(self) -> str:
        return getattr(self.document, "page_content", "") or ""

    @property
    def source(self) -> str:
        info = extract_article_info(self.content)
        if info["label"] and info["label"] != "unknown":
            meta = self.document.metadata or {}
            file_src = meta.get("source_file") or meta.get("source") or ""
            return f"{info['label']}" + (f" [{file_src}]" if file_src else "")
        meta = self.document.metadata or {}
        return meta.get("source_file") or meta.get("source") or "unknown"

    @property
    def is_meaningful(self) -> bool:
        text = self.content.strip()
        if len(text) < 20:
            return False

        body = re.sub(r'={5,}.*?={5,}', '', text)
        body = re.sub(
            r'(Book\s+\w+[^\n]*|Chapter\s+\w+[^\n]*|Article\s+\d+[^\n]*)',
            '',
            body,
            flags=re.IGNORECASE
        )

        STOP = {
            "the", "and", "or", "of", "in", "to", "is", "are", "was",
            "were", "a", "an", "as", "at", "by", "for", "on", "that",
            "this", "with", "which", "from", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "can", "could",
            "not", "no", "nor", "so", "if", "but", "than", "then",
            "its", "it", "they", "them", "their", "who", "whom",
        }
        words = re.findall(r'[a-zA-Z]{3,}', body.lower())
        meaningful = [w for w in words if w not in STOP]
        return len(meaningful) >= 5


# =========================
# SYSTEM LOADER
# =========================
class LegalRAGSystem:
    def __init__(self):
        self.llm = None
        self.retriever = None
        self.vectorstore = None
        self.embeddings = None
        self.cross_encoder = None
        self.bm25 = None
        self.all_docs: List[Document] = []
        self.doc_texts: List[str] = []
        self._load_system()

    def _load_system(self):
        print("🔧 Loading system components...")

        try:
            self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            print(f"   ✅ Embeddings: {EMBED_MODEL}")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            sys.exit(1)

        try:
            self.vectorstore = Chroma(
                persist_directory=VECTOR_DB,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            print(f"   ✅ Vector DB: {VECTOR_DB}")
        except Exception as e:
            print(f"❌ Failed to load vector DB: {e}")
            sys.exit(1)

        try:
            self.llm = OllamaLLM(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
            )
            print(f"   ✅ LLM: {LLM_MODEL}")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            sys.exit(1)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        if ENABLE_RERANK and RERANKER_IMPORTED:
            try:
                self.cross_encoder = CrossEncoder(RERANKER_MODEL)
                print(f"   ✅ Reranker: {RERANKER_MODEL}")
            except Exception as e:
                print(f"⚠️  Failed to load reranker: {e}")
                self.cross_encoder = None
        else:
            self.cross_encoder = None
            print("   ⚡ Reranker disabled for speed")

        self._build_bm25_index()
        print("✅ System ready!\n")

    def _build_bm25_index(self):
        try:
            all_data = self.vectorstore.get()
            texts = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])

            if not texts:
                print("   ⚠️  BM25: No documents found")
                return

            self.doc_texts = []
            self.all_docs = []

            for text, meta in zip(texts, metadatas):
                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 20:
                    self.doc_texts.append(cleaned)
                    self.all_docs.append(Document(page_content=cleaned, metadata=meta or {}))

            if self.doc_texts:
                tokenized = [tokenize(doc) for doc in self.doc_texts]
                self.bm25 = BM25Okapi(tokenized)
                print(f"   ✅ BM25 index: {len(self.doc_texts)} documents")
            else:
                print("   ⚠️  BM25: No valid documents")
        except Exception as e:
            print(f"   ⚠️  BM25 error: {e}")
            self.bm25 = None

    def hybrid_search(self, query: str) -> List[RetrievedDoc]:
        query_lower = query.lower()

        # 1) Embedding search
        try:
            embed_docs = self.retriever.invoke(query)
            print(f"🔍 Embedding: {len(embed_docs)} documents")
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            embed_docs = []

        # 2) BM25 search
        bm25_results: List[RetrievedDoc] = []
        if self.bm25:
            try:
                tokenized_query = tokenize(query_lower)
                scores = self.bm25.get_scores(tokenized_query)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
                bm25_rank = 0
                for i in top_idx:
                    if scores[i] > 0:
                        bm25_rank += 1
                        bm25_results.append(
                            RetrievedDoc(
                                document=self.all_docs[i],
                                bm25_rank=bm25_rank,
                                bm25_score=float(scores[i]),
                            )
                        )
                print(f"🔍 BM25: {len(bm25_results)} documents")
            except Exception as e:
                print(f"⚠️  BM25 error: {e}")

        # 3) Merge with deduplication
        doc_map: Dict[int, RetrievedDoc] = {}

        for i, doc in enumerate(embed_docs, 1):
            content = getattr(doc, "page_content", "")
            doc_id = hash(content[:200])
            doc_map[doc_id] = RetrievedDoc(
                document=doc,
                embedding_rank=i
            )

        for doc in bm25_results:
            doc_id = hash(doc.content[:200])
            if doc_id in doc_map:
                doc_map[doc_id].bm25_rank = doc.bm25_rank
                doc_map[doc_id].bm25_score = doc.bm25_score
            else:
                doc_map[doc_id] = doc

        # 4) Hybrid score with proper RRF
        k_rrf = 60
        results = []
        for doc in doc_map.values():
            emb_rrf = 1.0 / (doc.embedding_rank + k_rrf) if doc.embedding_rank > 0 else 0.0
            bm_rrf = 1.0 / (doc.bm25_rank + k_rrf) if doc.bm25_rank > 0 else 0.0
            doc.hybrid_score = emb_rrf + bm_rrf
            results.append(doc)

        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        results = [d for d in results if d.is_meaningful]
        return results[:TOP_K]

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if not self.cross_encoder or len(docs) <= 1:
            return docs

        print(f"🔄 Reranking {len(docs)} documents...")
        pairs = [[query, d.content] for d in docs]

        try:
            scores = self.cross_encoder.predict(pairs)
            for doc, score in zip(docs, scores):
                doc.rerank_score = float(score)

            docs.sort(key=lambda x: x.rerank_score, reverse=True)
            filtered = [d for d in docs if d.rerank_score > -2.0]
            if len(filtered) < 3:
                filtered = docs[:max(3, len(docs) // 2)]

            print(f"   ✅ Reranking: {len(filtered)} documents (filtered)")
            return filtered[:FINAL_K]
        except Exception as e:
            print(f"⚠️  Reranking error: {e}")
            return docs[:FINAL_K]

    def calculate_confidence(self, docs: List[RetrievedDoc]) -> Dict[str, float]:
        if not docs:
            return {"overall": 0.0, "retrieval": 0.0, "rerank": 0.0, "quality": 0.0}

        emb_scores = [1.0 / d.embedding_rank for d in docs if d.embedding_rank > 0]
        bm_scores = [1.0 / d.bm25_rank for d in docs if d.bm25_rank > 0]

        retrieval_cf = 0.0
        if emb_scores and bm_scores:
            retrieval_cf = (sum(emb_scores) / len(emb_scores) + sum(bm_scores) / len(bm_scores)) / 2
        elif emb_scores:
            retrieval_cf = sum(emb_scores) / len(emb_scores)
        elif bm_scores:
            retrieval_cf = sum(bm_scores) / len(bm_scores)

        r_scores = [d.rerank_score for d in docs if d.rerank_score != 0]
        if r_scores:
            avg_r = sum(r_scores) / len(r_scores)
            rerank_cf = 1 / (1 + math.exp(-avg_r * 0.5))
        else:
            rerank_cf = 0.0

        quality = sum(1 for d in docs if d.is_meaningful) / len(docs)
        overall = (retrieval_cf * 0.25) + (rerank_cf * 0.45) + (quality * 0.30)

        return {
            "overall": round(min(overall, 1.0), 3),
            "retrieval": round(min(retrieval_cf, 1.0), 3),
            "rerank": round(min(rerank_cf, 1.0), 3),
            "quality": round(quality, 3),
        }


# =========================
# PROMPT
# =========================
template = """You are an Indonesian legal assistant.

Rules:
1. Answer ONLY from the provided context.
2. Do not invent laws or article numbers.
3. Cite Book, Chapter, and Article when available.
4. If information is missing, say: "Not found in the available legal sources."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(template)


# =========================
# HELPERS
# =========================
def build_context(docs: List[RetrievedDoc], max_total_chars: int = MAX_TOTAL_CONTEXT) -> str:
    """
    Build context with Book/Chapter/Article labels from chunk structure.
    """
    if not docs:
        return ""

    parts = []
    current_len = 0

    for i, doc in enumerate(docs, 1):
        raw = doc.content
        content = clean_text(raw)
        if not is_quality_content(content):
            continue

        info = extract_article_info(raw)
        label_parts = [p for p in [info["article"], info["chapter"], info["book"]] if p]
        label = " | ".join(label_parts) if label_parts else doc.source

        content = truncate_content(content, MAX_DOC_LENGTH)
        entry = f"[DOCUMENT {i} | {label}]\n{content}\n"

        if current_len + len(entry) > max_total_chars:
            short = truncate_content(content, 400)
            s_entry = f"[DOCUMENT {i} | {label}]\n{short}\n"
            if current_len + len(s_entry) <= max_total_chars:
                parts.append(s_entry)
                current_len += len(s_entry)
            break

        parts.append(entry)
        current_len += len(entry)

    return "\n\n".join(parts)


def invoke_with_retry(chain, inputs: dict, max_retries: int = MAX_RETRIES) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            print(f"\n⚠️  Error ({attempt}/{max_retries}): {e}")
            time.sleep(1.5)
    return ""


def apply_fallback(confidence: Dict[str, float], docs: List[RetrievedDoc]) -> Tuple[bool, str]:
    overall = confidence.get("overall", 0)
    quality = confidence.get("quality", 0)
    rerank = confidence.get("rerank", 0)

    if overall < 0.25 or quality < 0.3:
        return False, (
            "⚠️  **Cannot answer** - Confidence too low (%.2f)\n"
            "The retrieved documents do not contain sufficiently meaningful content.\n"
            "💡 Tip: Rephrase your question or add more relevant documents." % overall
        )

    if rerank < 0.3 and overall < 0.0:
        return False, (
            "⚠️  **Documents not relevant enough**\n"
            "The reranker flagged the retrieved documents as not relevant to the question.\n"
            "💡 Tip: Use more specific keywords."
        )

    if overall < CONFIDENCE_THRESHOLD:
        return True, (
            "⚠️  **Disclaimer**: Low confidence answer (%.2f). Please verify with official sources.\n\n"
            % overall
        )

    return True, ""


# =========================
# MAIN
# =========================
def main():
    print("\n" + "=" * 60)
    print("🤖 LEGAL RAG — INDONESIAN LAW")
    print("=" * 60)

    if not os.path.exists(VECTOR_DB):
        print("❌ Vector DB not found:", VECTOR_DB)
        sys.exit(1)

    rag = LegalRAGSystem()
    chain = prompt | rag.llm

    print("\n✅ Ready! Type your legal question below.")
    print("   'q' to quit | 'clear' to clear screen\n")

    while True:
        try:
            question = input("Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break

        if not question:
            continue
        if question.lower() == "q":
            break
        if question.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        print("\n" + "-" * 60)
        start_time = time.time()

        # Hybrid Search
        try:
            docs = rag.hybrid_search(question)
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue

        if not docs:
            print("❌ No relevant sources found.")
            continue

        # Rerank if enabled
        if rag.cross_encoder:
            docs = rag.rerank(question, docs)

        # Confidence
        confidence = rag.calculate_confidence(docs)
        print(f"\n📊 Confidence:")
        print(f"   Overall:   {confidence['overall']:.3f}")
        print(f"   Retrieval: {confidence['retrieval']:.3f}")
        print(f"   Rerank:    {confidence['rerank']:.3f}")
        print(f"   Quality:   {confidence['quality']:.3f}")

        # Fallback check
        should_answer, fallback_msg = apply_fallback(confidence, docs)

        if not should_answer:
            print(f"\n{fallback_msg}")
            print(f"\n📚 Documents found:")
            for i, d in enumerate(docs[:3], 1):
                print(f"   {i}. {d.source}")
            print("-" * 60)
            continue

        # Build Context
        context = build_context(docs)
        if not context.strip():
            print("❌ Context is empty after cleaning.")
            continue

        # Retrieval preview
        print(f"\n🔍 Final ({len(docs)} documents):")
        for i, d in enumerate(docs, 1):
            preview = clean_text(d.content)[:70] + "..."
            print(f"   {i}. [{d.source}] score={d.rerank_score:.2f}")
            print(f"      {preview}")

        print("-" * 60)

        # Generate Answer
        print("\n📌 ANSWER:")
        if fallback_msg:
            print(fallback_msg)

        try:
            result = invoke_with_retry(chain, {
                "context": context,
                "question": question,
            })
            print(result)
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            continue

        elapsed = time.time() - start_time
        print(f"\n📚 SOURCES:")
        seen = set()
        for d in docs:
            if d.source not in seen:
                print(f"   • {d.source}")
                seen.add(d.source)

        print(f"\n⏱️  {elapsed:.2f}s | 📊 {confidence['overall']:.3f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
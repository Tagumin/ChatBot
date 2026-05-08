import re

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# =====================================
# CONFIG
# =====================================
TXT_FILE = "kuhp.txt"

VECTOR_DB = "./vectorstore"

COLLECTION_NAME = "uu_indonesia"

EMBED_MODEL = "nomic-embed-text"

# =====================================
# LOAD TXT
# =====================================
try:
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

except UnicodeDecodeError:

    print("⚠ UTF-8 gagal, fallback ke cp1252...")

    with open(TXT_FILE, "r", encoding="cp1252") as f:
        text = f.read()

print("✅ TXT loaded")

# =====================================
# SPLIT LINES
# =====================================
lines = text.splitlines()

documents = []

current_book = ""
current_chapter = ""
current_article = ""

article_content = []

# =====================================
# PARSER
# =====================================
for line in lines:

    line = line.strip()

    if not line:
        continue

    # =========================
    # DETECT BOOK
    # =========================
    if re.match(r"Book\s+\w+", line, re.IGNORECASE):

        current_book = line
        continue

    # =========================
    # DETECT CHAPTER
    # =========================
    if re.match(r"Chapter\s+[IVXLC]+", line, re.IGNORECASE):

        current_chapter = line
        continue

    # =========================
    # DETECT ARTICLE
    # =========================
    if re.match(r"article\s+\d+", line, re.IGNORECASE):

        # simpan article sebelumnya
        if current_article and article_content:

            full_text = f"""
{current_book}

{current_chapter}

{current_article}

{" ".join(article_content)}
"""

            doc = Document(
                page_content=full_text,
                metadata={
                    "book": current_book,
                    "chapter": current_chapter,
                    "article": current_article,
                    "source_file": TXT_FILE
                }
            )

            documents.append(doc)

        # reset content
        current_article = line
        article_content = []

        continue

    # =========================
    # ARTICLE CONTENT
    # =========================
    article_content.append(line)

# =====================================
# SAVE LAST ARTICLE
# =====================================
if current_article and article_content:

    full_text = f"""
{current_book}

{current_chapter}

{current_article}

{" ".join(article_content)}
"""

    doc = Document(
        page_content=full_text,
        metadata={
            "book": current_book,
            "chapter": current_chapter,
            "article": current_article,
            "source_file": TXT_FILE
        }
    )

    documents.append(doc)

print(f"✅ Total documents: {len(documents)}")

# =====================================
# EMBEDDINGS
# =====================================
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL
)

print("✅ Embedding model loaded")

# =====================================
# STORE VECTOR DB
# =====================================
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=VECTOR_DB,
    collection_name=COLLECTION_NAME
)
with open("preview_chunks.txt", "w", encoding="utf-8") as f:

    for i, doc in enumerate(documents):

        f.write(f"\n========== DOCUMENT {i+1} ==========\n")
        f.write(doc.page_content)
        f.write("\n\n")

print("✅ Vector DB berhasil dibuat")
print(f"📦 Lokasi DB: {VECTOR_DB}")

# =====================================
# PREVIEW
# =====================================
print("\n📄 SAMPLE DOCUMENT:\n")
print(documents[0].page_content)
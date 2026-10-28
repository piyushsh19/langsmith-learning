
import os
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_PATH = "islr.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# Models already downloaded on your Mac
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4


# --------------------------------------------------
# Local embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "mps",
        "local_files_only": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)


# --------------------------------------------------
# Local LLM
# --------------------------------------------------

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=LLM_MODEL,
    task="text-generation",
    model_kwargs={
        "local_files_only": True,
    },
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

llm = ChatHuggingFace(llm=llm_pipeline)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful PDF question-answering assistant. "
        "Answer only using the provided context. "
        "If the answer is not present in the context, "
        "say you do not know. Be concise and factual."
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Context:\n{context}"
    ),
])


# --------------------------------------------------
# Traced helper functions
# --------------------------------------------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()


@traceable(name="split_documents")
def split_documents(
    docs,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    return FAISS.from_documents(splits, embeddings)


def file_fingerprint(path: str) -> dict:
    p = Path(path)
    hasher = hashlib.sha256()

    with p.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            hasher.update(chunk)

    return {
        "sha256": hasher.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }


def index_key(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    metadata = {
        "pdf_fingerprint": file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
        "format": "local-v1",
    }

    serialized = json.dumps(
        metadata, sort_keys=True
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


# --------------------------------------------------
# Load or build FAISS index
# --------------------------------------------------

@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path):
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@traceable(name="build_index", tags=["index"])
def build_index_run(
    pdf_path: str,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
):
    docs = load_pdf(pdf_path)

    splits = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not splits:
        raise ValueError(
            "No text was extracted from the PDF."
        )

    vectorstore = build_vectorstore(splits)

    index_dir.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_dir))

    metadata = {
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
        "chunks": len(splits),
    }

    (index_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Indexed {len(splits)} chunks.")

    return vectorstore


def load_or_build_index(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_rebuild: bool = False,
):
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    key = index_key(
        pdf_path,
        chunk_size,
        chunk_overlap,
    )

    index_dir = INDEX_ROOT / key

    index_exists = (
        index_dir.is_dir()
        and (index_dir / "index.faiss").is_file()
        and (index_dir / "index.pkl").is_file()
    )

    if index_exists and not force_rebuild:
        print("Loading cached FAISS index...")
        return load_index_run(index_dir)

    print("Building FAISS index locally...")

    return build_index_run(
        pdf_path,
        index_dir,
        chunk_size,
        chunk_overlap,
    )


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_rebuild: bool = False,
):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild,
    )


@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        force_rebuild,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa", "local-llm"],
            "metadata": {
                "k": RETRIEVAL_K,
                "llm": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
            },
        },
    )


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    print("Local PDF RAG is ready.")
    print(f"LLM: {LLM_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL}")

    try:
        while True:
            question = input(
                "\nQuestion (or 'exit' to quit): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            answer = setup_pipeline_and_query(
                PDF_PATH,
                question,
            )

            print("\nAnswer:\n")
            print(answer)

    except KeyboardInterrupt:
        print("\nExiting PDF RAG.")

# -*- coding: utf-8 -*-
"""
RAG EF – FAISS + Ollama (DeepSeek por defecto)
"""
import os, sys, argparse, time, textwrap
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    HAS_DOCX = True
except Exception:
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import PyPDFLoader, TextLoader
    HAS_DOCX = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

INDEX_DIR = "faiss_index_EF"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = os.getenv("EF_AI_MODEL", "deepseek-r1:14b")
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".doc"}

def parse_meta(pairs: Optional[List[str]]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for p in pairs or []:
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta

def _load_single(path: Path) -> List:
    s = path.suffix.lower()
    if s == ".pdf":
        return PyPDFLoader(str(path)).load()
    if s in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if s in {".doc", ".docx"} and HAS_DOCX:
        return Docx2txtLoader(str(path)).load()
    print(f"Aviso: formato no soportado {path.suffix} → {path.name}")
    return []

def load_docs(inputs: List[str], default_meta: Dict) -> List:
    docs = []
    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            print(f"Aviso: no existe {p}")
            continue
        files: List[Path] = []
        if p.is_dir():
            files = [q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED]
        else:
            files = [p]
        for f in files:
            for d in _load_single(f):
                d.metadata.update(default_meta or {})
                d.metadata.setdefault("source", str(f))
            docs.extend(_load_single(f))
    return docs

def build_index(input_paths: List[str], chunk_size: int = 500, overlap: int = 100, default_meta: Optional[Dict] = None):
    docs = load_docs(input_paths, default_meta or {})
    if not docs:
        print("No se cargaron documentos.")
        sys.exit(1)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print(f"Índice FAISS creado en '{INDEX_DIR}' con {len(chunks)} chunks.")

def add_to_index(input_paths: List[str], chunk_size: int = 500, overlap: int = 100, default_meta: Optional[Dict] = None):
    docs = load_docs(input_paths, default_meta or {})
    if not docs:
        print("No se cargaron documentos para añadir.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    if Path(INDEX_DIR).exists():
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print(f"Añadidos {len(chunks)} chunks al índice '{INDEX_DIR}'.")

def load_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def make_qa(db, model_name: str = DEFAULT_MODEL, temperature: float = 0.0):
    """Devuelve un objeto QA cuya .run() retorna:
    {"answer": str, "sources": [ {source, unidad, score, snippet} ] }.
    """
    from langchain_ollama import ChatOllama
    from langchain.prompts import ChatPromptTemplate
    llm = ChatOllama(model=model_name, temperature=temperature)

    system_prompt = (
    "Eres profesor de Educación Física. Responde basándote EXCLUSIVAMENTE en el CONTEXTO. "
    "Resume, organiza y conecta la información del contexto para contestar. "
    "Si el contexto no aporta nada útil o no es relevante para la pregunta, responde exactamente: 'No está en el contexto'. "
    "NO muestres razonamiento ni etiquetas 'think'. Devuelve solo la respuesta final, breve y clara, en español."
)
    template = ChatPromptTemplate.from_template(
        "<sistema>\n{system}\n</sistema>\n<CONTEXTO>\n{context}\n</CONTEXTO>\n<PREGUNTA>\n{question}\n</PREGUNTA>"
    )

    class QA:
        def __init__(self, llm, db):
            self.llm, self.db = llm, db

        def _retrieve(self, question: str, k: int, unidad: Optional[str]):
            results = self.db.similarity_search_with_score(question, k=24)
            def _filter(rs, unit):
                picked = []
                for doc, sc in rs:
                    if unit and doc.metadata.get("unidad") != unit:
                        continue
                    picked.append((doc, sc))
                    if len(picked) >= k:
                        break
                return picked
            picked = _filter(results, unidad)
            # Fallback: si con unidad no hay nada, reintenta SIN filtrar
            if not picked:
                picked = _filter(results, None)
            return picked

        def run(self, question: str, k: int = 3, unidad: Optional[str] = None, show: bool = False):
            pairs = self._retrieve(question, k=k, unidad=unidad)
            if not pairs:
                return {"answer": "No está en el contexto", "sources": []}

            docs = [d for (d, _) in pairs]
            context = "\n\n".join(d.page_content for d in docs)
            prompt = template.format(system=system_prompt, context=context, question=question)
            out = self.llm.invoke(prompt).content.strip()

            sources = []
            for (doc, score) in pairs:
                src = doc.metadata.get("source", "?")
                uni = doc.metadata.get("unidad", "?")
                txt = doc.page_content.replace("\n", " ")
                snippet = (txt[:240] + "…") if len(txt) > 240 else txt
                sources.append({"source": src, "unidad": uni, "score": float(score), "snippet": snippet})

            if show:
                print("\n=== CHUNKS RECUPERADOS ===")
                for i, s in enumerate(sources, 1):
                    print(f"[{i}] {s['source']} | unidad={s['unidad']} | score={s['score']:.4f}\n  {s['snippet']}\n")

            return {"answer": out, "sources": sources}

    return QA(llm, db)

def inspect_index():
    db = load_index()
    vals = db.docstore._dict.values()
    sources = sorted({(d.metadata.get("source","?"), d.metadata.get("unidad","?")) for d in vals})
    for s in sources: print(s)
    print("Total chunks:", len(vals))

def query(question: str, model_name: str = DEFAULT_MODEL, k: int = 3, unidad: Optional[str] = None, show: bool = False):
    if not Path(INDEX_DIR).exists():
        print("No existe el índice. Ejecuta primero: python ef_rag.py build --docs <rutas|carpetas>")
        sys.exit(1)
    db = load_index()
    qa = make_qa(db, model_name=model_name)
    t0 = time.time()
    ans = qa.run(question, k=k, unidad=unidad, show=show)
    print(ans)
    print(f"\n[Retrieval+Respuesta: {time.time()-t0:.2f}s]")

def main():
    p = argparse.ArgumentParser(description="RAG EF – FAISS + Ollama")
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build", help="Construye índice (acepta carpetas y archivos)")
    pb.add_argument("--docs", nargs="+", required=True)
    pb.add_argument("--chunk", type=int, default=500)
    pb.add_argument("--overlap", type=int, default=100)
    pb.add_argument("--meta", nargs="*", help='Metadatos (ej: unidad=Voleibol)')

    pa = sub.add_parser("add", help="Añadir documentos al índice")
    pa.add_argument("--docs", nargs="+", required=True)
    pa.add_argument("--chunk", type=int, default=500)
    pa.add_argument("--overlap", type=int, default=100)
    pa.add_argument("--meta", nargs="*", help='Metadatos (ej: unidad=General)')

    pq = sub.add_parser("query", help="Consulta")
    pq.add_argument("--q", required=True)
    pq.add_argument("--k", type=int, default=3)
    pq.add_argument("--model", default=DEFAULT_MODEL)
    pq.add_argument("--unidad")

    sub.add_parser("inspect", help="Fuentes y metadatos")
    args = p.parse_args()

    if args.cmd == "build":
        build_index(args.docs, chunk_size=args.chunk, overlap=args.overlap, default_meta=parse_meta(args.meta))
    elif args.cmd == "add":
        add_to_index(args.docs, chunk_size=args.chunk, overlap=args.overlap, default_meta=parse_meta(args.meta))
    elif args.cmd == "query":
        query(args.q, model_name=args.model, k=args.k, unidad=args.unidad)
    elif args.cmd == "inspect":
        inspect_index()

if __name__ == "__main__":
    main()
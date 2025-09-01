# corrective_rag.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

from duckduckgo_search import DDGS
import requests
import trafilatura

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

import os
import yaml
from pathlib import Path


@dataclass
class SourceItem:
    source: str
    unidad: str
    text: str
    kind: str   # "local" | "web"

# --- Model resolution helpers -------------------------------------------------

def _get_default_model() -> str:
    """Return preferred model name: ENV > config.yaml > deepseek-r1:14b."""
    # 1) environment variable
    m = os.getenv("EF_AI_MODEL")
    if m:
        return m
    # 2) config.yaml next to this file (project root)
    try:
        cfg_path = Path(__file__).resolve().parent / "config.yaml"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            name = (data.get("model") or {}).get("name")
            if name:
                return name
    except Exception:
        pass
    # 3) safe fallback
    return "deepseek-r1:14b"

def make_judge(model: Optional[str] = None, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model or _get_default_model(), temperature=temperature)

def relevance_score(judge: ChatOllama, query: str, text: str) -> Tuple[float,str]:
    """
    Devuelve (score 0..100, rationale). Robusto a salidas ruidosas.
    """
    tmpl = (
        "Evalúa la RELEVANCIA del fragmento para responder la consulta.\n"
        "Devuelve SOLO en una línea: score=<0-100>; rationale=<breve>\n\n"
        "Consulta: {q}\n---\nFragmento:\n{text}\n---"
    )
    out = judge.invoke(ChatPromptTemplate.from_template(tmpl).format(q=query, text=text[:1200])).content.strip()
    m = re.search(r"score\s*=\s*(\d{1,3})", out, flags=re.I)
    score = float(m.group(1)) if m else 0.0
    score = max(0.0, min(100.0, score))
    rationale = re.sub(r"^.*?rationale\s*=\s*", "", out, flags=re.I) if "rationale" in out.lower() else out
    return score, rationale[:280]

def grade_and_filter(judge: ChatOllama, query: str, items: List[SourceItem],
                     threshold: float = 55.0, keep: int = 4) -> Tuple[List[Tuple[SourceItem,float]], float]:
    scored = []
    best = 0.0
    for it in items:
        s, _ = relevance_score(judge, query, it.text)
        best = max(best, s)
        scored.append((it, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [(it,s) for it,s in scored if s >= threshold][:keep]
    return filtered, best

def web_search_and_scrape(query: str, n: int = 3, timeout: int = 12) -> List[SourceItem]:
    out: List[SourceItem] = []
    # 1) búsqueda
    with DDGS() as ddg:
        results = ddg.text(query, max_results=n, safesearch="moderate")
    for r in results or []:
        url = r.get("href") or r.get("link") or r.get("url")
        if not url:
            continue
        # 2) descarga + extracción
        text = None
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent":"ef-ai/1.0"})
            resp.raise_for_status()
            text = trafilatura.extract(resp.text) or ""
        except Exception:
            text = ""
        # 3) fallback a snippet/título
        snippet = r.get("body") or r.get("snippet") or r.get("description") or ""
        title   = r.get("title") or ""
        payload = (text or "").strip() or (f"{title}\n{snippet}".strip())
        if not payload:
            continue
        out.append(SourceItem(source=url, unidad="web", text=payload, kind="web"))
    return out

def aggregate_and_answer(llm: ChatOllama, query: str, picks: List[Tuple[SourceItem,float]]) -> Tuple[str,List[str]]:
    # construir contexto + citar fuentes
    ctx_parts, cites = [], []
    for it, s in picks:
        cites.append(f"{it.kind}:{it.source}")
        ctx_parts.append(f"[{it.kind}:{it.source}] {it.text[:1200]}")
    ctx = "\n\n".join(ctx_parts) or "NO_CONTEXT"
    tmpl = (
        "Responde a la consulta SOLO con el contexto dado. Si falta, di 'No está en el contexto'.\n"
        "Añade al final una sección 'Fuentes' listando las URLs/paths usadas.\n\n"
        "Consulta: {q}\n\nContexto:\n{ctx}\n\nRespuesta:"
    )
    ans = llm.invoke(ChatPromptTemplate.from_template(tmpl).format(q=query, ctx=ctx)).content.strip()
    return ans, cites

def corrective_rag_answer(
    query: str,
    local_items: List[SourceItem],
    model_gen: Optional[str] = None,
    model_judge: Optional[str] = None,
    threshold: float = 55.0,
    allow_web: bool = True,
    web_k: int = 3,
) -> Tuple[str, List[str], float]:
    """
    Orquesta: valida relevancia local -> (opcional) añade web -> responde.
    Devuelve: (respuesta, citas, best_local_score)
    """
    model_gen = model_gen or _get_default_model()
    judge = make_judge(model_judge or model_gen, temperature=0.0)
    # 1) validar local
    local_filtered, best_local = grade_and_filter(judge, query, local_items, threshold=threshold, keep=4)
    picks = local_filtered[:]
    # 2) si poco contexto útil, buscar web
    if allow_web and (not picks or best_local < threshold + 5):
        web_items = web_search_and_scrape(query, n=web_k)
        web_filtered, _ = grade_and_filter(judge, query, web_items, threshold=threshold, keep=3)
        picks.extend(web_filtered)
    # 3) responder
    llm = ChatOllama(model=model_gen, temperature=0.2)
    answer, cites = aggregate_and_answer(llm, query, picks)
    return answer, cites, best_local
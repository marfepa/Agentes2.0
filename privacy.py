# privacy.py
from __future__ import annotations
import re
from typing import Any

# Patrones básicos de PII (ES + genérico)
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", re.I)
RE_PHONE = re.compile(r"(?:(?:\+?\d{1,3})?[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}")
RE_DNI = re.compile(r"\b\d{8}[A-Z]\b")
RE_NIE = re.compile(r"\b[XYZ]\d{7}[A-Z]\b", re.I)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)

# Heurística de nombres propios (riesgo de falsos positivos controlado)
RE_NAME = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}){0,2}\b")

def _redact_line(s: str) -> str:
    s = RE_EMAIL.sub("[[EMAIL]]", s)
    s = RE_URL.sub("[[URL]]", s)
    s = RE_DNI.sub("[[ID]]", s)
    s = RE_NIE.sub("[[ID]]", s)
    s = RE_PHONE.sub("[[PHONE]]", s)
    # Nombres al final para no romper emails/URLs ya sustituidos
    s = RE_NAME.sub("[[NAME]]", s)
    return s

def redact_pii_text(s: str) -> str:
    if not s:
        return s
    # Normalización mínima
    s = s.replace("\u00A0", " ")
    return _redact_line(s)

def redact_pii_any(obj: Any) -> Any:
    """Redacta PII recursivamente en strings dentro de dict/list/tuplas."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return redact_pii_text(obj)
    if isinstance(obj, dict):
        return {k: redact_pii_any(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_pii_any(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_pii_any(x) for x in obj)
    return obj
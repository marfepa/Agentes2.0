# guards.py
from __future__ import annotations
import re
from typing import Tuple

# Fallback ligero por si NeMo no está disponible
BLOCK_PATTERNS = [
    r"\binsulto\b", r"\bobsceno\b", r"\bviolencia\b", r"\bdroga(s)?\b",
    r"\bsexo\b", r"\border\b"  # puedes ajustar
]
BLOCK_RE = re.compile("|".join(BLOCK_PATTERNS), re.I)

# Intentar usar NeMo Guardrails si está instalado y configurado
rails = None
try:
    from nemoguardrails import LLMRails, RailsConfig
    # Lee configuración desde ./config/rails (si existe)
    rails_cfg = RailsConfig.from_path("config/rails")
    rails = LLMRails(rails_cfg)
except Exception:
    rails = None

def moderate_text(text: str) -> Tuple[bool, str, str]:
    """
    Devuelve (allowed, output_text, reason).
    Con NeMo: deja que el rail intervenga si corresponde.
    Fallback: bloquea si encuentra patrones sensibles.
    """
    if rails:
        try:
            # generate() aplicará los rails definidos en config/rails
            out = rails.generate(messages=[{"role": "user", "content": text}])
            # Si hay intervención, NeMo ya devolvería respuesta segura
            return True, out["content"] if isinstance(out, dict) and "content" in out else str(out), "nemo"
        except Exception as e:
            # Si NeMo falla, cae a fallback
            pass

    if BLOCK_RE.search(text or ""):
        return False, "Contenido bloqueado por moderación (política educativa).", "fallback"
    return True, text, "fallback"
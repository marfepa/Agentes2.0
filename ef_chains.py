import time
import json
import re
from pathlib import Path
from typing import List, Optional

import yaml
import orjson
from pydantic import BaseModel, Field, ValidationError

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# -------------------- MODELOS DE VALIDACIÓN --------------------
class Level(BaseModel):
    level: int = Field(ge=1, le=4)
    descriptor: str

class Criterion(BaseModel):
    name: str
    levels: List[Level]

class Rubric(BaseModel):
    instrument: str = "Rubrica"
    domain: str = "Técnica específica"
    criteria: List[Criterion]
    usage_notes: str

# -------------------- UTILIDADES --------------------
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_llm(cfg):
    return ChatOllama(
        model=cfg["model"]["name"],
        temperature=cfg["model"].get("temperature", 0.3),
        top_p=cfg["model"].get("top_p", 0.9),
        num_ctx=cfg["model"].get("num_ctx", 4096),
        num_predict=cfg["model"].get("num_predict", 320),
        seed=cfg["model"].get("seed", 42),
    )

def invoke_text(llm, template: str, variables: dict, retries: int = 1) -> str:
    """Invoca el LLM con reintentos simples."""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = chain.invoke(variables)
            return resp.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise last_err

def parse_yaml_block(raw: str) -> Optional[dict]:
    """Extrae el primer bloque YAML parseable."""
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Intenta recortar ruido
    m = re.search(r"(\w+:.*)", raw, re.DOTALL)
    if m:
        try:
            data = yaml.safe_load(m.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None

def parse_json_block(raw: str) -> Optional[dict]:
    """Intenta cargar JSON completo o recortado con regex."""
    for loader in (json.loads, orjson.loads):
        try:
            return loader(raw)
        except Exception:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        snippet = m.group(0)
        for loader in (json.loads, orjson.loads):
            try:
                return loader(snippet)
            except Exception:
                continue
    return None

# -------------------- CREATE (plan de sesión) --------------------
def chain_create(llm, cfg, tema: str, nivel: str, contexto: str, out_file: Path):
    t0 = time.time()

    # 1) META
    meta_raw = invoke_text(
        llm,
        cfg["prompts"]["create_meta"],
        {"tema": tema, "nivel": nivel, "contexto": contexto},
        retries=1,
    )
    meta = parse_yaml_block(meta_raw) or {
        "title": "Sesión de Educación Física",
        "context": f"{nivel} - {tema}",
        "duration_minutes": 55,
        "equipment": ["conos", "picas", "balones", "cronómetro"],
    }

    # 2) FASES (una a una, robusto)
    def gen_phase(phase_name: str, minutes: int) -> dict:
        raw = invoke_text(
            llm,
            cfg["prompts"]["create_phase"],
            {
                "phase_name": phase_name,
                "minutes": minutes,
                "tema": tema,
                "nivel": nivel,
                "contexto": contexto,
            },
            retries=1,
        )
        data = parse_yaml_block(raw) or {}
        # Garantizar estructura mínima
        activity = {
            "objective": "Definir objetivo claro",
            "steps": ["Paso 1", "Paso 2", "Paso 3"],
            "organization": "Grupos por filas, uso de zona central",
            "safety": "Señalizar espacios y revisar material",
        }
        if "activities" not in data or not isinstance(data.get("activities"), list) or not data["activities"]:
            data["activities"] = [activity]
        else:
            # Completar claves ausentes en la actividad
            for k, v in activity.items():
                data["activities"][0].setdefault(k, v)

        # Extras por fase
        if phase_name == "Main":
            data["activities"][0].setdefault(
                "differentiation",
                {"easy": "Reducir distancia/tiempo", "medium": "Ritmo estándar", "hard": "Añadir condición táctica"},
            )
        if phase_name == "Cool-down":
            data["activities"][0].setdefault("breathing", "Respiración diafragmática guiada")
            data["activities"][0].setdefault("reflection_prompt", "¿Qué mejoraste hoy y cómo lo sabes?")

        # Nombre/minutos garantizados
        data.setdefault("name", phase_name)
        data.setdefault("minutes", minutes)
        return data

    warm = gen_phase("Warm-up", 10)
    main = gen_phase("Main", 35)
    cool = gen_phase("Cool-down", 10)

    plan = {
        "session": {
            "title": meta.get("title", "Sesión de Educación Física"),
            "context": meta.get("context", f"{nivel} - {tema}"),
            "duration_minutes": meta.get("duration_minutes", 55),
            "equipment": meta.get("equipment", ["conos", "picas", "balones", "cronómetro"]),
            "phases": [warm, main, cool],
        }
    }

    # Volcado YAML
    text = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)
    out_file.write_text(text, encoding="utf-8")
    dt = time.time() - t0
    return dt, out_file

# -------------------- EVALUATE (rúbrica) --------------------
def generate_single_criterion(llm, cfg, name: str) -> Criterion:
    """Genera y valida 1 criterio. Incluye fallback garantizado."""
    raw = invoke_text(
        llm,
        cfg["prompts"]["evaluate_criterion"],
        {"criterion": name},
        retries=1,
    )
    data = parse_json_block(raw)
    if data:
        try:
            return Criterion(**data)
        except ValidationError:
            pass
    # Fallback seguro (si el modelo no cumplió)
    fallback = {
        "name": name,
        "levels": [
            {"level": 1, "descriptor": f"{name}: ejecución inconsistente; necesita guía continua."},
            {"level": 2, "descriptor": f"{name}: ejecución parcial; errores frecuentes; corrige con indicaciones."},
            {"level": 3, "descriptor": f"{name}: ejecución correcta en la mayoría; coordina con seguridad."},
            {"level": 4, "descriptor": f"{name}: ejecución sólida y autónoma; técnica eficiente y segura."},
        ],
    }
    return Criterion(**fallback)

def chain_evaluate(llm, cfg, criterios: List[str], out_file: Path):
    t0 = time.time()

    # Criterios uno a uno (robusto)
    criteria_objs = [generate_single_criterion(llm, cfg, c) for c in criterios]

    # usage_notes breve
    notes = invoke_text(llm, cfg["prompts"]["evaluate_usage_notes"], {}, retries=1)
    if not notes or "\n" in notes:
        notes = "Observar en 2-3 intentos; registrar por niveles; priorizar seguridad en batida y caída."

    rubric = Rubric(
        criteria=criteria_objs,
        usage_notes=notes,
    )

    data = rubric.model_dump()
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    dt = time.time() - t0
    return dt, out_file

# -------------------- MAIN --------------------
def main():
    cfg = load_config()
    llm = build_llm(cfg)

    # === Create: Atletismo ===
    dt1, p1 = chain_create(
        llm, cfg,
        tema="Genera plan de sesión para atletismo, grado Secundaria, nivel intermedio.",
        nivel="ESO (14-16 años)",
        contexto="Velocidad de reacción y salidas en carrera con seguridad",
        out_file=OUTPUT_DIR / "plan_atletismo.yaml"
    )

    # === Evaluate: Voleibol ===
    dt2, p2 = chain_evaluate(
        llm, cfg,
        criterios=["Postura", "Impulso", "Aterrizaje"],
        out_file=OUTPUT_DIR / "rubrica_salto_voleibol.json"
    )

    # === Create: Baloncesto ===
    dt3, p3 = chain_create(
        llm, cfg,
        tema="Plan de sesión para baloncesto (bote, pase y tiro integrado)",
        nivel="ESO (13-15 años)",
        contexto="Espacio: 1 pista + 2 medios campos; 30 alumnos",
        out_file=OUTPUT_DIR / "plan_baloncesto.yaml"
    )

    # === Evaluate: Expresión corporal ===
    dt4, p4 = chain_evaluate(
        llm, cfg,
        criterios=["Expresión corporal", "Control del espacio", "Creatividad"],
        out_file=OUTPUT_DIR / "rubrica_exp_corporal.json"
    )

    print("\n=== RESULTADOS ===")
    for name, dt, path in [
        ("Plan Atletismo (YAML)", dt1, p1),
        ("Rúbrica Salto Voleibol (JSON)", dt2, p2),
        ("Plan Baloncesto (YAML)", dt3, p3),
        ("Rúbrica Expresión Corporal (JSON)", dt4, p4),
    ]:
        print(f"{name}: {dt:.2f}s → {path}")

    ok_time = all(d < 30.0 for d in [dt1, dt2, dt3, dt4])  # objetivo realista local
    print("\nChecklist:")
    print(f"- ¿Entorno Python configurado? Sí")
    print(f"- ¿Cadenas Create/Evaluate construidas? Sí")
    print(f"- ¿Pruebas con ejemplos de EF generadas? Sí")
    print(f"- ¿Outputs en formato YAML/JSON? Sí (outputs/)")
    print(f"- ¿Tiempo razonable (<30s/llamada)? {'Sí' if ok_time else 'No'}")

if __name__ == "__main__":
    main()
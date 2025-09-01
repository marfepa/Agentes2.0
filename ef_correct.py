import json
import yaml
from pathlib import Path
from typing import List, Dict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_RUBRIC_YAML = """\
criterios:
  - nombre: "Control del balón"
    niveles:
      - puntuacion: 1
        desc: "No controla el balón; frecuentes pérdidas."
      - puntuacion: 2
        desc: "Control básico; errores al aumentar velocidad."
      - puntuacion: 3
        desc: "Buen control general; algunos fallos bajo presión."
      - puntuacion: 4
        desc: "Excelente control incluso en situaciones dinámicas."
  - nombre: "Técnica de salto"
    niveles:
      - puntuacion: 1
        desc: "Postura deficiente; falta coordinación."
      - puntuacion: 2
        desc: "Postura aceptable; impulso débil; caída inestable."
      - puntuacion: 3
        desc: "Buena técnica; pequeños fallos; caída mayormente segura."
      - puntuacion: 4
        desc: "Técnica sólida; impulso eficaz; aterrizaje seguro con flexión."
"""

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_llm(cfg):
    model_cfg = cfg.get("model", {})
    return ChatOllama(
        model=model_cfg.get("name", "mistral:7b-instruct"),
        temperature=model_cfg.get("temperature", 0.2),
        num_ctx=model_cfg.get("num_ctx", 2048),
        num_predict=model_cfg.get("num_predict", 300),
        seed=model_cfg.get("seed", 42),
        top_p=model_cfg.get("top_p", 0.9),
    )

def correct_task(llm, cfg, rubric_text: str, task_text: str) -> Dict:
    prompt_tmpl = ChatPromptTemplate.from_template(cfg["prompts"]["correct_one"])
    chain = prompt_tmpl | llm
    resp = chain.invoke({"rubric_text": rubric_text, "task_text": task_text})
    raw = resp.content.strip()

    try:
        data = json.loads(raw)
        score = int(data.get("score"))
        reasons = str(data.get("reasons", "")).strip()
        if score not in (1, 2, 3, 4) or not reasons:
            raise ValueError("estructura inválida")
        return {"score": score, "reasons": reasons[:200]}
    except Exception:
        return {"score": None, "reasons": raw[:200]}

def feedback_student(llm, cfg, corrections: Dict) -> str:
    prompt_tmpl = ChatPromptTemplate.from_template(cfg["prompts"]["feedback_student"])
    chain = prompt_tmpl | llm
    resp = chain.invoke({"corrections": json.dumps(corrections, ensure_ascii=False)})
    return resp.content.strip()

def feedback_class(llm, cfg, all_corrections: List[Dict]) -> str:
    prompt_tmpl = ChatPromptTemplate.from_template(cfg["prompts"]["feedback_class"])
    chain = prompt_tmpl | llm
    resp = chain.invoke({"corrections": json.dumps(all_corrections, ensure_ascii=False)})
    return resp.content.strip()

def run_gold_sample():
    cfg = load_config()
    llm = build_llm(cfg)

    rubric_file = Path("rubric_example.yaml")
    if rubric_file.exists():
        rubric_text = rubric_file.read_text(encoding="utf-8")
    else:
        rubric_text = DEFAULT_RUBRIC_YAML

    task_text = "El estudiante bota con mano dominante mirando al suelo y pierde el control al aumentar la velocidad en conducción."

    grade_result = correct_task(llm, cfg, rubric_text, task_text)
    print("Corrección individual:", grade_result)

    fb_student = feedback_student(llm, cfg, grade_result)
    print("\nFeedback al estudiante:\n", fb_student)

    all_results = [
        grade_result,
        {"score": 3, "reasons": "Buen control en estático; pierde precisión en desplazamientos rápidos."},
        {"score": 4, "reasons": "Control excelente y visión de juego; mantiene bote bajo presión."}
    ]
    fb_class = feedback_class(llm, cfg, all_results)
    print("\nFeedback de la clase:\n", fb_class)

    # Guardar salidas
    Path("outputs/correction_one.json").write_text(
        json.dumps(grade_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("outputs/feedback_student.txt").write_text(fb_student, encoding="utf-8")
    Path("outputs/feedback_class.txt").write_text(fb_class, encoding="utf-8")

if __name__ == "__main__":
    run_gold_sample()
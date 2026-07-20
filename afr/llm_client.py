"""
Local GGUF inference client via llama-cpp-python. Loads the model once per
process and provides two entry points: chat() for answer generation and
chat_json() for the Security Layer.

chat_json() applies a fail-secure default (Saltzer & Schroeder, 1975): if the
model does not return parseable JSON, the verdict is UNSAFE, since a parse
error is a system failure rather than a content decision.

Author: J. Fermin, Master's thesis, LMU München, 2026.

Inference is CPU-only (n_gpu_layers=0); the context window is set to 4096
tokens.
"""

from llama_cpp import Llama
import json
import os

MODEL_PATH = os.environ.get(
    "AFR_MODEL_PATH",
    "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)

_llm = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        model_label = os.path.basename(MODEL_PATH)  # zeigt den echten Dateinamen
        print(f"Lade Modell: {model_label} (Pfad: {MODEL_PATH})")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=0,     # explizit CPU-only (kein NVIDIA-Treiber auf Server verfuegbar)
            verbose=False
        )
        print(f"Modell bereit: {model_label}")
    return _llm

def chat(system: str, user: str, temperature: float = 0.3,
         max_tokens: int = 1024) -> str:
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"]

def chat_json(system: str, user: str) -> dict:
    """
    Speziell fuer den Security Layer:
    Gibt garantiert ein dict zurueck, auch bei Parse-Fehlern.
    """
    raw = chat(system, user, temperature=0.0, max_tokens=256)
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        # Fallback: UNSAFE
        # Fail-safe default (Saltzer & Schroeder, 1975): ein Parse-Fehler ist ein
        # Systemversagen, keine inhaltliche Entscheidung. UNSAFE verhindert, dass
        # technische Fehler unbemerkt zu Leaks fuehren.
        return {"verdict": "UNSAFE", "reason": f"Parse error: {raw[:100]}",
                "violated_rule": None}

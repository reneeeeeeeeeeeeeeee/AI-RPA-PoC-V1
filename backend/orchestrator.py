# Orchestrator - Gemma-4-31B (oder kleineres Modell) als Planungs-LLM
# Liest Modell aus dem Projektordner llm

import json
import uuid
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
LLM_PATH = ROOT_DIR / "llm"
JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


class Orchestrator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pending_context: Optional[dict] = None
        self._load_model()

    def _load_model(self):
        # Gemma wird NICHT beim Start geladen - nur Tokenizer
        # Modell wird bei Bedarf kurz geladen, dann wieder entladen
        # so teilen sich CogAgent und Gemma die 34 GB VRAM zeitversetzt
        try:
            from transformers import AutoTokenizer
            print(f"[Orchestrator] Lade Tokenizer aus {LLM_PATH} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(LLM_PATH))
            print("[Orchestrator] Tokenizer geladen. Modell wird bei Bedarf geladen.")
        except Exception as e:
            print(f"[Orchestrator] WARNUNG: Tokenizer konnte nicht geladen werden: {e}")
            print("[Orchestrator] Laeuft im Fallback-Modus (regelbasiert).")

    def llm_ready(self) -> bool:
        # Tokenizer geladen reicht - Modell wird on-demand geladen
        return self.tokenizer is not None

    def _load_model_gpu(self):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        print("[Orchestrator] Lade Gemma auf GPU (4-bit, ~17 GB VRAM)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(LLM_PATH),
            device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        print("[Orchestrator] Gemma geladen.")

    def _unload_model(self):
        import torch, gc
        if self.model is not None:
            print("[Orchestrator] Entlade Gemma von GPU...")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[Orchestrator] VRAM fuer CogAgent freigegeben.")

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if self.tokenizer is None:
            return self._fallback(prompt)

        import torch
        loaded_here = False
        try:
            if self.model is None:
                self._load_model_gpu()
                loaded_here = True

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
            decoded = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return decoded.strip()
        except Exception as e:
            print(f"[Orchestrator] Fehler beim Generieren: {e}")
            return self._fallback(prompt)
        finally:
            # Immer entladen wenn wir es selbst geladen haben
            if loaded_here:
                self._unload_model()

    def chat(self, user_text: str) -> str:
        prompt = f"""Du bist ein ERP-Assistent. Antworte kurz und hilfreich auf Deutsch.

Nutzer: {user_text}
Assistent:"""
        return self._generate(prompt, max_new_tokens=256)

    def plan(self, user_prompt: str, analysis: dict) -> dict:
        cols = list(analysis["column_map"].keys())
        sample = analysis["rows"][:2] if analysis["rows"] else []

        prompt = f"""Du bist ein ERP-Automatisierungs-Assistent für ein Web-ERP, das im Browser läuft.
Aufgabe des Nutzers: {user_prompt}

Excel-Spalten: {cols}
Beispieldaten: {json.dumps(sample, ensure_ascii=False)}

Erstelle ein JSON-Steps-Template für die ERP-Eingabe. Gib NUR valides JSON zurück, kein Text davor oder danach.
Format:
{{
  "erp_module": "string (z.B. commande/fournisseur/index.php)",
  "steps": [
    {{"action": "navigate", "target": "URL-Pfad"}},
    {{"action": "click",    "selector": "CSS oder Text"}},
    {{"action": "fill",     "selector": "CSS oder Label", "from_col": "Excel-Spaltenname"}},
    {{"action": "select",   "selector": "CSS oder Label", "from_col": "Excel-Spaltenname"}},
    {{"action": "click",    "selector": "Speichern"}}
  ]
}}"""

        raw = self._generate(prompt, max_new_tokens=512)

        # JSON aus Antwort extrahieren
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            # Fallback: Standard-ERP-Schritte
            return self._default_erp_steps(analysis)

    def _default_erp_steps(self, analysis: dict) -> dict:
        return {
            "erp_module": "commande/fournisseur/card.php?action=create",
            "steps": [
                {"action": "navigate", "target": "commande/fournisseur/card.php?action=create"},
                {"action": "fill",     "selector": "select[name='socid']",      "from_col": "Lieferant"},
                {"action": "fill",     "selector": "input[name='qty']",          "from_col": "Menge"},
                {"action": "fill",     "selector": "input[name='ref_supplier']", "from_col": "Artikel"},
                {"action": "click",    "selector": "Speichern"},
            ]
        }

    def save_job(self, plan: dict, analysis: dict) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "steps_template": plan.get("steps", []),
            "erp_module": plan.get("erp_module", ""),
            "rows": analysis["rows"],
            "file_path": analysis.get("file_path", ""),
            "column_map": analysis.get("column_map", {}),
        }
        with open(JOBS_DIR / f"{job_id}.json", "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        path = JOBS_DIR / f"{job_id}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _fallback(self, prompt: str) -> str:
        return "Verstanden. Wie kann ich helfen?"

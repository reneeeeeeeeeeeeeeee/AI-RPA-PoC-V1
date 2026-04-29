# Orchestrator - Gemma-4-31B (or smaller model) as the planning LLM
# Loads the model from the project's llm folder

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
        """Loads only the tokenizer at startup; the model is loaded on demand.

        Gemma and CogAgent both require substantial VRAM. Loading Gemma only when
        needed and unloading it immediately after lets both models share the same
        GPU in a time-sliced fashion without exceeding available VRAM.
        """
        try:
            from transformers import AutoTokenizer
            print(f"[Orchestrator] Loading tokenizer from {LLM_PATH} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(LLM_PATH))
            print("[Orchestrator] Tokenizer loaded. Model will be loaded on demand.")
        except Exception as e:
            print(f"[Orchestrator] WARNING: Tokenizer could not be loaded: {e}")
            print("[Orchestrator] Running in fallback mode (rule-based).")

    def llm_ready(self) -> bool:
        # Tokenizer loaded is sufficient - model is loaded on demand
        return self.tokenizer is not None

    def _load_model_gpu(self):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        print("[Orchestrator] Loading Gemma on GPU (4-bit, ~17 GB VRAM)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(LLM_PATH),
            device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        print("[Orchestrator] Gemma loaded.")

    def _unload_model(self):
        import torch, gc
        if self.model is not None:
            print("[Orchestrator] Unloading Gemma from GPU...")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[Orchestrator] VRAM freed for CogAgent.")

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
            print(f"[Orchestrator] Error during generation: {e}")
            return self._fallback(prompt)
        finally:
            # Always unload if we loaded it here
            if loaded_here:
                self._unload_model()

    def chat(self, user_text: str) -> str:
        prompt = f"""You are an ERP assistant. Reply briefly and helpfully.

User: {user_text}
Assistant:"""
        return self._generate(prompt, max_new_tokens=256)

    def plan(self, user_prompt: str, analysis: dict) -> dict:
        cols = list(analysis["column_map"].keys())
        sample = analysis["rows"][:2] if analysis["rows"] else []

        prompt = f"""You are an ERP automation assistant for a web-based ERP running in the browser.
User task: {user_prompt}

Excel columns: {cols}
Sample data: {json.dumps(sample, ensure_ascii=False)}

Create a JSON steps template for the ERP input. Return ONLY valid JSON, no text before or after.
Format:
{{
  "erp_module": "string (e.g. commande/fournisseur/index.php)",
  "steps": [
    {{"action": "navigate", "target": "URL path"}},
    {{"action": "click",    "selector": "CSS or text"}},
    {{"action": "fill",     "selector": "CSS or label", "from_col": "Excel column name"}},
    {{"action": "select",   "selector": "CSS or label", "from_col": "Excel column name"}},
    {{"action": "click",    "selector": "Save"}}
  ]
}}"""

        raw = self._generate(prompt, max_new_tokens=512)

        # Extract JSON from response
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            # Fallback: default ERP steps
            return self._default_erp_steps(analysis)

    def _default_erp_steps(self, analysis: dict) -> dict:
        return {
            "erp_module": "commande/fournisseur/card.php?action=create",
            "steps": [
                {"action": "navigate", "target": "commande/fournisseur/card.php?action=create"},
                {"action": "fill",     "selector": "select[name='socid']",      "from_col": "Lieferant"},
                {"action": "fill",     "selector": "input[name='qty']",          "from_col": "Menge"},
                {"action": "fill",     "selector": "input[name='ref_supplier']", "from_col": "Artikel"},
                {"action": "click",    "selector": "Save"},
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
        return "Understood. How can I help?"

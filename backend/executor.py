# ERP Executor - CogAgent-9B-20241220
# Fix: patcht transformers/modeling_utils.py direkt auf Disk
# um 'list' object has no attribute 'keys' zu beheben

import base64
import io
import re
import shutil
import subprocess
import time
from pathlib import Path

import pyautogui
import pygetwindow as gw
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
COGAGENT_PATH = ROOT_DIR / "ui"
LAST_SCREENSHOT_PATH = ROOT_DIR / "last_screenshot.png"

# Tesseract OCR Pfad setzen
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
except ImportError:
    pass
ERP_URL  = "http://localhost"

# Pfad zu transformers modeling_utils.py
import transformers
TRANSFORMERS_UTILS = Path(transformers.__file__).parent / "modeling_utils.py"


class ERPExecutor:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.aborted   = False
        self._load_cogagent()

    # ------------------------------------------------------------------
    # Patch 1: ChatGLMConfig.max_length
    # ------------------------------------------------------------------
    def _patch_config(self):
        f = COGAGENT_PATH / "configuration_chatglm.py"
        if not f.exists():
            return
        text = f.read_text(encoding="utf-8")
        if "PATCH_MAX_LENGTH" in text:
            return
        patch = (
            "\n"
            "    # PATCH_MAX_LENGTH\n"
            "    @property\n"
            "    def max_length(self):\n"
            "        return getattr(self, 'seq_length', 8192)\n"
            "\n"
            "    @max_length.setter\n"
            "    def max_length(self, v):\n"
            "        self.seq_length = v\n"
        )
        marker = "class ChatGLMConfig(PretrainedConfig):"
        if marker not in text:
            return
        idx = text.index(marker) + len(marker)
        nl  = text.index("\n", idx)
        f.write_text(text[:nl+1] + patch + text[nl+1:], encoding="utf-8")
        print("[CogAgent] Patch 1: ChatGLMConfig.max_length OK")

    # ------------------------------------------------------------------
    # Patch 2: modeling_chatglm.py - GenerationMixin + tied weights
    # ------------------------------------------------------------------
    def _patch_modeling(self):
        f = COGAGENT_PATH / "modeling_chatglm.py"
        if not f.exists():
            return
        text = f.read_text(encoding="utf-8")
        changed = False

        # GenerationMixin Import
        if "PATCH_GENMIXIN" not in text:
            if "from transformers.generation.utils import GenerationMixin" not in text and "from transformers import GenerationMixin" not in text:
                text = "from transformers.generation.utils import GenerationMixin  # PATCH_GENMIXIN\n" + text
            old_cls = "class ChatGLMForConditionalGeneration(PreTrainedModel):"
            new_cls = "class ChatGLMForConditionalGeneration(PreTrainedModel, GenerationMixin):  # PATCH_GENMIXIN"
            if old_cls in text:
                text = text.replace(old_cls, new_cls)
                changed = True
                print("[CogAgent] Patch 2a: GenerationMixin OK")

        # all_tied_weights_keys als Dict
        if "PATCH_TIED" not in text:
            marker = "class ChatGLMForConditionalGeneration"
            if marker in text:
                idx = text.index(marker)
                colon = text.index(":", idx)
                nl = text.index("\n", colon)
                patch = (
                    "\n"
                    "    _tied_weights_keys = []  # PATCH_TIED\n"
                    "    all_tied_weights_keys = {}  # PATCH_TIED_DICT\n"
                )
                text = text[:nl+1] + patch + text[nl+1:]
                changed = True
                print("[CogAgent] Patch 2b: all_tied_weights_keys OK")

        if changed:
            f.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Patch 3: transformers/modeling_utils.py direkt auf Disk patchen
    # Alle Stellen wo all_tied_weights_keys.keys() aufgerufen wird
    # werden ersetzt durch eine sichere Version die Listen unterstuetzt
    # ------------------------------------------------------------------
    def _patch_transformers_disk(self):
        if not TRANSFORMERS_UTILS.exists():
            print(f"[CogAgent] modeling_utils.py nicht gefunden: {TRANSFORMERS_UTILS}")
            return

        text = TRANSFORMERS_UTILS.read_text(encoding="utf-8")

        if "PATCH_TIED_LIST" in text:
            print("[CogAgent] Patch 3: transformers bereits gepatcht.")
            return

        replacements = [
            # mark_tied_weights_as_initialized
            (
                'for tied_param in getattr(self, "all_tied_weights_keys", {}).keys():',
                'for tied_param in (getattr(self, "all_tied_weights_keys", []) or []):  # PATCH_TIED_LIST'
            ),
            # get_total_byte_count
            (
                'tied_param_names = model.all_tied_weights_keys.keys()',
                'tied_param_names = list(model.all_tied_weights_keys) if isinstance(model.all_tied_weights_keys, list) else model.all_tied_weights_keys.keys()  # PATCH_TIED_LIST'
            ),
            # infer_auto_device_map (vorheriger Fehler)
            (
                'if len(model.all_tied_weights_keys) > 0:',
                'if len(model.all_tied_weights_keys or []) > 0:  # PATCH_TIED_LIST'
            ),
        ]

        changed = False
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                changed = True
                print(f"[CogAgent] Patch 3: ersetzt: {old[:60]}...")

        if changed:
            TRANSFORMERS_UTILS.write_text(text, encoding="utf-8")
            print("[CogAgent] Patch 3: transformers/modeling_utils.py gepatcht.")
        else:
            print("[CogAgent] Patch 3: Keine passenden Stellen gefunden - moeglicherweise andere transformers-Version.")
            # Zeige relevante Zeilen zur Diagnose
            for i, line in enumerate(text.splitlines()):
                if "all_tied_weights_keys" in line:
                    print(f"  Zeile {i+1}: {line.strip()}")

    # ------------------------------------------------------------------
    # Modell laden
    # ------------------------------------------------------------------
    def _load_cogagent(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[CogAgent] Lade Modell aus {COGAGENT_PATH} ...")

            self._patch_config()
            self._patch_modeling()
            self._patch_transformers_disk()

            # transformers neu importieren damit Patches wirksam sind
            import importlib
            import transformers.modeling_utils
            importlib.reload(transformers.modeling_utils)
            print("[CogAgent] transformers.modeling_utils neu geladen.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(COGAGENT_PATH),
                trust_remote_code=True,
            )

            if not torch.cuda.is_available():
                print("[CogAgent] CUDA nicht verfuegbar.")
                return

            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[CogAgent] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f} GB VRAM)")

            self.model = AutoModelForCausalLM.from_pretrained(
                str(COGAGENT_PATH),
                trust_remote_code=True,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()

            # GenerationMixin + generation_config zur Laufzeit injizieren
            from transformers.generation.utils import GenerationMixin
            from transformers import GenerationConfig
            import torch

            # 1. GenerationMixin in Klasse eintragen
            cls = type(self.model)
            if GenerationMixin not in cls.__mro__:
                cls.__bases__ = cls.__bases__ + (GenerationMixin,)
                print("[CogAgent] GenerationMixin injiziert.")

            # 2. Config bereinigen und fehlende Attribute ergaenzen
            if hasattr(self.model, "config"):
                cfg = self.model.config

                # max_length entfernen - stoert generate() in neuen transformers
                if hasattr(cfg, "__dict__") and "max_length" in cfg.__dict__:
                    del cfg.__dict__["max_length"]

                # Pflichtattribute die transformers/GenerationMixin erwartet
                cfg.is_encoder_decoder = False

                # num_hidden_layers - CogAgent nutzt num_layers
                if not hasattr(cfg, "num_hidden_layers"):
                    cfg.num_hidden_layers = getattr(cfg, "num_layers", 28)

                # num_attention_heads
                if not hasattr(cfg, "num_attention_heads"):
                    cfg.num_attention_heads = getattr(cfg, "num_attention_heads_per_partition",
                                              getattr(cfg, "multi_query_group_num", 2))

                # num_key_value_heads
                if not hasattr(cfg, "num_key_value_heads"):
                    cfg.num_key_value_heads = getattr(cfg, "multi_query_group_num",
                                             getattr(cfg, "num_attention_heads", 2))

                # hidden_size
                if not hasattr(cfg, "hidden_size"):
                    cfg.hidden_size = getattr(cfg, "hidden_size", 4096)

                print(f"[CogAgent] Config: num_hidden_layers={cfg.num_hidden_layers}, "
                      f"num_key_value_heads={cfg.num_key_value_heads}")

            # 3. generation_config direkt setzen (ueberschreibt config-Werte)
            eos = getattr(self.tokenizer, "eos_token_id", 2) or 2
            pad = getattr(self.tokenizer, "pad_token_id", None) or eos
            self.model.generation_config = GenerationConfig(
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=pad,
            )
            print("[CogAgent] generation_config gesetzt.")

            # 4. main_input_name setzen
            cls.main_input_name = "input_ids"

            # 5. Fehlende GenerationMixin-Methoden hinzufuegen
            if not hasattr(self.model, "_extract_past_from_model_output"):
                from transformers.generation.utils import GenerationMixin as GM
                if hasattr(GM, "_extract_past_from_model_output"):
                    cls._extract_past_from_model_output = GM._extract_past_from_model_output
                else:
                    def _extract_past(self_inner, outputs, *args, **kwargs):
                        return "past_key_values", getattr(outputs, "past_key_values", None)
                    cls._extract_past_from_model_output = _extract_past
                print("[CogAgent] _extract_past_from_model_output gesetzt.")

            print("[CogAgent] Modell erfolgreich geladen.")

        except Exception as e:
            print(f"[CogAgent] WARNUNG: Modell nicht geladen: {e}")
            import traceback
            traceback.print_exc()

    def cogagent_ready(self) -> bool:
        return self.model is not None

    def abort(self):
        self.aborted = True

    def _focus_ie(self):
        """Bringt das ERP-Fenster (IE, Edge oder Chrome mit ERP) in den Vordergrund."""
        import pygetwindow as gw, time

        # Alle Fenster-Keywords die das ERP enthalten koennten
        keywords = [
            "Internet Explorer", "ERP",
            "localhost", "127.0.0.1",
            "Edge",   # Microsoft Edge (auch mit Zero-Width-Space: Microsoft\u200bEdge)
            "edge",
        ]

        # Alle offenen Fenster anzeigen (nur beim ersten Aufruf)
        if not hasattr(self, "_windows_logged"):
            self._windows_logged = True
            all_titles = [w.title for w in gw.getAllWindows() if w.title]
            print(f"[CogAgent] Offene Fenster: {all_titles}")

        wins = [w for w in gw.getAllWindows()
                if any(k.lower() in w.title.lower() for k in keywords)]

        if not wins:
            # Kein passendes Fenster - alle Fenster ausgeben
            print("[CogAgent] Kein ERP-Fenster gefunden!")
            print(f"[CogAgent] Verfuegbar: {[w.title for w in gw.getAllWindows() if w.title]}")
            return False

        try:
            w = wins[0]
            print(f"[CogAgent] Fokussiere: {w.title}")
            w.restore()
            w.activate()
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[CogAgent] Fokussieren fehlgeschlagen: {e}")
            return False

    def _screenshot(self):
        """Screenshot nur vom Edge/IE-Fenster - kein CMD oder andere Fenster."""
        focused = self._focus_ie()
        import time
        time.sleep(0.8)

        # Versuche Screenshot nur vom Edge-Fenster
        try:
            import pygetwindow as gw
            from PIL import ImageGrab, Image as PILImage
            wins = [w for w in gw.getAllWindows()
                    if any(k.lower() in w.title.lower() for k in
                           ["edge", "Internet Explorer", "erp", "localhost"])]
            if wins:
                w = wins[0]
                # Fenster maximieren und warten bis es wirklich gross ist
                try:
                    w.maximize()
                    time.sleep(0.8)
                    w.activate()
                    time.sleep(0.5)
                except Exception:
                    pass

                # Nochmal Groesse pruefen nach maximize
                import pygetwindow as gw2
                wins2 = [x for x in gw2.getAllWindows() if "edge" in x.title.lower()]
                if wins2:
                    w = wins2[0]

                left   = max(0, w.left)
                top    = max(0, w.top)
                right  = w.left + w.width
                bottom = w.top + w.height
                img    = ImageGrab.grab(bbox=(left, top, right, bottom))
                iw, ih = img.size
                print(f"[CogAgent] Fenster: {iw}x{ih} @ ({left},{top})")

                # Skalieren auf 1920x1080 fuer konsistente Ergebnisse
                from PIL import Image as PILImage2
                img_out = img.resize((1920, 1080), PILImage2.LANCZOS)
                self._window_offset = (left, top)
                self._window_size   = (iw, ih)
                self._scaled_size   = (1920, 1080)
                img_out.save(LAST_SCREENSHOT_PATH)
                return img_out
        except Exception as e:
            print(f"[CogAgent] Fenster-Screenshot fehlgeschlagen: {e}")

        # Fallback: ganzer Screen skaliert
        self._window_offset = (0, 0)
        img = pyautogui.screenshot()
        w, h = img.size
        self._window_size = (w, h)
        if w > 1920:
            from PIL import Image as PILImage
            img = img.resize((1920, 1080), PILImage.LANCZOS)
            self._window_size = (1920, 1080)
        return img


    def _ask_cogagent(self, img, task: str, history: list) -> dict:
        if self.model is None:
            return {"action": "error", "thought": "Modell nicht geladen"}

        import torch

        # History String gemaess CogAgent-Doku
        history_str = "\nHistory steps: "
        for idx, h in enumerate(history):
            history_str += f"\n{idx}. {h.get('operation','')}"

        # CogAgent unterstuetzt verschiedene Format-Strings:
        # - "Action-Operation-Sensitive" -> gibt manchmal keine Koordinaten
        # - "Grounded-Operation" -> erzwingt CLICK(box=[[...]]) Format
        platform_str = "(Platform: WIN)\n"
        format_str   = "(Answer in Grounded-Operation format.)\n"
        query = f"Task: {task}{history_str}\n{platform_str}{format_str}"

        # Korrektes Format laut tokenization_chatglm.py:
        # Ein Dict mit role + image + content (alle zusammen)
        # content MUSS ein String sein, image ist PIL.Image
        conversation = [
            {
                "role":    "user",
                "image":   img.convert("RGB"),
                "content": query,
            }
        ]

        try:
            result = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )

            input_ids = result["input_ids"].to(self.model.device)

            # Attention mask NACH dem Tokenisieren neu berechnen
            # Tokenizer gibt Maske fuer Text (36 tokens) zurueck,
            # aber input_ids enthaelt Bild-Patches (1600 extra tokens)
            # -> Maske muss auf volle Laenge der input_ids passen
            seq_len = input_ids.shape[1]
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.model.device)

            model_inputs = {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
            }
            if "images" in result and result["images"] is not None:
                img_t = result["images"][0].to(self.model.device).to(torch.bfloat16)
                model_inputs["images"] = img_t.unsqueeze(0)  # [1, C, H, W]

        except Exception as e:
            print(f"[CogAgent] Tokenizer Fehler: {e}")
            import traceback; traceback.print_exc()
            return {"action": "error", "thought": str(e)}

        try:
            # ChatGLM4 EOS Tokens korrekt setzen
            eos_token = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            eot_token = self.tokenizer.convert_tokens_to_ids("<|user|>")
            eos = eos_token if eos_token and eos_token > 0 else getattr(self.tokenizer, "eos_token_id", 2) or 2
            pad = getattr(self.tokenizer, "pad_token_id", None) or eos
            stop_tokens = {eos, pad, eot_token} - {0, None}
            print(f"[CogAgent] EOS tokens: {stop_tokens}")

            # Manueller Greedy-Decode statt model.generate()
            # generate() verliert images beim prepare_inputs_for_generation
            input_ids = model_inputs["input_ids"]
            images    = model_inputs.get("images")
            attn_mask = model_inputs.get("attention_mask")
            generated = input_ids.clone()
            past_kv   = None

            with torch.no_grad():
                for step in range(256):
                    if past_kv is None:
                        cur_ids = generated
                        cur_len = generated.shape[1]
                        pos_ids = torch.arange(cur_len, dtype=torch.long,
                                               device=self.model.device).unsqueeze(0)
                        attn    = torch.ones((1, cur_len), dtype=torch.long,
                                             device=self.model.device)
                    else:
                        cur_ids = generated[:, -1:]
                        cur_len = generated.shape[1]
                        pos_ids = torch.tensor([[cur_len - 1]], dtype=torch.long,
                                               device=self.model.device)
                        attn    = torch.ones((1, 1), dtype=torch.long,
                                             device=self.model.device)

                    fwd = {
                        "input_ids":      cur_ids,
                        "position_ids":   pos_ids,
                        "attention_mask": attn,
                        "use_cache":      True,
                        "return_dict":    True,
                    }
                    if images is not None and past_kv is None:
                        fwd["images"] = images
                    if past_kv is not None:
                        fwd["past_key_values"] = past_kv

                    out    = self.model(**fwd)
                    logits = out.logits[:, -1, :]

                    so_far = self.tokenizer.decode(
                        generated[0][input_ids.shape[1]:],
                        skip_special_tokens=True
                    )

                    next_tok = logits.argmax(dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_tok], dim=-1)
                    past_kv   = getattr(out, "past_key_values", None)

                    if next_tok.item() in stop_tokens and step >= 15:
                        break
                    if step >= 10 and ("]])" in so_far or "END()" in so_far):
                        break

            input_len = input_ids.shape[1]
            raw = self.tokenizer.decode(
                generated[0][input_len:],
                skip_special_tokens=True,
            ).strip()
            print(f"[CogAgent] Antwort: {raw[:300]}")
        except Exception as e:
            print(f"[CogAgent] Fehler: {e}")
            import traceback; traceback.print_exc()
            return {"action": "error", "thought": str(e)}

        return self._parse_cogagent_output(raw, task=task)


    def _parse_cogagent_output(self, raw: str, task: str = "") -> dict:
        result = {"raw": raw, "thought": raw}
        # CogAgent gibt Koordinaten im 0-1000 Raum relativ zum skalierten Bild
        # _scaled_size: Groesse des an CogAgent uebergebenen Bildes (z.B. 1920x1080)
        # _window_size: Originale Fenstergroesse fuer Rueckskalierung
        # _window_offset: Fensterposition auf dem Screen
        scaled_w, scaled_h = getattr(self, "_scaled_size", pyautogui.size())
        win_w,    win_h    = getattr(self, "_window_size",  pyautogui.size())
        off_x,    off_y    = getattr(self, "_window_offset", (0, 0))
        # Erst: 0-1000 -> skaliertes Bild (z.B. 1920x1080)
        # Dann: skaliertes Bild -> originales Fenster
        sw, sh = scaled_w, scaled_h

        # OCR zuerst fuer "click on TEXT" Aufgaben - zuverlaessiger als CogAgent-Koordinaten
        import re as _re2
        m_task = _re2.search(r"(?:click|tap|press)\s+(?:on\s+)?(?:the\s+)?([A-Za-z][A-Za-z0-9/\- ]{2,40}?)(?=\s*(?:\d|$|\n|\.|,))", task, _re2.IGNORECASE)
        if m_task:
            print(f"[CogAgent] OCR-first: task={task!r}, m_task={bool(m_task)}")
            target = m_task.group(1).strip()

            ocr_hits = self._find_text_on_screen(target)
            print(f"[CogAgent] OCR-target: {target!r}")

            ocr_hits = self._find_text_on_screen(target)
            if ocr_hits:
                ox, oy, conf = ocr_hits[0]
                scaled_w2, scaled_h2 = getattr(self, "_scaled_size", (sw, sh))
                # OCR-Koordinaten sind im skalierten Bild (z.B. 1920x1080)
                # Direkt als absolute Fenster-Koordinaten zurueckgeben
                # _execute_action soll NICHT nochmal skalieren
                wx = int(ox * win_w / scaled_w2)
                wy = int(oy * win_h / scaled_h2)
                off_x2, off_y2 = getattr(self, "_window_offset", (0, 0))
                abs_x = wx + off_x2
                abs_y = wy + off_y2
                print(f"[CogAgent] OCR-Click: '{target}' -> absolut ({abs_x},{abs_y})")
                result.update({"action": "ocr_click", "coordinate": [abs_x, abs_y], "operation": f"OCR({target})"})
                return result

        # CLICK - nur wenn Koordinaten plausibel (> 10px vom Rand)
        MIN_COORD = 20  # Koordinaten unter diesem Wert sind meist Fehler

        m = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)", raw)
        if m:
            x1,y1,x2,y2 = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
            cx = int((x1+x2)/2 * sw / 1000)
            cy = int((y1+y2)/2 * sh / 1000)
            if cx > MIN_COORD and cy > MIN_COORD:
                result.update({"action":"click","coordinate":[cx,cy],"operation":m.group(0)})
                print(f"[CogAgent] CLICK(4pts): ({cx},{cy})")
                return result
            print(f"[CogAgent] CLICK(4pts) ungueltig: ({cx},{cy}) - zu nahe am Rand")

        m = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)", raw)
        if m:
            x1, y1 = int(m.group(1)), int(m.group(2))
            cx = int(x1 * sw / 1000)
            cy = int(y1 * sh / 1000)
            if cx > MIN_COORD and cy > MIN_COORD:
                result.update({"action":"click","coordinate":[cx,cy],"operation":m.group(0)})
                print(f"[CogAgent] CLICK(2pts): ({cx},{cy})")
                return result
            print(f"[CogAgent] CLICK(2pts) ungueltig: ({cx},{cy}) - zu nahe am Rand")

        # Einzelne Koordinate oder ungueltige Coords -> direkt zu OCR

        # TYPE
        m = re.search(r"TYPE\(.*?text='([^']+)'", raw)
        if m:
            result.update({"action":"type","text":m.group(1),"operation":m.group(0)})
            print(f"[CogAgent] TYPE: {m.group(1)[:50]}")
            return result

        # PRESS_KEY
        m = re.search(r"PRESS_KEY\(key='([^']+)'", raw)
        if m:
            result.update({"action":"key","key":m.group(1),"operation":m.group(0)})
            return result

        # SCROLL
        m = re.search(r"SCROLL\(.*?direction='([^']+)'", raw)
        if m:
            result.update({"action":"scroll","direction":m.group(1),"operation":m.group(0)})
            return result

        # END / aufgabe erledigt
        if "END()" in raw or "task is completed" in raw.lower():
            result["action"] = "done"
            return result

        # OCR Fallback: suche Target-Text aus Task oder CogAgent-Antwort
        search_targets = []
        # 1. Aus dem original task
        import re as _re3
        tm = _re3.search(r"(?:click|tap|press)\s+(?:on\s+)?(?:the\s+)?([A-Za-z0-9/\- ]{3,50})", task, _re3.IGNORECASE)
        if tm: search_targets.append(tm.group(1).strip())
        # 2. Aus CogAgent Antwort
        tm2 = _re3.search(r"['\"]([A-Za-z0-9/\- ]{3,40})['\"]", raw)
        if tm2: search_targets.append(tm2.group(1).strip())
        for target in search_targets:
            ocr_hits2 = self._find_text_on_screen(target)
            if ocr_hits2:
                ox2,oy2,_ = ocr_hits2[0]
                sw3,sh3 = getattr(self,"_scaled_size",(sw,sh))
                wx2 = int(ox2*win_w/sw3); wy2 = int(oy2*win_h/sh3)
                off2 = getattr(self,"_window_offset",(0,0))
                abs2 = (wx2+off2[0], wy2+off2[1])
                print(f"[CogAgent] OCR-Fallback: {target!r} -> absolut {abs2}")
                result.update({"action":"ocr_click","coordinate":list(abs2),"operation":f"OCR({target})"})
                return result

        result["action"] = "unknown"
        print(f"[CogAgent] Unbekanntes Format: {raw[:100]}")
        return result

    def _find_text_on_screen(self, search_text: str, img=None) -> list:
        """Findet Text auf dem Screen per OCR. Gibt Liste von (x,y) zurueck."""
        try:
            import pytesseract
            if img is None:
                img = self._screenshot()
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            results = []
            for i, text in enumerate(data['text']):
                if search_text.lower() in text.lower() and data['conf'][i] > 30:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    results.append((x, y, data['conf'][i]))
            return results
        except Exception as e:
            print(f"[CogAgent] OCR Fehler: {e}")
            return []

    def _execute_action(self, action: dict):
        act = action.get("action")
        if act in ("click", "ocr_click"):
            x, y = action["coordinate"]
            if act == "ocr_click":
                # OCR-Koordinaten sind bereits absolut
                print(f"[CogAgent] OCR-Klick absolut: ({x},{y})")
                pyautogui.click(x, y)
            else:
                # CogAgent-Koordinaten: skaliert -> fenster -> absolut
                scaled_w, scaled_h = getattr(self, "_scaled_size",  pyautogui.size())
                win_w,    win_h    = getattr(self, "_window_size",   pyautogui.size())
                off_x,    off_y    = getattr(self, "_window_offset", (0, 0))
                win_x = int(x * win_w / scaled_w)
                win_y = int(y * win_h / scaled_h)
                abs_x = win_x + off_x
                abs_y = win_y + off_y
                print(f"[CogAgent] Klick: scaled({x},{y}) -> fenster({win_x},{win_y}) + offset({off_x},{off_y}) = absolut({abs_x},{abs_y})")
                pyautogui.click(abs_x, abs_y)
        elif act == "type":
            pyautogui.typewrite(action["text"], interval=0.05)
        elif act == "key":
            key = action["key"].lower()
            pyautogui.hotkey(*key.split("+")) if "+" in key else pyautogui.press(key)
        elif act == "scroll":
            pyautogui.scroll(-3 if action.get("direction") == "down" else 3)
        time.sleep(0.6)

    def _open_erp(self, module_path: str):
        """Navigiert zu einer ERP-Seite im Internet Explorer."""
        import time, subprocess
        import pygetwindow as gw

        url = f"{ERP_URL}/{module_path}"

        # Pruefen ob IE schon offen
        ie_wins = [w for w in gw.getAllWindows()
                   if any(k in w.title for k in ["Internet Explorer", "ERP"])]

        if ie_wins:
            # IE schon offen - Fenster aktivieren und URL per Adressleiste eingeben
            try:
                ie_wins[0].restore()
                ie_wins[0].activate()
                time.sleep(0.5)
                # Adressleiste fokussieren und URL eingeben
                pyautogui.hotkey("alt", "d")
                time.sleep(0.3)
                pyautogui.hotkey("ctrl", "a")
                pyautogui.typewrite(url, interval=0.03)
                pyautogui.press("enter")
                time.sleep(2.5)
                return
            except Exception as e:
                print(f"[CogAgent] IE Navigation fehlgeschlagen: {e}")

        # IE neu starten - verschiedene Pfade probieren
        ie_binary = shutil.which("iexplore")
        if ie_binary:
            subprocess.Popen([ie_binary, url])
            time.sleep(3)
        else:
            # Fallback: ueber cmd starten
            subprocess.Popen(f'start iexplore "{url}"', shell=True)
            time.sleep(3)

        # Fenster in Vordergrund
        for _ in range(5):
            wins = [w for w in gw.getAllWindows()
                    if any(k in w.title for k in ["Internet Explorer", "ERP"])]
            if wins:
                wins[0].activate()
                break
            time.sleep(0.8)


    def process_row(self, row: dict, steps_template: list) -> dict:
        self.aborted = False
        history = []
        try:
            steps = []
            for step in steps_template:
                s = dict(step)
                if "from_col" in s:
                    s["value"] = str(row.get(s["from_col"], ""))
                    del s["from_col"]
                steps.append(s)

            nav = next((s for s in steps if s["action"] == "navigate"), None)
            if nav:
                self._open_erp(nav["target"])

            for step in steps:
                if self.aborted:
                    return {"success": False, "error": "Abgebrochen"}
                if step["action"] == "navigate":
                    continue

                if step["action"] == "fill":
                    task = f"Fill field '{step.get('selector','')}' with value '{step.get('value','')}'."
                elif step["action"] == "click":
                    task = f"Click on '{step.get('selector','')}'."
                elif step["action"] == "select":
                    task = f"Select '{step.get('value','')}' in dropdown '{step.get('selector','')}'."
                else:
                    task = str(step)

                for _ in range(3):
                    img    = self._screenshot()
                    action = self._ask_cogagent(img, task, history)
                    if action.get("action") == "error":
                        break
                    self._execute_action(action)
                    history.append(action)
                    time.sleep(0.8)
                    if self._screen_changed(img, self._screenshot()):
                        break

            return {
                "success":   True,
                "beleg_nr":  self._extract_beleg_nr(self._screenshot()),
                "excel_row": row.get("_row"),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "excel_row": row.get("_row")}

    def _screen_changed(self, before: Image.Image, after: Image.Image) -> bool:
        import numpy as np
        a = np.array(before.convert("L").resize((64, 64)))
        b = np.array(after.convert("L").resize((64, 64)))
        return abs(a.astype(int) - b.astype(int)).mean() > 3.0

    def _extract_beleg_nr(self, img: Image.Image) -> str:
        if self.model is None:
            return ""
        try:
            result = self._ask_cogagent(
                img,
                "What is the order or document number shown? Reply with only the number.",
                []
            )
            raw = result.get("raw", "")
            m = re.search(r"\b([A-Z]{2,}-?\d{5,}|\d{6,})\b", raw)
            return m.group(1) if m else raw[:20]
        except Exception:
            return ""

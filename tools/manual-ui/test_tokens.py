"""
test_tokens.py - Mit Digit-Forcing fuer vollstaendige Koordinaten
"""
import sys, time, re, torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from executor import ERPExecutor
from PIL import Image, ImageGrab
import pyautogui, pygetwindow as gw
from transformers import AutoTokenizer

print("Lade Modell...")
ex  = ERPExecutor()
tok = AutoTokenizer.from_pretrained(
    str(PROJECT_ROOT / "ui"), trust_remote_code=True)

ex._focus_ie(); time.sleep(1.5)

# Edge-Fenster Screenshot
wins = [w for w in gw.getAllWindows() if "edge" in w.title.lower()]
w    = wins[0]
img  = ImageGrab.grab(bbox=(max(0,w.left), max(0,w.top), w.left+w.width, w.top+w.height))
WIN_W, WIN_H = img.size
OFF_X, OFF_Y = max(0,w.left), max(0,w.top)
print(f"Fenster: {WIN_W}x{WIN_H} @ ({OFF_X},{OFF_Y})")

query  = "Task: click on Company/Organization\nHistory steps: \n(Platform: WIN)\n(Answer in Grounded-Operation format.)\n"
conv   = [{"role": "user", "image": img.convert("RGB"), "content": query}]
result = tok.apply_chat_template(conv, add_generation_prompt=True,
                                  tokenize=True, return_tensors="pt", return_dict=True)
input_ids = result["input_ids"].to(ex.model.device)
imgs = None
if "images" in result and result["images"] is not None:
    imgs = result["images"][0].to(ex.model.device).to(torch.bfloat16).unsqueeze(0)

eos = {151336, 151329}
generated = input_ids.clone()
past_kv = None
new_ids = []
# Komma-Token-ID herausfinden
comma_encoded = tok.encode(",", add_special_tokens=False)
comma_id = comma_encoded[0] if comma_encoded else 11
print(f"Komma Token-ID: {comma_id} = {tok.decode([comma_id])!r}")

print(f"\nGeneriere (mit Digit-Forcing):\n")

with torch.no_grad():
    for step in range(60):
        if past_kv is None:
            L = generated.shape[1]
            pos = torch.arange(L, device=ex.model.device).unsqueeze(0)
            fwd = {"input_ids": generated, "position_ids": pos,
                   "attention_mask": torch.ones((1,L), dtype=torch.long, device=ex.model.device),
                   "use_cache": True, "return_dict": True}
            if imgs is not None: fwd["images"] = imgs
        else:
            L = generated.shape[1]
            pos = torch.tensor([[L-1]], device=ex.model.device)
            fwd = {"input_ids": generated[:,-1:], "position_ids": pos,
                   "attention_mask": torch.ones((1,1), dtype=torch.long, device=ex.model.device),
                   "past_key_values": past_kv, "use_cache": True, "return_dict": True}

        out    = ex.model(**fwd)
        logits = out.logits[0, -1, :]
        so_far = tok.decode(new_ids, skip_special_tokens=True)

        # Digit-Forcing: nach Komma oder Doppelpunkt in CLICK-Box
        in_click = bool(re.search(r"CLICK\(box=\[\[", so_far))
        after_sep = bool(re.search(r"CLICK\(box=\[\[[\d\s,]+[,:][\s]*$", so_far))

        if in_click and after_sep:
            # Nur Zahl-Tokens erlauben
            digit_ids = [i for i in range(len(logits))
                        if re.match(r"^\s*\d+\s*$", tok.decode([i]))]
            if digit_ids:
                mask = torch.full_like(logits, float("-inf"))
                mask[digit_ids] = logits[digit_ids]
                logits = mask
                print(f"  step {step:2d}: DIGIT FORCED")

        # Doppelpunkt in CLICK-Box -> Komma
        top1_id = logits.argmax().item()
        top1_str = tok.decode([top1_id])
        if in_click and top1_str.strip() == ":" and re.search(r"\d$", so_far.rstrip()):
            top1_id = comma_id
            print(f"  step {step:2d}: ':' -> ','")

        next_tok  = torch.tensor([[top1_id]], device=ex.model.device)
        new_ids.append(top1_id)
        print(f"  step {step:2d}: [{top1_id:6d}] {tok.decode([top1_id])!r:20s}  so_far: {so_far[-30:]!r}")
        generated = torch.cat([generated, next_tok], dim=-1)
        past_kv   = getattr(out, "past_key_values", None)

        if top1_id in eos: print("  -> EOS"); break
        if "]])" in so_far: print("  -> DONE"); break

raw = tok.decode(new_ids, skip_special_tokens=True).strip()
print(f"\nKomplett: {raw}")

# Koordinaten extrahieren und Klick berechnen
m = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)", raw)
m2 = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)", raw)
m1 = re.search(r"CLICK\(box=\[\[\s*(\d+)", raw)

# CogAgent Koordinaten sind relativ zu 1000x1000 des Eingabebilds
# Eingabebild ist WIN_W x WIN_H -> skalieren entsprechend
if m:
    x1,y1,x2,y2 = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
    cx = int((x1+x2)/2 * WIN_W / 1000)
    cy = int((y1+y2)/2 * WIN_H / 1000)
    print(f"\nCLICK 4pts: [{x1},{y1},{x2},{y2}] -> Fenster ({cx},{cy}) -> Screen ({cx+OFF_X},{cy+OFF_Y})")
elif m2:
    x1,y1 = int(m2.group(1)),int(m2.group(2))
    cx = int(x1 * WIN_W / 1000)
    cy = int(y1 * WIN_H / 1000)
    print(f"\nCLICK 2pts: [{x1},{y1}] -> Fenster ({cx},{cy}) -> Screen ({cx+OFF_X},{cy+OFF_Y})")
elif m1:
    x1 = int(m1.group(1))
    cx = int(x1 * WIN_W / 1000)
    cy = WIN_H // 3
    print(f"\nCLICK 1pt: [{x1}] -> Fenster ({cx},{cy}) -> Screen ({cx+OFF_X},{cy+OFF_Y})")
else:
    print("Keine Koordinaten."); sys.exit(0)

abs_x, abs_y = cx + OFF_X, cy + OFF_Y
confirm = input(f"\nKlicken bei absolut ({abs_x},{abs_y})? (j/n): ").strip().lower()
if confirm == "j":
    ex._focus_ie(); time.sleep(0.5)
    pyautogui.click(abs_x, abs_y)
    print("Geklickt!")
    time.sleep(1)
    pyautogui.screenshot().save(SCRIPT_DIR / "ss_after.png")
    print("ss_after.png gespeichert")

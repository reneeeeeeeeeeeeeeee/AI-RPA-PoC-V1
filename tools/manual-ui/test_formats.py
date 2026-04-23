"""
test_formats.py - Screenshot-Check mit 4K-Skalierung
"""
import sys, time, re, torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from executor import ERPExecutor
import pyautogui
from PIL import Image

ex = ERPExecutor()
if not ex.cogagent_ready():
    print("CogAgent nicht geladen!"); sys.exit(1)

print("Fokussiere Edge...")
ex._focus_ie()
time.sleep(1.5)

# Screenshot mit Skalierung
img_raw = pyautogui.screenshot()
w_raw, h_raw = img_raw.size
img = img_raw.resize((1920, 1080), Image.LANCZOS) if w_raw > 2000 else img_raw
w, h = img.size
img.save(SCRIPT_DIR / "screenshot_current.png")
print(f"Screenshot: {w_raw}x{h_raw} -> {w}x{h} gespeichert")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "ui"), trust_remote_code=True)
query = "Task: click on Company/Organization\nHistory steps: \n(Platform: WIN)\n(Answer in Grounded-Operation format.)\n"
conv  = [{"role": "user", "image": img.convert("RGB"), "content": query}]
result = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
input_ids = result["input_ids"].to(ex.model.device)
imgs = result["images"][0].to(ex.model.device).to(torch.bfloat16).unsqueeze(0) if "images" in result and result["images"] is not None else None

eos = {151336, 151329}
generated = input_ids.clone()
past_kv = None
with torch.no_grad():
    for step in range(80):
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
        out = ex.model(**fwd)
        next_tok = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=-1)
        past_kv = getattr(out, "past_key_values", None)
        if next_tok.item() in eos: break

raw = tok.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
print(f"\nCogAgent:\n{raw}\n")

# Parse
m4 = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)", raw)
m2 = re.search(r"CLICK\(box=\[\[\s*(\d+)[^\d]+(\d+)", raw)
if m4:
    x1,y1,x2,y2 = int(m4.group(1)),int(m4.group(2)),int(m4.group(3)),int(m4.group(4))
    cx = int((x1+x2)/2 * w / 1000)
    cy = int((y1+y2)/2 * h / 1000)
    print(f"CLICK(4pts): [{x1},{y1},{x2},{y2}] -> ({cx},{cy}) auf {w}x{h}")
elif m2:
    x1,y1 = int(m2.group(1)),int(m2.group(2))
    cx = int(x1 * w / 1000)
    cy = int(y1 * h / 1000)
    print(f"CLICK(2pts): [{x1},{y1}] -> ({cx},{cy}) auf {w}x{h}")
else:
    print("Keine Koordinaten."); sys.exit(0)

confirm = input("Klick ausfuehren? (j/n): ").strip().lower()
if confirm == "j":
    # Zurueck zu echten Bildschirmkoordinaten skalieren
    real_x = int(cx * w_raw / w)
    real_y = int(cy * h_raw / h)
    print(f"Klicke auf echten Screen bei ({real_x},{real_y})")
    ex._focus_ie()
    time.sleep(0.5)
    pyautogui.click(real_x, real_y)
    time.sleep(1)
    pyautogui.screenshot().save(SCRIPT_DIR / "screenshot_after.png")
    print("screenshot_after.png gespeichert")

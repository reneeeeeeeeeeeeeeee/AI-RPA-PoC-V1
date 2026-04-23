"""
fix_transformers.py - Final Fix
Behebt: local variable 'torch' referenced before assignment
in modeling_chatglm.py Cache
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "ui"
MODEL = PROJECT_ROOT / "ui"

def fix_torch_conflict(f: Path):
    if not f.exists():
        print(f"  Nicht gefunden: {f}")
        return
    text = f.read_text(encoding="utf-8")

    # Entferne den fehlerhaften Patch der 'import torch' lokal einfuegt
    bad_patch = (
        "                if isinstance(images, list):  # PATCH_IMAGES_TO\n"
        "                    import torch\n"
        "                    images = torch.stack([img if isinstance(img, torch.Tensor) else img[0] for img in images]).to(dtype=inputs_embeds.dtype)\n"
        "                else:\n"
        "                    images = images.to(dtype=inputs_embeds.dtype)"
    )
    original = "                images = images.to(dtype=inputs_embeds.dtype)"

    if bad_patch in text:
        text = text.replace(bad_patch, original)
        f.write_text(text, encoding="utf-8")
        print(f"  [OK] torch-Konflikt behoben: {f.name}")
    elif original in text:
        print(f"  Bereits sauber: {f.name}")
    else:
        print(f"  Zeige images-Zeilen in {f.name}:")
        for i, line in enumerate(text.splitlines(), 1):
            if "images" in line and "dtype" in line:
                print(f"    {i}: {line.rstrip()}")

print("=== Behebe torch-Konflikt ===")
fix_torch_conflict(CACHE / "modeling_chatglm.py")
fix_torch_conflict(MODEL / "modeling_chatglm.py")
print("\nFertig! Starte start.bat neu.")

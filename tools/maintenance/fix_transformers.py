"""
fix_transformers.py - Final Fix
Fixes: local variable 'torch' referenced before assignment
in modeling_chatglm.py cache
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "ui"
MODEL = PROJECT_ROOT / "ui"

def fix_torch_conflict(f: Path):
    if not f.exists():
        print(f"  Not found: {f}")
        return
    text = f.read_text(encoding="utf-8")

    # Remove the faulty patch that inserts 'import torch' locally
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
        print(f"  [OK] torch conflict fixed: {f.name}")
    elif original in text:
        print(f"  Already clean: {f.name}")
    else:
        print(f"  Showing images-related lines in {f.name}:")
        for i, line in enumerate(text.splitlines(), 1):
            if "images" in line and "dtype" in line:
                print(f"    {i}: {line.rstrip()}")

print("=== Fixing torch conflict ===")
fix_torch_conflict(CACHE / "modeling_chatglm.py")
fix_torch_conflict(MODEL / "modeling_chatglm.py")
print("\nDone! Restart start.bat.")

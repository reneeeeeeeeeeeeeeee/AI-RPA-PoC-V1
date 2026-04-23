import transformers
from pathlib import Path

# Alle moeglichen Cache-Pfade suchen
search_roots = [
    Path.home() / ".cache" / "huggingface",
    Path(transformers.__file__).parent,
    Path(transformers.__file__).parent.parent,
]

print("Suche modeling_chatglm.py...")
for root in search_roots:
    if root.exists():
        for p in root.rglob("modeling_chatglm.py"):
            print(f"  GEFUNDEN: {p}")

# Zeige auch wo transformers liegt
print(f"\ntransformers: {transformers.__file__}")
print(f"transformers_modules Pfad: {Path(transformers.__file__).parent.parent / 'transformers_modules'}")
print(f"Existiert: {(Path(transformers.__file__).parent.parent / 'transformers_modules').exists()}")

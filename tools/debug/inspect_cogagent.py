import transformers
from pathlib import Path

# Search all likely cache paths
search_roots = [
    Path.home() / ".cache" / "huggingface",
    Path(transformers.__file__).parent,
    Path(transformers.__file__).parent.parent,
]

print("Searching for modeling_chatglm.py...")
for root in search_roots:
    if root.exists():
        for p in root.rglob("modeling_chatglm.py"):
            print(f"  FOUND: {p}")

# Show where transformers is installed
print(f"\ntransformers: {transformers.__file__}")
print(f"transformers_modules path: {Path(transformers.__file__).parent.parent / 'transformers_modules'}")
print(f"Exists: {(Path(transformers.__file__).parent.parent / 'transformers_modules').exists()}")

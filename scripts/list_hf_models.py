"""
List and categorize all Hugging Face models by nabin2004, focusing on Manim models.
"""
import json
from huggingface_hub import HfApi

api = HfApi()
models = list(api.list_models(author="nabin2004", full=True))

results = []
for m in models:
    files = [f.rfilename for f in (m.siblings or [])]
    tags = [t for t in (m.tags or [])]
    tags_lower = [t.lower() for t in tags]
    name_lower = m.id.lower()
    
    # Check if related to AOS / Manim / Manimator
    is_manim = (
        "manim" in name_lower
        or "manimator" in name_lower
        or any("manim" in t for t in tags_lower)
        or "aos-qwen3-8b" in name_lower
    )
    
    # Classify format
    is_gguf = any(f.endswith(".gguf") for f in files) or "gguf" in name_lower or "gguf" in tags_lower
    is_adapter = any("adapter" in f for f in files) or "lora" in tags_lower or "peft" in tags_lower or "adapter" in name_lower
    is_merged = any(f.endswith(".safetensors") and "adapter" not in f for f in files) or "merged" in name_lower
    
    fmt = []
    if is_gguf:
        fmt.append("GGUF")
    if is_adapter:
        fmt.append("LoRA/Adapter")
    if is_merged:
        fmt.append("Merged (Full Weights)")
    if not fmt:
        fmt.append("Other")

    base = None
    for t in tags:
        if t.startswith("base_model:"):
            base = t.split("base_model:", 1)[1]
            break

    results.append({
        "id": m.id,
        "is_manim": is_manim,
        "format": " + ".join(fmt),
        "base_model": base or "N/A",
        "downloads": m.downloads or 0,
        "likes": m.likes or 0,
        "pipeline_tag": m.pipeline_tag,
        "tags": tags,
        "files": files[:8]
    })

print(f"Total HF models by nabin2004: {len(results)}")
manim_models = [r for r in results if r["is_manim"]]
print(f"Total Manim-related models: {len(manim_models)}")

with open("scripts/manim_hf_models.json", "w", encoding="utf-8") as f:
    json.dump(manim_models, f, indent=2)

print("\n--- Manim Models Overview ---")
for idx, r in enumerate(manim_models, 1):
    print(f"{idx:2d}. {r['id']:<45} | Format: {r['format']:<25} | Base: {r['base_model']}")

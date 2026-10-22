import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT = Path("data/processed/apple/10k_2025.txt")
OUTPUT = Path("data/processed/apple/chunks_2025.jsonl")

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

print("Chargement du tokenizer...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tokenizer = model.tokenizer

texte = INPUT.read_text(encoding="utf-8")

print("Caracteres du rapport :", len(texte))

# IMPORTANT : desactiver explicitement la troncature
encoding = tokenizer(
    texte,
    add_special_tokens=False,
    truncation=False,
    return_offsets_mapping=True,
    verbose=False
)

offsets = encoding["offset_mapping"]
total_tokens = len(offsets)

print("Tokens du rapport :", total_tokens)

assert total_tokens > 200, "Erreur : texte probablement tronque"

chunks = []
step = CHUNK_SIZE - CHUNK_OVERLAP

for start in range(0, total_tokens, step):
    end = min(start + CHUNK_SIZE, total_tokens)

    start_char = offsets[start][0]
    end_char = offsets[end - 1][1]

    chunk_text = texte[start_char:end_char]

    token_count = len(
        tokenizer(
            chunk_text,
            add_special_tokens=True,
            truncation=False,
            verbose=False
        )["input_ids"]
    )

    assert token_count <= model.max_seq_length

    chunks.append({
        "chunk_id": len(chunks) + 1,
        "source": "apple_10k_2025",
        "start_token": start,
        "end_token": end,
        "token_count": token_count,
        "texte": chunk_text
    })

    if end == total_tokens:
        break

assert len(chunks) > 1, "Erreur : un seul chunk genere"

with OUTPUT.open("w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("\n--- RESULTATS ---")
print("Nombre de chunks :", len(chunks))
print("Maximum tokens :", max(c["token_count"] for c in chunks))
print("Chunks trop longs :", sum(
    c["token_count"] > model.max_seq_length
    for c in chunks
))
print("Chunking termine avec succes !")

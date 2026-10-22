import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT = Path("data/processed/apple/chunks_2025.jsonl")
OUTPUT = Path("data/processed/apple/embeddings_2025.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Chargement des chunks...")

with INPUT.open("r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

textes = [chunk["texte"] for chunk in chunks]

print("Nombre de chunks :", len(textes))

print("Chargement du modele...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

print("Generation des embeddings...")

embeddings = model.encode(
    textes,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

assert embeddings.shape == (len(chunks), 384)

np.save(OUTPUT, embeddings)

print("\nGeneration terminee !")
print("Dimensions :", embeddings.shape)
print("Fichier :", OUTPUT)
print("Premier vecteur (10 valeurs) :")
print(embeddings[0][:10])

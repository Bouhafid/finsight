import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/apple"

YEARS = [2023, 2024, 2025]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Chargement du modele MiniLM...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

tokenizer = model.tokenizer

for year in YEARS:
    print(f"\n========== FY{year} ==========")

    input_file = DATA / f"chunks_structured_{year}.jsonl"
    output_file = DATA / f"embeddings_structured_{year}.npy"
    manifest_file = DATA / f"embeddings_manifest_{year}.json"

    raw = input_file.read_bytes()

    chunks = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]

    if not chunks:
        raise RuntimeError(f"Aucun chunk FY{year}")

    ids = [c["chunk_id"] for c in chunks]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Identifiants de chunks dupliques")

    for chunk in chunks:
        if chunk["year"] != year:
            raise RuntimeError("Annee incoherente")

        if not chunk.get("source_url"):
            raise RuntimeError("Source SEC manquante")

        if not chunk.get("accession_number"):
            raise RuntimeError("Accession SEC manquant")

        if not chunk.get("texte", "").strip():
            raise RuntimeError("Chunk vide")

    texts = [c["texte"] for c in chunks]

    # Verification avec le tokenizer reel du modele
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        verbose=False
    )

    lengths = [
        len(tokens)
        for tokens in encoded["input_ids"]
    ]

    if max(lengths) > model.max_seq_length:
        raise RuntimeError(
            f"Chunks trop longs FY{year} : "
            f"{max(lengths)} tokens"
        )

    print("Chunks :", len(chunks))
    print("Maximum tokens :", max(lengths))

    print("Generation des embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.shape != (len(chunks), 384):
        raise RuntimeError(
            "Dimensions des embeddings incorrectes"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embeddings invalides : NaN ou inf"
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1, atol=1e-4):
        raise RuntimeError(
            "Les embeddings ne sont pas normalises"
        )

    # Sauvegarde atomique pour ne pas laisser
    # un fichier final incomplet en cas d'erreur
    temporary = output_file.with_suffix(".npy.tmp")

    with temporary.open("wb") as f:
        np.save(f, embeddings)

    temporary.replace(output_file)

    manifest = {
        "year": year,
        "model": MODEL_NAME,
        "dimensions": 384,
        "normalized": True,
        "chunk_count": len(chunks),
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "embeddings_file": output_file.name
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("Dimensions :", embeddings.shape)
    print("Sauvegarde :", output_file.name)
    print("Manifest :", manifest_file.name)

print("\nEMBEDDINGS MULTI-ANNEES TERMINES")

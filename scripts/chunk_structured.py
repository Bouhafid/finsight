import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/apple"

YEARS = [2023, 2024, 2025]

MAX_TOKENS = 220
OVERLAP = 20

print("Chargement du tokenizer MiniLM...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tokenizer = model.tokenizer
MODEL_LIMIT = model.max_seq_length

assert MAX_TOKENS <= MODEL_LIMIT


def token_count(text):
    return len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            verbose=False
        )["input_ids"]
    )


def prefix_for(year, section, kind):
    section = str(section or "Unknown")[:120]

    return (
        f"Apple Form 10-K FY{year}\n"
        f"Section: {section}\n"
        f"Content type: {kind}\n"
    )


def split_long_text(text, prefix):
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False
    )

    offsets = encoding["offset_mapping"]

    if not offsets:
        return

    start = 0

    while start < len(offsets):
        end = min(start + 180, len(offsets))

        while end > start:
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]

            piece = text[start_char:end_char]

            if token_count(prefix + piece) <= MAX_TOKENS:
                break

            end -= 1

        if end <= start:
            raise RuntimeError(
                "Impossible de respecter la limite de tokens"
            )

        yield piece

        if end == len(offsets):
            break

        start = max(start + 1, end - OVERLAP)


def render_row(row, index):
    values = [
        str(cell.get("text", "")).strip()
        for cell in row
    ]

    return (
        f"Row {index}: "
        + " | ".join(values)
    )


def create_chunks(blocks, year):
    chunks = []

    buffer = []
    buffer_ids = []
    current_section = None

    def add_chunk(
        text, kind, section, block_ids,
        source, row_range=None
    ):
        if not text.strip():
            return

        total = token_count(text)

        if total > MODEL_LIMIT:
            raise RuntimeError(
                f"Chunk trop long : {total} tokens"
            )

        chunks.append({
            "chunk_id": f"apple-{year}-{len(chunks)+1:05d}",
            "company": "Apple",
            "year": year,
            "kind": kind,
            "section": section,
            "source_block_ids": list(block_ids),
            "table_row_range": row_range,
            "source_url": source["source_url"],
            "accession_number": source["accession_number"],
            "token_count": total,
            "texte": text
        })

    def flush(source):
        nonlocal buffer, buffer_ids

        if buffer:
            text = (
                prefix_for(year, current_section, "text")
                + "\n\n".join(buffer)
            )

            add_chunk(
                text,
                "text",
                current_section,
                buffer_ids,
                source
            )

        buffer = []
        buffer_ids = []

    if not blocks:
        raise RuntimeError(f"Aucun bloc FY{year}")

    source = {
        "source_url": blocks[0]["source_url"],
        "accession_number": blocks[0]["accession_number"]
    }

    for block in blocks:
        if block["year"] != year:
            raise RuntimeError("Annee incoherente")

        if (
            block.get("source_url") != source["source_url"]
            or block.get("accession_number")
            != source["accession_number"]
        ):
            raise RuntimeError(
                "Provenance incoherente entre les blocs"
            )

        section = block.get("section") or "Unknown"
        block_id = block["block_id"]
        kind = block["type"]

        if section != current_section:
            flush(source)
            current_section = section

        if kind == "table":
            flush(source)

            rows = block.get("rows") or []

            if not rows:
                raise RuntimeError(
                    f"Tableau sans lignes : bloc {block_id}"
                )

            base = prefix_for(year, section, "table")

            header = (
                "First row context: "
                + render_row(rows[0], 1)
                + "\n"
            )

            if token_count(base + header) > 90:
                header = ""

            table_buffer = []
            row_start = None
            row_end = None

            def flush_table():
                nonlocal table_buffer, row_start, row_end

                if table_buffer:
                    add_chunk(
                        base + header
                        + "\n".join(table_buffer),
                        "table",
                        section,
                        [block_id],
                        source,
                        [row_start, row_end]
                    )

                table_buffer = []
                row_start = None
                row_end = None

            for row_index, row in enumerate(rows, 1):
                line = render_row(row, row_index)

                candidate = "\n".join(
                    table_buffer + [line]
                )

                if (
                    token_count(base + header + candidate)
                    <= MAX_TOKENS
                ):
                    if row_start is None:
                        row_start = row_index

                    row_end = row_index
                    table_buffer.append(line)
                    continue

                flush_table()

                if (
                    token_count(base + header + line)
                    <= MAX_TOKENS
                ):
                    table_buffer = [line]
                    row_start = row_index
                    row_end = row_index
                else:
                    for piece in split_long_text(
                        line,
                        base + header
                    ):
                        add_chunk(
                            base + header + piece,
                            "table",
                            section,
                            [block_id],
                            source,
                            [row_index, row_index]
                        )

            flush_table()
            continue

        content = str(block.get("text", "")).strip()

        if not content:
            continue

        base = prefix_for(year, section, "text")

        candidate = "\n\n".join(
            buffer + [content]
        )

        if token_count(base + candidate) <= MAX_TOKENS:
            buffer.append(content)
            buffer_ids.append(block_id)
            continue

        flush(source)

        if token_count(base + content) <= MAX_TOKENS:
            buffer = [content]
            buffer_ids = [block_id]
        else:
            for piece in split_long_text(content, base):
                add_chunk(
                    base + piece,
                    "text",
                    section,
                    [block_id],
                    source
                )

    flush(source)

    return chunks


for year in YEARS:
    input_file = DATA / f"structured_{year}.jsonl"
    output_file = DATA / f"chunks_structured_{year}.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Fichier manquant : {input_file}. "
            "Executer parse_10k_structured.py."
        )

    with input_file.open("r", encoding="utf-8") as f:
        blocks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    chunks = create_chunks(blocks, year)

    if not chunks:
        raise RuntimeError(f"Aucun chunk FY{year}")

    too_long = sum(
        c["token_count"] > MODEL_LIMIT
        for c in chunks
    )

    assert too_long == 0

    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    table_chunks = sum(
        c["kind"] == "table"
        for c in chunks
    )

    print(f"\n========== FY{year} ==========")
    print("Blocs source :", len(blocks))
    print("Chunks generes :", len(chunks))
    print("Chunks de tableaux :", table_chunks)
    print(
        "Maximum tokens :",
        max(c["token_count"] for c in chunks)
    )
    print("Chunks trop longs :", too_long)
    print("Fichier :", output_file)

print("\nCHUNKING STRUCTURE TERMINE")

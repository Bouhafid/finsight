import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/apple"
OUTPUT = ROOT / "data/processed/apple"
OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = {
    2023: RAW / "10k_2023.html",
    2024: RAW / "10k_2024.html",
    2025: RAW / "10k_2025.html",
}

BLOCK_TAGS = {
    "div", "p", "h1", "h2", "h3", "h4",
    "h5", "h6", "table", "section",
    "article", "main", "ul", "ol",
    "li", "blockquote"
}

REMOVE_TAGS = {
    "head", "script", "style", "noscript",
    "ix:header", "ix:hidden",
    "ix:references", "ix:resources"
}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_table(table):
    rows = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(
            ["td", "th"],
            recursive=False
        )

        if not cells:
            continue

        row = []

        for cell in cells:
            value = clean(
                cell.get_text(" ", strip=True)
            )

            row.append({
                "text": value,
                "colspan": int(
                    cell.get("colspan", 1)
                ),
                "rowspan": int(
                    cell.get("rowspan", 1)
                )
            })

        if any(c["text"] for c in row):
            rows.append(row)

    return rows


def detect_section(text):
    if len(text) > 140:
        return None

    if re.match(
        r"^(ITEM\s+\d{1,2}[A-Z]?\s*\.|PART\s+[IVX]+\b)",
        text,
        flags=re.IGNORECASE
    ):
        return text

    return None


def extract_blocks(file_path, year):
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    print(f"\nLecture du rapport FY{year}...")

    soup = BeautifulSoup(
        file_path.read_bytes(),
        "lxml"
    )

    # Supprimer les metadonnees iXBRL
    # et le contenu technique invisible.
    for tag in list(soup.find_all(True)):
        if tag.name is None or tag.parent is None:
            continue

        name = tag.name.lower()

        style = re.sub(
            r"\s+",
            "",
            tag.get("style", "").lower()
        )

        if (
            name in REMOVE_TAGS
            or tag.has_attr("hidden")
            or "display:none" in style
        ):
            tag.decompose()

    blocks = []
    section = "Unknown"

    def add_block(kind, text, rows=None):
        nonlocal section

        text = clean(text)

        if not text:
            return

        new_section = detect_section(text)

        if new_section:
            section = new_section

        if len(text) < 15 and kind != "heading":
            return

        blocks.append({
            "block_id": len(blocks) + 1,
            "year": year,
            "type": kind,
            "section": section,
            "text": text,
            "rows": rows
        })

    def visit(element):
        if not isinstance(element, Tag):
            return

        name = element.name.lower()

        if name == "table":
            rows = extract_table(element)

            if rows:
                lines = [
                    " | ".join(
                        cell["text"] for cell in row
                    )
                    for row in rows
                ]

                add_block(
                    "table",
                    "\n".join(lines),
                    rows
                )

            return

        if name in {
            "p", "h1", "h2", "h3",
            "h4", "h5", "h6", "li",
            "blockquote"
        }:
            kind = (
                "heading"
                if name.startswith("h")
                else "paragraph"
            )

            add_block(
                kind,
                element.get_text(" ", strip=True)
            )
            return

        buffer = []

        def flush():
            if buffer:
                add_block(
                    "paragraph",
                    " ".join(buffer)
                )
                buffer.clear()

        for child in element.children:
            if isinstance(child, NavigableString):
                buffer.append(str(child))

            elif isinstance(child, Tag):
                if child.name == "br":
                    flush()

                elif child.name in BLOCK_TAGS:
                    flush()
                    visit(child)

                else:
                    buffer.append(
                        child.get_text(" ", strip=True)
                    )

        flush()

    visit(soup.body or soup)

    return blocks


def get_sources():
    manifest_path = RAW / "filings_manifest.json"

    sources = {}

    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        for item in manifest["filings"]:
            sources[item["fiscal_year"]] = {
                "url": item["source_url"],
                "accession": item["accession_number"]
            }

    return sources


sources = get_sources()

for year, file_path in FILES.items():
    if year not in sources:
        raise RuntimeError(
            f"Provenance SEC absente pour FY{year}"
        )

    blocks = extract_blocks(file_path, year)

    if not blocks:
        raise RuntimeError(
            f"Aucun contenu extrait pour FY{year}"
        )

    for block in blocks:
        block["source_url"] = sources[year]["url"]
        block["accession_number"] = (
            sources[year]["accession"]
        )

    output_file = OUTPUT / f"structured_{year}.jsonl"

    with output_file.open(
        "w", encoding="utf-8"
    ) as f:
        for block in blocks:
            f.write(
                json.dumps(
                    block,
                    ensure_ascii=False
                ) + "\n"
            )

    paragraphs = sum(
        b["type"] == "paragraph"
        for b in blocks
    )

    tables = sum(
        b["type"] == "table"
        for b in blocks
    )

    print(f"\n========== FY{year} ==========")
    print("Blocs :", len(blocks))
    print("Paragraphes :", paragraphs)
    print("Tableaux :", tables)
    print("Fichier :", output_file)

    if tables:
        sample = next(
            b for b in blocks
            if b["type"] == "table"
        )

        print("\nPremier tableau (apercu) :")
        print(sample["text"][:500])

print("\nEXTRACTION STRUCTUREE TERMINEE")

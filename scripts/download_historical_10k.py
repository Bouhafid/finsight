import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data" / "raw" / "apple"
DATA.mkdir(parents=True, exist_ok=True)

CIK = "0000320193"
YEARS = [2023, 2024, 2025]

agent = os.getenv("SEC_USER_AGENT")

if not agent:
    raise RuntimeError("SEC_USER_AGENT manquant dans .env")

session = requests.Session()
session.headers.update({
    "User-Agent": agent,
    "Accept-Encoding": "gzip, deflate",
})


def fetch(url):
    response = session.get(url, timeout=90)
    response.raise_for_status()
    time.sleep(0.5)
    return response.content


def extract_10k_filings(data):
    filings = []

    for i, form in enumerate(data["form"]):
        if form != "10-K":
            continue

        filings.append({
            "accession": data["accessionNumber"][i],
            "report_date": data["reportDate"][i],
            "filing_date": data["filingDate"][i],
            "document": data["primaryDocument"][i],
        })

    return filings


print("Lecture de l'historique SEC...")

submissions = json.loads(
    (DATA / "submissions.json").read_text(
        encoding="utf-8"
    )
)

filings = extract_10k_filings(
    submissions["filings"]["recent"]
)


# Consulter les archives supplementaires si necessaire
for archive in submissions["filings"].get("files", []):

    missing = [
        year for year in YEARS
        if not any(
            f["report_date"].startswith(f"{year}-")
            for f in filings
        )
    ]

    if not missing:
        break

    filename = archive["name"]
    archive_path = DATA / filename

    if not archive_path.exists():
        archive_url = (
            "https://data.sec.gov/submissions/"
            + filename
        )

        archive_path.write_bytes(fetch(archive_url))

    old_data = json.loads(
        archive_path.read_text(encoding="utf-8")
    )

    filings.extend(extract_10k_filings(old_data))


manifest = []

for year in YEARS:
    matches = [
        f for f in filings
        if f["report_date"].startswith(f"{year}-")
    ]

    if not matches:
        raise RuntimeError(
            f"Aucun rapport 10-K trouve pour FY{year}"
        )

    # Le premier depot original de l'exercice
    filing = sorted(
        matches,
        key=lambda x: x["filing_date"]
    )[0]

    accession = filing["accession"]

    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(CIK)}/"
        f"{accession.replace('-', '')}/"
        f"{filing['document']}"
    )

    output = DATA / f"10k_{year}.html"

    print(f"\n========== FY{year} ==========")
    print("Date du rapport :", filing["report_date"])
    print("Date du depot :", filing["filing_date"])
    print("Accession :", accession)
    print("URL :", url)

    if not output.exists():
        content = fetch(url)

        if b"<html" not in content[:50000].lower():
            raise RuntimeError(
                f"Le fichier FY{year} ne semble pas etre HTML"
            )

        output.write_bytes(content)
    else:
        print("Fichier existant : reutilisation")

    content = output.read_bytes()

    sha256 = hashlib.sha256(content).hexdigest()

    manifest.append({
        "company": "Apple",
        "cik": CIK,
        "fiscal_year": year,
        "report_date": filing["report_date"],
        "filing_date": filing["filing_date"],
        "accession_number": accession,
        "source_url": url,
        "local_file": output.relative_to(ROOT).as_posix(),
        "sha256": sha256,
    })

    print("Taille :", len(content), "octets")
    print("SHA256 :", sha256[:24] + "...")

manifest_path = DATA / "filings_manifest.json"

manifest_path.write_text(
    json.dumps(
        {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "filings": manifest
        },
        indent=2
    ),
    encoding="utf-8"
)

print("\nTELECHARGEMENT TERMINE")
print("Rapports :", len(manifest))
print("Manifest :", manifest_path)

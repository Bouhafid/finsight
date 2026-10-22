import json
import os
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data" / "raw" / "apple"
YEARS = [2023, 2024, 2025]
CIK = "0000320193"

METRICS = {
    "Total net sales": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "duration",
    ),
    "Net income": ("NetIncomeLoss", "duration"),
    "Operating income": ("OperatingIncomeLoss", "duration"),
    "Total assets": ("Assets", "instant"),
    "Total liabilities": ("Liabilities", "instant"),
}

print("Chargement des fichiers SEC...")

submissions = json.loads(
    (DATA / "submissions.json").read_text(encoding="utf-8")
)

companyfacts = json.loads(
    (DATA / "companyfacts.json").read_text(encoding="utf-8")
)

if str(companyfacts["cik"]) != str(int(CIK)):
    raise ValueError("CIK incorrect dans companyfacts")

gaap = companyfacts["facts"]["us-gaap"]


def extract_filings(data):
    result = []

    for i, form in enumerate(data["form"]):
        if form != "10-K":
            continue

        result.append({
            "accession": data["accessionNumber"][i],
            "report_date": data["reportDate"][i],
            "filing_date": data["filingDate"][i],
            "document": data["primaryDocument"][i],
        })

    return result


filings = extract_filings(submissions["filings"]["recent"])


def find_filing(year):
    matches = [
        f for f in filings
        if f["report_date"].startswith(f"{year}-")
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Rapport FY{year} absent ou ambigu : "
            f"{len(matches)} correspondances"
        )

    return matches[0]


# Recuperer les archives SEC si necessaire
missing = [
    year for year in YEARS
    if not any(
        f["report_date"].startswith(f"{year}-")
        for f in filings
    )
]

if missing:
    agent = os.getenv("SEC_USER_AGENT")

    if not agent:
        raise RuntimeError("SEC_USER_AGENT manquant")

    for archive in submissions["filings"].get("files", []):
        filename = archive["name"]
        filepath = DATA / filename

        if not filepath.exists():
            url = (
                "https://data.sec.gov/submissions/"
                + filename
            )

            response = requests.get(
                url,
                headers={"User-Agent": agent},
                timeout=60,
            )
            response.raise_for_status()
            filepath.write_bytes(response.content)
            time.sleep(0.5)

        data = json.loads(
            filepath.read_text(encoding="utf-8")
        )

        filings.extend(extract_filings(data))

        if all(
            any(
                f["report_date"].startswith(f"{year}-")
                for f in filings
            )
            for year in YEARS
        ):
            break


def extract_fact(tag, kind, filing, year):
    observations = gaap[tag]["units"]["USD"]

    candidates = []

    for item in observations:
        if item.get("accn") != filing["accession"]:
            continue

        if item.get("form") != "10-K":
            continue

        if item.get("fy") != year:
            continue

        if item.get("fp") != "FY":
            continue

        if item.get("end") != filing["report_date"]:
            continue

        start = item.get("start")

        if kind == "duration":
            if not start:
                continue

            days = (
                date.fromisoformat(item["end"])
                - date.fromisoformat(start)
            ).days

            if not 330 <= days <= 380:
                continue

        else:
            if start is not None:
                continue

        candidates.append(item)

    distinct = {
        (str(f["val"]), f.get("start"), f["end"])
        for f in candidates
    }

    if len(distinct) != 1:
        raise ValueError(
            f"Valeur absente ou ambigue : "
            f"{tag} FY{year}"
        )

    return candidates[0]


# Preparer les 15 observations avant toute insertion
records = []
filing_by_year = {}

for year in YEARS:
    filing = find_filing(year)
    filing_by_year[year] = filing

    accession = filing["accession"]

    source_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(CIK)}/{accession.replace('-', '')}/"
        f"{filing['document']}"
    )

    print(f"\n========== FY{year} ==========")

    for name, (tag, kind) in METRICS.items():
        item = extract_fact(tag, kind, filing, year)

        value = Decimal(str(item["val"]))

        records.append({
            "year": year,
            "name": name,
            "value": value,
            "start": item.get("start"),
            "end": item["end"],
            "accession": accession,
            "filing_date": filing["filing_date"],
            "source_url": source_url,
            "concept": "us-gaap:" + tag,
            "kind": kind,
        })

        print(f"{name}: {value:,.0f} USD")

assert len(records) == 15


config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}

print("\nConnexion PostgreSQL...")

with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:
        document_ids = {}

        for year in YEARS:
            filing = filing_by_year[year]
            accession = filing["accession"]

            url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(CIK)}/{accession.replace('-', '')}/"
                f"{filing['document']}"
            )

            cur.execute(
                """
                SELECT id FROM documents
                WHERE entreprise = %s
                  AND type = %s
                  AND exercice = %s
                ORDER BY id
                """,
                ("Apple", "10-K", year),
            )

            existing = cur.fetchall()

            if len(existing) > 1:
                raise ValueError(
                    f"Plusieurs documents FY{year}"
                )

            if existing:
                doc_id = existing[0][0]
            else:
                cur.execute(
                    """
                    INSERT INTO documents
                    (entreprise, type, exercice, url)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    ("Apple", "10-K", year, url),
                )

                doc_id = cur.fetchone()[0]

            document_ids[year] = doc_id

        for r in records:
            origin = (
                f"SEC_XBRL:{r['concept']}:"
                f"{r['accession']}"
            )

            cur.execute(
                """
                INSERT INTO financial_facts (
                    document_id, entreprise, metrique,
                    periode, valeur, unite, origine,
                    xbrl_concept, accession_number,
                    period_start, period_end,
                    filing_date, source_url, period_kind
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (
                    document_id, metrique, periode, origine
                )
                WHERE origine IS NOT NULL
                DO UPDATE SET
                    valeur = EXCLUDED.valeur,
                    unite = EXCLUDED.unite,
                    xbrl_concept = EXCLUDED.xbrl_concept,
                    accession_number = EXCLUDED.accession_number,
                    period_start = EXCLUDED.period_start,
                    period_end = EXCLUDED.period_end,
                    filing_date = EXCLUDED.filing_date,
                    source_url = EXCLUDED.source_url,
                    period_kind = EXCLUDED.period_kind
                """,
                (
                    document_ids[r["year"]],
                    "Apple",
                    r["name"],
                    f"FY{r['year']}",
                    r["value"],
                    "USD",
                    origin,
                    r["concept"],
                    r["accession"],
                    r["start"],
                    r["end"],
                    r["filing_date"],
                    r["source_url"],
                    r["kind"],
                ),
            )

print("\nIMPORT TERMINE")
print("Observations traitees :", len(records))
print("Annees :", YEARS)

"""Load audited XBRL facts (with their SEC provenance) from Postgres."""

from decimal import Decimal

from finsight.config import COMPANY, FORM_TYPE, METRICS
from finsight.metrics import Facts
from finsight.models import Source

QUERY = """
    SELECT d.exercice, f.metrique, f.valeur, f.unite,
           f.source_url, f.accession_number, f.xbrl_concept
    FROM financial_facts f
    JOIN documents d ON d.id = f.document_id
    WHERE d.entreprise = %s
      AND d.type = %s
      AND d.exercice = ANY(%s::int[])
      AND f.periode = 'FY' || d.exercice::text
    ORDER BY d.exercice, f.metrique
"""


class DataIntegrityError(Exception):
    pass


def fetch_financial_facts(conn, years) -> tuple[Facts, dict[str, Source]]:
    """Return fact values keyed by (year, metric) and one source per fact.

    Anything suspicious (wrong unit, missing provenance, duplicate or missing
    metric) stops the pipeline: a report is never built on doubtful figures.
    """
    with conn.cursor() as cur:
        cur.execute(QUERY, (COMPANY, FORM_TYPE, list(years)))
        rows = cur.fetchall()

    facts: Facts = {}
    sources: dict[str, Source] = {}

    for year, metric, value, unit, url, accession, concept in rows:
        if metric not in METRICS:
            continue

        if unit != "USD" or value is None:
            raise DataIntegrityError(f"Invalid XBRL unit or value: FY{year} {metric}")

        if not (url and accession and concept):
            raise DataIntegrityError(f"Missing XBRL provenance: FY{year} {metric}")

        if (year, metric) in facts:
            raise DataIntegrityError(f"Duplicate metric: FY{year} {metric}")

        facts[(year, metric)] = Decimal(value)

        ref = f"F{year}_{METRICS[metric]}"
        sources[ref] = Source(
            id=ref,
            kind="XBRL",
            label=f"FY{year} {metric}: {value} USD (XBRL {concept}, accession {accession})",
            url=url,
        )

    missing = [
        f"FY{year} {metric}" for year in years for metric in METRICS if (year, metric) not in facts
    ]
    if missing:
        raise DataIntegrityError(f"Missing XBRL facts: {', '.join(missing)}")

    return facts, sources

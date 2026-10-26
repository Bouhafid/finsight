#!/bin/sh
# Full data pipeline, in order: SEC files -> XBRL facts -> 10-K chunks -> embeddings -> Postgres.
# Run once on an empty database: docker compose run --rm ingest
set -e

for step in \
    download_sec \
    download_historical_10k \
    ingest_xbrl_multiyear \
    parse_10k_structured \
    chunk_structured \
    generate_structured_embeddings \
    insert_structured_postgres
do
    echo "==> $step"
    python "scripts/$step.py"
done

echo "Ingestion complete."

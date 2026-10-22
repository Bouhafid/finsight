ALTER TABLE financial_facts
ADD COLUMN IF NOT EXISTS xbrl_concept TEXT,
ADD COLUMN IF NOT EXISTS accession_number TEXT,
ADD COLUMN IF NOT EXISTS period_start DATE,
ADD COLUMN IF NOT EXISTS period_end DATE,
ADD COLUMN IF NOT EXISTS filing_date DATE,
ADD COLUMN IF NOT EXISTS source_url TEXT,
ADD COLUMN IF NOT EXISTS period_kind TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique_source
ON financial_facts (document_id, metrique, periode, origine)
WHERE origine IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_facts_company_period
ON financial_facts (entreprise, periode, metrique);

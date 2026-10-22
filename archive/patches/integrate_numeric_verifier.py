from pathlib import Path

path = Path("scripts/financial_rag.py")
code = path.read_text(encoding="utf-8-sig")

if "from numeric_verifier import" in code:
    raise RuntimeError(
        "Le verificateur semble deja integre. "
        "Patch non applique."
    )

original = code


def replace_once(old, new):
    global code

    count = code.count(old)

    if count != 1:
        raise RuntimeError(
            f"Bloc introuvable ou ambigu : {count}\n"
            f"{old[:120]}"
        )

    code = code.replace(old, new, 1)


replace_once(
    "from llm_client import get_llm",
    """from llm_client import get_llm
from numeric_verifier import (
    build_numeric_claims,
    filter_document_claims,
)"""
)

replace_once(
    "- Use the financial facts for exact amounts.",
    """- Do NOT generate numerical financial statements.
- Do NOT write amounts, percentages, ratios or growth rates.
- Python generates the numerical report separately.
- Generate ONLY qualitative claims supported by DOCUMENT sources."""
)

replace_once(
    '    print("\\n4. Generation du rapport avec Qwen...")',
    '''    verified_numeric_claims = build_numeric_claims(
        facts,
        years
    )

    for claim in verified_numeric_claims:
        for ref in claim["source_ids"]:
            if ref not in sources:
                raise RuntimeError(
                    f"Reference numerique absente : {ref}"
                )

    print("\\n4. Generation du rapport avec Qwen...")'''
)

replace_once(
    '''    if not claims:
        print("Preuves insuffisantes pour repondre.")
        return

    cited = set()

    for claim in claims:
''',
    '''    document_claims, rejected = filter_document_claims(
        claims
    )

    if rejected:
        print(
            f"\\n{rejected} affirmation(s) du LLM "
            "ecartee(s) du rapport final."
        )

    print(
        "Les chiffres sont issus exclusivement "
        "des calculs Python/XBRL."
    )

    cited = set()

    for claim in verified_numeric_claims + document_claims:
'''
)

backup = path.with_name(
    "financial_rag_before_numeric_verifier.py"
)

backup.write_text(original, encoding="utf-8")
path.write_text(code, encoding="utf-8")

print("INTEGRATION NUMERIQUE TERMINEE")
print("Sauvegarde :", backup)

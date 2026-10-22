from pathlib import Path

path = Path("scripts/financial_rag.py")

if not path.exists():
    raise FileNotFoundError(path)

code = path.read_text(encoding="utf-8-sig")
original = code

def replace_once(old, new):
    global code
    count = code.count(old)

    if count != 1:
        raise RuntimeError(
            f"Modification impossible : bloc trouve {count} fois.\n"
            f"Debut recherche : {old[:90]!r}"
        )

    code = code.replace(old, new, 1)


# 1. Preparer explicitement les identifiants autorises
replace_once(
    '    allowed = set(sources)\n',
    '''    allowed = set(sources)
    valid_ids = "\\n".join(sorted(allowed))
'''
)


# 2. Donner au LLM la liste exacte des citations
replace_once(
    '        + "\\n\\n---\\n\\n".join(docs)\n    )',
    '''        + "\\n\\n---\\n\\n".join(docs)
        + "\\n\\nAUTHORIZED SOURCE IDS (EXACT SPELLING):\\n"
        + valid_ids
        + "\\n\\nIMPORTANT: Copy source_ids exactly from this list."
        + " Never invent or rename a source ID."
    )'''
)


# 3. Accepter seulement une petite difference de format,
# par exemple "[F2025_REVENUE]" au lieu de "F2025_REVENUE".
# Ne jamais accepter un identifiant qui n'existe pas.
replace_once(
    '''                if any(
                    not isinstance(ref, str) or ref not in allowed
                    for ref in ids
                ):
                    raise ValueError("Citation inconnue")
''',
    '''                normalized = []

                for ref in ids:
                    if not isinstance(ref, str):
                        raise ValueError(
                            f"Type de citation invalide : {ref!r}"
                        )

                    clean_ref = ref.strip().strip("[]").strip()

                    if clean_ref not in allowed:
                        raise ValueError(
                            f"Citation inconnue : {ref!r}"
                        )

                    normalized.append(clean_ref)

                claim["source_ids"] = list(
                    dict.fromkeys(normalized)
                )
'''
)


# 4. Renvoyer a Qwen l'erreur precise et la liste autorisee
replace_once(
    '''                        f"Fix the JSON error: {error}. "
                        "Use only the provided source IDs."
''',
    '''                        f"Fix the JSON error: {error}. "
                        "Return the full corrected JSON. "
                        "Copy source IDs exactly from this list:\\n"
                        + valid_ids
'''
)


# 5. Si Qwen echoue encore, fournir uniquement les chiffres
# verifies, plutot qu'inventer des citations ou crasher.
replace_once(
    '''    raise RuntimeError(
        "Le LLM n'a pas produit de rapport valide"
    )
''',
    '''    print(
        "\\nAVERTISSEMENT : Qwen n'a pas respecte "
        "les citations. Retour aux donnees SQL verifiees."
    )

    fallback = []

    years = sorted({
        int(ref[1:5])
        for ref in allowed
        if ref.startswith("F")
        and ref.endswith("_REVENUE")
        and ref[1:5].isdigit()
    })

    for year in years:
        revenue_ref = f"F{year}_REVENUE"
        income_ref = f"F{year}_NET_INCOME"
        margin_ref = f"C{year}_NET_MARGIN"

        revenue_line = next(
            (line for line in facts_text
             if line.startswith(f"[{revenue_ref}] ")),
            None
        )

        income_line = next(
            (line for line in facts_text
             if line.startswith(f"[{income_ref}] ")),
            None
        )

        if revenue_line is None or income_line is None:
            continue

        revenue = Decimal(
            revenue_line.split(" | ")[2].split()[0]
        ) / Decimal("1000000000")

        income = Decimal(
            income_line.split(" | ")[2].split()[0]
        ) / Decimal("1000000000")

        text = (
            f"FY{year} : chiffre d'affaires de "
            f"{revenue:.3f} milliards USD ; "
            f"benefice net de {income:.3f} milliards USD."
        )

        references = [revenue_ref, income_ref]

        margin_line = next(
            (line for line in calculations
             if line.startswith(f"[{margin_ref}] ")),
            None
        )

        if margin_line is not None:
            margin = margin_line.split(
                "NET_MARGIN = ", 1
            )[1].split("%", 1)[0]

            text += f" Marge nette : {margin}%."
            references.append(margin_ref)

        fallback.append({
            "text": text,
            "source_ids": references
        })

    if not fallback:
        raise RuntimeError(
            "Aucune donnee financiere valide pour le repli"
        )

    return fallback
'''
)

backup = path.with_name("financial_rag_before_citation_fix.py")

backup.write_text(original, encoding="utf-8")
path.write_text(code, encoding="utf-8")

print("PATCH APPLIQUE")
print("Sauvegarde :", backup)
print("Citations autorisees explicites")
print("Validation stricte conservee")
print("Repli SQL deterministe active")

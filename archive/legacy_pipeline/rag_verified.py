import sys

from rag_answer import retrieve
from cited_answer import generate_cited_answer
from claim_verifier import verify_claim


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What are Apple's main business risks?"

    print("\nQUESTION :", question)

    # Etape 1 : recherche vectorielle
    print("\n1. Recherche vectorielle...")

    results = retrieve(question, top_k=5)

    if not results:
        print("Aucun document trouve.")
        return

    # Dictionnaire des sources autorisees
    sources = {
        chunk_id: {
            "texte": texte,
            "url": url
        }
        for chunk_id, texte, similarity, url in results
    }

    # Etape 2 : generation avec citations
    print("\n2. Generation des affirmations...")

    answer = generate_cited_answer(question, results)

    if answer["insufficient_context"]:
        print("Contexte insuffisant.")
        return

    # Etape 3 : verification
    print("\n3. Verification des affirmations...")

    verified_claims = []

    for number, claim in enumerate(answer["claims"], 1):

        text = claim["text"]
        chunk_ids = claim["chunk_ids"]

        evidence = "\n\n".join(
            f"[chunk:{chunk_id}]\n"
            f"{sources[chunk_id]['texte']}"
            for chunk_id in chunk_ids
        )

        try:
            verification = verify_claim(
                text,
                evidence
            )

        except Exception as error:
            verification = {
                "verdict": "INSUFFICIENT",
                "reason": (
                    "Verification unavailable: "
                    f"{type(error).__name__}"
                )
            }

        verdict = verification["verdict"]

        print(f"\nAffirmation {number}")
        print("Texte :", text)
        print("Verdict :", verdict)
        print("Raison :", verification["reason"])

        if verdict == "SUPPORTED":
            verified_claims.append(claim)

    # Etape 4 : resultat
    print("\n========== REPONSE CONTROLEE ==========\n")

    if not verified_claims:
        print(
            "Aucune affirmation n'a ete retenue "
            "par le controle automatique."
        )
        return

    for claim in verified_claims:
        citations = " ".join(
            f"[chunk:{i}]"
            for i in claim["chunk_ids"]
        )

        print(f"- {claim['text']} {citations}")

    print("\n========== SOURCES ==========\n")

    used_ids = {
        i
        for claim in verified_claims
        for i in claim["chunk_ids"]
    }

    for chunk_id in sorted(used_ids):
        source = sources[chunk_id]

        print("Chunk :", chunk_id)
        print("Document :", source["url"])
        print("Extrait :", source["texte"][:350])
        print()


if __name__ == "__main__":
    main()

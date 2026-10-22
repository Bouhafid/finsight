from pathlib import Path
from bs4 import BeautifulSoup
import re
import warnings

INPUT = Path("data/raw/apple/10k_latest.html")
OUTPUT = Path("data/processed/apple/10k_2025.txt")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Lecture du rapport Apple...")

# Analyser le document HTML
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    soup = BeautifulSoup(INPUT.read_bytes(), "lxml")

# Supprimer les metadonnees XBRL et le contenu invisible
for tag in soup.find_all(True):
    if tag.name and tag.name.lower() in {
        "head", "script", "style", "noscript",
        "ix:header", "ix:hidden", "ix:references",
        "ix:resources"
    }:
        tag.decompose()

# Conserver un espace entre les elements HTML
texte = soup.get_text(separator=" ", strip=True)

# Nettoyer les espaces multiples
texte = re.sub(r"\s+", " ", texte).strip()

# Sauvegarder le texte
OUTPUT.write_text(texte, encoding="utf-8")

print("Extraction terminee !")
print("Caracteres :", len(texte))
print("Mots :", len(texte.split()))

print("\n--- APERCU ---\n")
print(texte[:1000])

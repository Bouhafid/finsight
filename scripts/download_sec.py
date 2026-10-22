import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CIK = "0000320193"
OUTPUT = Path("data/raw/apple")
OUTPUT.mkdir(parents=True, exist_ok=True)

user_agent = os.getenv("SEC_USER_AGENT")

if not user_agent:
    raise RuntimeError("SEC_USER_AGENT manquant dans .env")

session = requests.Session()
session.headers.update({
    "User-Agent": user_agent,
    "Accept-Encoding": "gzip, deflate",
})


def fetch(url):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(0.5)
    return response


print("1. Telechargement des submissions...")

url = f"https://data.sec.gov/submissions/CIK{CIK}.json"
submissions = fetch(url).json()

with open(OUTPUT / "submissions.json", "w", encoding="utf-8") as f:
    json.dump(submissions, f, indent=2)

print("Entreprise :", submissions["name"])


print("2. Telechargement des donnees XBRL...")

url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
companyfacts = fetch(url).json()

with open(OUTPUT / "companyfacts.json", "w", encoding="utf-8") as f:
    json.dump(companyfacts, f, indent=2)


print("3. Recherche du dernier rapport 10-K...")

recent = submissions["filings"]["recent"]
indices = [
    i for i, form in enumerate(recent["form"])
    if form == "10-K"
]

if not indices:
    raise RuntimeError("Aucun 10-K dans les depots recents")

i = indices[0]

accession = recent["accessionNumber"][i].replace("-", "")
document = recent["primaryDocument"][i]
report_date = recent["reportDate"][i]

url = (
    f"https://www.sec.gov/Archives/edgar/data/"
    f"{int(CIK)}/{accession}/{document}"
)

print("Rapport :", report_date)
print("URL :", url)

try:
    html = fetch(url).content
    (OUTPUT / "10k_latest.html").write_bytes(html)
    print("Rapport HTML sauvegarde")
except requests.RequestException as error:
    print("Erreur telechargement HTML :", error)

print("Termine !")

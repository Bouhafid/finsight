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


print("Termine !")

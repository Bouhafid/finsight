# FinSight — Résultats et journal des décisions

Ce document rassemble les **mesures** du projet et le **pourquoi** de chaque changement : ce que
j'ai modifié, pour quelle raison, comment, et comment je l'ai vérifié. Les entrées les plus
récentes sont en haut. Les fichiers de résultats bruts sont dans [`eval/results/`](../eval/results).

---

## 1. Résultats d'évaluation

### 1.1 Comment lire les chiffres

- **Le banc de recherche** : 91 questions ([`eval/questions.jsonl`](../eval/questions.jsonl)),
  dont 86 ont une réponse dans les 10-K (71 en anglais, 15 en français). Pour chacune, je connais
  la phrase exacte du rapport qui contient la réponse.
- **Hit@k** : part des questions dont le bon passage apparaît dans les **k premiers** résultats.
- **MRR** (de 0 à 1) : récompense un bon passage placé **tout en haut**. 1er = 1 ; 2e = 0,5 ;
  3e = 0,33 ; absent = 0.
- **Le chiffre qui compte le plus est Hit@2** : en production, l'application garde 2 passages
  par année (`top_k=2`). Si le bon passage n'y est pas, le LLM ne peut rien affirmer de vérifié.
- **Latence** : temps médian de recherche par question et par année, sur le CPU de mon PC
  (8 cœurs), modèles déjà chargés.

### 1.2 Recherche : comparaison des variantes

| Variante | Date | Hit@1 | Hit@2 | Hit@3 | Hit@5 | Hit@10 | MRR | Hit@2 EN | Hit@2 FR | Latence |
|---|---|---|---|---|---|---|---|---|---|---|
| **dense-minilm** (production actuelle) | 2026-10-23 | 0,56 | 0,70 | 0,77 | 0,85 | 0,88 | 0,67 | 0,79 | 0,27 | 15 ms |
| bm25 | 2026-10-27 | 0,56 | 0,76 | 0,78 | 0,81 | 0,83 | 0,68 | 0,86 | 0,27 | 1 ms |
| hybrid-minilm | 2026-10-27 | 0,62 | 0,80 | 0,83 | 0,88 | 0,90 | 0,73 | 0,90 | 0,33 | 18 ms |
| dense-e5-small | 2026-10-27 | 0,63 | 0,80 | 0,93 | 0,98 | 0,99 | 0,77 | 0,86 | 0,53 | 27 ms |
| hybrid-e5-small | 2026-10-27 | 0,67 | 0,84 | 0,88 | 0,92 | 0,98 | 0,79 | 0,92 | 0,47 | 30 ms |
| dense-e5-small+rerank-mini | 2026-10-28 | 0,88 | 0,97 | 0,99 | 1,00 | 1,00 | 0,94 | 0,97 | 0,93 | 3,7 s |
| dense-e5-small+rerank | 2026-10-27 | 0,91 | 0,99 | 1,00 | 1,00 | 1,00 | 0,95 | 0,99 | 1,00 | 31 s |
| hybrid-e5-small+rerank | 2026-10-28 | 0,88 | 0,97 | 0,99 | 0,99 | 0,99 | 0,93 | 0,97 | 0,93 | 39 s* |

\* Mesurée en même temps qu'une autre variante : la latence est surestimée. La qualité n'est
pas affectée.

Composants :
- **minilm** : `sentence-transformers/all-MiniLM-L6-v2` (anglais, 384 dim., vecteurs de la base) ;
- **e5-small** : `intfloat/multilingual-e5-small` (multilingue, 384 dim., préfixes `query:` /
  `passage:`), extraits encodés à la volée ;
- **bm25** : Okapi BM25 (k1 = 1,5, b = 0,75), mots en minuscules, sans racinisation ;
- **hybrid** : fusion des classements (Reciprocal Rank Fusion, k = 60, 50 candidats par moteur) ;
- **rerank** : cross-encodeur `BAAI/bge-reranker-v2-m3` (environ 570 M de paramètres) sur les
  20 premiers résultats ; **rerank-mini** : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
  (environ 120 M de paramètres), même principe.

Commande : `finsight-eval retrieval --name <variante>`.

### 1.3 Analyse

1. **Le modèle multilingue est le gain le moins cher.** e5-small fait passer Hit@2 de 0,70 à
   0,80, et de 0,27 à 0,53 en français, pour 12 ms de plus. Il retrouve les 3 questions anglaises
   jamais trouvées par MiniLM (q005, q024, q038 : rangs 1, 2 et 2). Mais il n'est pas meilleur
   partout : 7 questions bien classées par MiniLM descendent au rang 3 ou 4 (q016, q026, q040,
   q049, q063, q110, q111). Même taille de vecteur (384) : la base n'a pas à changer de schéma,
   il suffit de recalculer les embeddings.
2. **La recherche hybride aide l'anglais mais pénalise le français.** Avec e5, Hit@2 anglais passe
   de 0,86 à 0,92, mais le français baisse de 0,53 à 0,47 : les mots d'une question française ne
   se retrouvent pas dans un texte anglais, donc BM25 n'ajoute que du bruit. Avec le gros
   reranker, partir de la recherche hybride donne aussi un moins bon résultat (0,97 contre 0,99).
3. **Le reranking donne la meilleure qualité, mais coûte cher sur CPU.** Le gros reranker atteint
   Hit@2 = 0,99 (1,00 en français) pour 31 s par question et par année. Le petit fait presque
   aussi bien (0,97 ; 0,93 en français ; 3 questions hors des 2 premiers : q002, q067, q106) en
   3,7 s. La recherche se fait pour chaque année demandée : il faut multiplier ce temps
   par le nombre d'années.
4. **Alternative sans reranker** : avec e5, le bon passage est dans les 3 premiers pour 93 % des
   questions (98 % dans les 5 premiers). Passer `top_k` de 2 à 3 donnerait l'essentiel du gain
   sans latence de recherche, mais envoie 50 % de texte en plus au LLM (coût et durée de la
   vérification). Ce compromis est à mesurer de bout en bout, avec un LLM rapide.
5. **Prochaine décision** : passer la production à e5-small (gain sûr, coût négligeable), puis
   choisir entre `top_k=3` et le petit reranker après une évaluation de bout en bout.

**Prudence** : je n'ai pas optimisé les réglages (nombre de candidats, k de la fusion) sur ce
banc, pour ne pas surajuster sur 86 questions. Les composants sont des modèles publics, utilisés
tels quels. Les 15 questions françaises sont peu nombreuses : une question vaut 7 points. Le jeu
a été écrit à partir du texte des 10-K, ce qui peut avantager les formulations proches du rapport.

### 1.4 Diagnostic initial (référence `dense-minilm`, 2026-10-23)

**Par langue**

| Langue | Questions | Hit@1 | Hit@2 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|---|
| Anglais | 71 | 0,63 | 0,79 | 0,86 | 0,94 | 0,96 | 0,76 |
| Français | 15 | 0,20 | 0,27 | 0,33 | 0,40 | 0,53 | 0,29 |

**Par année**

| Année | Questions | Hit@1 | Hit@2 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| 2023 | 8 | 0,62 | 0,75 | 1,00 | 1,00 | 0,75 |
| 2024 | 11 | 0,64 | 0,73 | 0,91 | 0,91 | 0,73 |
| 2025 | 67 | 0,54 | 0,69 | 0,82 | 0,87 | 0,65 |

Questions jamais trouvées dans le top 10 : q005, q024, q038 (anglais) et q103, q104, q106, q112,
q113, q114, q115 (français).

Ce diagnostic m'a donné trois pistes, toutes testées dans le tableau 1.2 :
1. le français est le point faible majeur (Hit@2 de 0,27 contre 0,79 en anglais) : MiniLM est
   un modèle anglais → **modèle multilingue** ;
2. en anglais, le bon passage est souvent proche mais pas en tête (Hit@10 = 0,96, Hit@2 = 0,79)
   → **reranker** ;
3. les questions avec des termes précis échouent (q038 « fourth quarter of 2025 », q024
   « cyberattacks ») → **recherche hybride** avec BM25.

### 1.5 Vérificateur de citations (`finsight-eval verifier`)

**Le banc** : 64 affirmations ([`eval/claims.jsonl`](../eval/claims.jsonl)), chacune vérifiée
contre le vrai extrait du 10-K, comme en production. 30 sont vraies et doivent être
**acceptées**, dont 5 paraphrases. 34 sont des pièges qui doivent être **rejetés** : mauvaise
entité, négation, sens inversé, cause inventée, risque présenté comme un fait, injection de
prompt…

**Les métriques** :
- **Taux de faux acceptés** : part des pièges acceptés, c'est-à-dire des hallucinations qui
  arriveraient à l'utilisateur. C'est le chiffre de sécurité, le plus important.
- **Taux de faux rejets** : part des affirmations vraies perdues. C'est le prix de la prudence.
- **Échecs de format** : affirmations sans réponse valide du LLM après 3 essais. Elles sont
  comptées comme rejetées, comme en production.

| Modèle | Date | Exactitude | Faux acceptés | Faux rejets | Précision | Échecs de format | s / affirmation |
|---|---|---|---|---|---|---|---|
| **qwen2.5:3b** (Ollama, CPU) | 2026-10-23 | 0,81 | **0,06** (2/34) | 0,33 (10/30) | 0,91 | 18/64 | 116 |

**Analyse** ([`eval/results/verifier_qwen2.5-3b.json`](../eval/results/verifier_qwen2.5-3b.json)) :
1. **Très peu d'hallucinations passent** : 2 sur 34 (c017, c020). Les deux sont des « risques
   présentés comme des faits » : le 10-K dit « pourrait », l'affirmation dit « est ». C'est la
   seule catégorie de pièges mal détectée (3/5). Tous les autres pièges sont rejetés.
2. **Mais une partie de ces rejets est due au hasard** : 12 des 34 pièges ne sont rejetés que
   parce que le modèle n'a pas produit de réponse valide en 3 essais. Sur les 46 affirmations où
   le modèle a vraiment tranché, l'exactitude est de 0,87 (40/46).
3. **Le petit modèle perd beaucoup de vraies affirmations** : 10 sur 30. 6 sont des échecs de
   format (citations pas recopiées mot pour mot, JSON invalide) et 4 sont des propositions jugées
   à tort INSUFFICIENT ou CONTRADICTED.
4. **Suite** : refaire le même banc avec Claude Haiku sur Amazon Bedrock. J'attends beaucoup
   moins d'échecs de format, donc moins de faux rejets, et je vérifierai que le taux de faux
   acceptés reste bas.
5. **Limites** : 64 affirmations, et certaines catégories n'en ont que 1 ou 2. La durée moyenne
   est gonflée par une mise en veille du PC pendant la mesure (c030 : 1 218 s).

---

## 2. Journal des décisions

### 2026-10-28 — README
- **Quoi** : `README.md` (nouveau).
- **Pourquoi** : la page d'accueil du dépôt doit permettre de comprendre le projet en deux
  minutes : le problème, l'approche, un exemple réel, les résultats et comment le lancer.
  `pyproject.toml` déclarait déjà ce fichier, qui manquait.
- **Comment** : un exemple de réponse tiré d'une vraie exécution de l'API (6 affirmations
  proposées, 4 rejetées faute de citation exacte) ; schéma Mermaid, affiché directement par
  GitHub ; résultats résumés avec un renvoi vers ce journal pour le détail ; limites écrites
  explicitement (lenteur du LLM local, Bedrock pas encore évalué en conditions réelles).
- **Vérification** : chiffres repris de `eval/results/` et de la section 1 ; commandes reprises
  des sections déjà testées.

### 2026-10-28 — Résultat de la variante hybride + gros reranker, latence remesurée
- **Quoi** : `src/finsight/evaluation.py` : variante `hybrid-e5-small+rerank` remise dans
  `VARIANTS` ; `eval/results/retrieval_hybrid-e5-small+rerank.json` ajouté ;
  `eval/results/retrieval_dense-e5-small+rerank-mini.json` remesuré ; tableau 1.2 et analyse 1.3.
- **Pourquoi** : je pensais avoir arrêté la variante hybride + gros reranker, jugée trop lente.
  Elle avait en fait terminé, et elle a tourné **en même temps** que la mesure du petit reranker :
  les deux latences étaient faussées par le partage du processeur.
- **Comment** : la variante revient dans le code pour que son résultat reste reproductible. Le
  petit reranker est remesuré seul : 3,7 s au lieu de 4,1 s, qualité identique. La latence de la variante hybride reste marquée comme
  surestimée : la remesurer prendrait près d'une heure pour une variante déjà écartée.
- **Résultat** : hybride + gros reranker : Hit@2 = 0,97, moins bien que le gros reranker seul
  (0,99), et plus lent. La recherche hybride n'est pas retenue.
- **Vérification** : `pytest` et `ruff` (voir le commit).

### 2026-10-28 — Comparaison de 6 variantes de recherche
- **Quoi** :
  - `src/finsight/evaluation.py` : `DenseRetriever` accepte n'importe quel modèle d'embedding
    (avec préfixes E5) ; nouvelles classes `BM25Retriever`, `HybridRetriever` (fusion RRF) et
    `RerankRetriever` (cross-encodeur) ; dictionnaire `VARIANTS` des configurations nommées ;
    `--name` n'accepte plus qu'une variante de ce dictionnaire ; latence médiane mesurée par
    question (`latency_ms`), après un préchargement des modèles et des index ;
  - `tests/test_evaluation.py` : 4 tests (BM25, fusion, reranking, latence) ;
  - `eval/results/retrieval_<variante>.json` : nouveaux résultats ; `retrieval_dense-minilm.json`
    régénéré avec la latence.
- **Pourquoi** : tester, une par une et sur le même banc, les trois pistes du diagnostic du
  2026-10-23 (modèle multilingue, recherche hybride, reranking), avant de toucher à la
  production.
- **Comment** :
  - Une variante = une configuration écrite dans le code et un nom : chaque fichier de résultats
    se reproduit avec `finsight-eval retrieval --name <variante>`.
  - Les autres modèles encodent les extraits à la volée (1 115 extraits, environ 2 min sur CPU) ;
    la base et les fichiers `.npy` de production ne changent pas.
  - BM25 écrit à la main (une vingtaine de lignes) plutôt qu'avec une bibliothèque : pas de
    dépendance pour un outil d'évaluation, et la formule reste lisible.
  - Fusion par rangs (RRF) plutôt que par scores : les scores BM25 et cosinus n'ont pas la même
    échelle, les rangs si.
  - Modèles publics utilisés tels quels, réglages standard non optimisés sur le banc, pour ne pas
    surajuster sur 86 questions.
- **Résultats** : voir 1.2 et 1.3.
- **Vérification** : la référence `dense-minilm` relancée avec le nouveau code donne exactement
  les mêmes rangs, question par question ; `pytest` → 164 tests passés ; `ruff check` et
  `ruff format --check` OK.

### 2026-10-27 — Fournisseur LLM Amazon Bedrock (Claude) et interface LLM neutre
- **Quoi** :
  - `src/finsight/bedrock.py` (nouveau) : `BedrockClaude`, client Claude sur Amazon Bedrock via le
    SDK Anthropic (`AnthropicBedrockMantle`). Modèle par défaut : `anthropic.claude-haiku-5-5` ;
  - `src/finsight/llm.py` : interface `LLM` (Protocol) ; erreurs neutres `LLMError` et
    `LLMTimeout` ; l'ancien client devient `OpenAICompatibleLLM` ; `LLM_PROVIDER=bedrock` ;
  - `src/finsight/generation.py`, `src/finsight/verification.py` : schémas JSON de sortie
    (`DRAFT_SCHEMA`, `VERDICT_SCHEMA`) transmis au LLM ; les erreurs attrapées sont les erreurs
    neutres, et non plus celles du SDK OpenAI ;
  - `pyproject.toml` : dépendance `anthropic[bedrock]` ; `.env.example` et `docker-compose.yml` :
    variables `AWS_REGION`, `AWS_BEARER_TOKEN_BEDROCK`, `BEDROCK_MODEL`, `LLM_EFFORT` ;
  - `tests/test_llm.py` (nouveau, 7 tests) ; `tests/conftest.py` : le faux LLM accepte `schema`.
- **Pourquoi** : qwen2.5:3b sur CPU met environ 20 minutes par question et rate le format JSON
  dans 18 cas sur 64 : impossible de faire une démo ni d'itérer sur des améliorations. J'ai
  choisi AWS Bedrock, avec Claude Haiku 5.5, le modèle le moins cher de la gamme.
- **Comment** :
  - **Interface neutre** : `generation.py` et `verification.py` ne connaissent plus aucun SDK.
    Chaque client traduit ses propres erreurs ; ajouter un fournisseur ne touche qu'à `llm.py`.
  - **Erreurs temporaires ou de configuration** : seuls les délais dépassés, les coupures réseau,
    les limites de débit (429) et les erreurs serveur (5xx) deviennent des `LLMError`, qui
    déclenchent un nouvel essai. Une clé invalide ou un modèle inconnu remonte tel quel.
    **Changement de comportement** : avant, toute erreur de l'API était réessayée puis comptée
    comme un rejet ; une mauvaise configuration pouvait fausser un banc entier sans message clair.
  - **Sortie JSON garantie** : avec Claude, le schéma est imposé par les « structured outputs »
    (`output_config.format`), donc le JSON est toujours valide et complet. Les citations restent
    vérifiées en Python, comme avant. Le client OpenAI garde le simple mode JSON : les petits
    modèles locaux gèrent mal un schéma complet, et le banc qwen reste ainsi comparable.
  - **Réflexion (« thinking »)** : Claude Haiku 5.5 réfléchit avant de répondre, et ces tokens
    comptent dans `max_tokens`. 8 000 tokens de marge sont ajoutés pour que le JSON ne soit jamais
    coupé ; seuls les tokens réellement produits sont facturés. Le niveau d'effort se règle avec
    `LLM_EFFORT` (`medium` par défaut).
  - Pas de `temperature` envoyée : ce paramètre n'est plus accepté par les modèles Claude récents.
  - SDK Anthropic plutôt que l'API Converse de boto3 : même interface que l'API directe
    d'Anthropic, et les « structured outputs » sont disponibles sur Bedrock.
- **Vérification** : `pytest` → 160 tests passés ; `ruff check` et `ruff format --check` OK ;
  `docker compose config` valide. Pas encore d'appel réel à Bedrock : le compte AWS reste à créer.

### 2026-10-26 — Dockerisation complète : API, ingestion et schéma reproductibles
- **Quoi** :
  - `Dockerfile` (nouveau) : une étape commune `base` (Python 3.14, PyTorch CPU, package
    `finsight`, modèle MiniLM intégré à l'image) et deux cibles : `api` (uvicorn, utilisateur
    non root, healthcheck sur `/api/health`, port lu dans `$PORT`) et `ingest` (dépendances
    d'ingestion + `scripts/`) ;
  - `docker-compose.yml` : service `api` ; service `ingest` (profil `ingest`, lancé à la
    demande) ; `sql/` monté dans `/docker-entrypoint-initdb.d` ; healthcheck de la base en TCP ;
    variables de connexion partagées (`x-database`) ; suppression des `container_name` fixes ;
  - `.dockerignore` (nouveau) ; `scripts/ingest.sh` (nouveau) ;
  - `sql/001_init.sql` : `CREATE EXTENSION vector` ajouté en tête ; BOM UTF-8 retirés de
    `002` et `003` ;
  - `scripts/download_historical_10k.py` télécharge aussi 2025 ; `scripts/download_sec.py` ne
    télécharge plus « le dernier 10-K » ; `scripts/parse_10k_structured.py` lit `10k_2025.html`
    et prend la provenance de toutes les années dans `filings_manifest.json`.
- **Pourquoi** : sur un clone neuf, il était impossible d'obtenir une démo qui marche. Le schéma
  n'était pas appliqué, et il aurait échoué : `001` utilise le type `VECTOR` alors que l'extension
  n'était créée que dans `003`. La base restait vide, faute d'ingestion dans Docker. Enfin,
  `download_sec.py` enregistrait le **dernier** 10-K comme celui de 2025 : dès la publication du
  10-K 2026 d'Apple (fin octobre 2026), la reconstruction aurait chargé le mauvais rapport sans
  erreur.
- **Comment** :
  - Une seule image de base pour `api` et `ingest` : même version de Python, de PyTorch et du
    modèle d'embedding, donc les mêmes vecteurs aux arrondis près.
  - L'ingestion est un service à lancer une fois (`docker compose run --rm ingest`), pas une
    dépendance de l'API : sinon chaque `docker compose up` retéléchargerait les fichiers SEC.
  - Je reconstruis les données depuis la source publique plutôt que de fournir une sauvegarde
    de la base : la chaîne complète reste vérifiable, et rien de volumineux n'est versionné.
  - Healthcheck en TCP : pendant l'initialisation, Postgres n'écoute que sur son socket local ;
    `pg_isready` sur le socket le déclarerait prêt avant la fin de la création du schéma.
  - L'année 2025 est choisie comme 2023 et 2024 : le premier dépôt 10-K dont la date de clôture
    est dans l'année.
  - Ollama reste sur la machine hôte (`host.docker.internal`) : sur CPU, un conteneur n'apporte
    rien et la taille du modèle alourdirait l'installation.
  - Image : environ 2,3 Go, dont l'essentiel est PyTorch.
- **Vérification** :
  - projet Compose séparé (`finsight-fresh`) avec une base vide et un dossier de données vide :
    le schéma est créé au démarrage, puis les étapes 2 à 7 de l'ingestion passent (15 faits XBRL,
    368 + 371 + 376 extraits) et `/api/health` répond `ok` ;
  - une vraie question envoyée à l'API de ce projet atteint Ollama sur l'hôte et reçoit une
    réponse complète (chiffres XBRL et deux affirmations vérifiées), en 20 minutes sur CPU ;
  - les extraits produits dans Docker ont le même contenu que les fichiers locaux (seules les fins
    de ligne diffèrent) ; écart maximal entre les embeddings : 2·10⁻⁷ ;
  - l'étape 1 (`download_sec.py`) reste à tester dans Docker avec un `SEC_USER_AGENT` ; les
    fichiers SEC déjà téléchargés ont été copiés à la place ;
  - `pytest` → 153 tests passés ; `ruff check` et `ruff format --check` OK ;
    `docker compose config` valide.

### 2026-10-26 — Dossier `build/` exclu de git
- **Quoi** : `.gitignore` : ajout de `build/`.
- **Pourquoi** : un `pip install .` (installation non éditable) crée un dossier `build/` à la
  racine. Ce sont des fichiers générés, qui ne doivent pas être commités par erreur.
- **Vérification** : `git status` ne liste plus `build/`.

### 2026-10-23 — Première évaluation complète du vérificateur (qwen2.5:3b)
- **Quoi** : `eval/results/verifier_qwen2.5-3b.json`.
- **Pourquoi** : mesurer combien d'hallucinations le vérificateur laisse passer, et combien
  d'affirmations vraies il perd, avant de comparer avec d'autres modèles.
- **Comment** : `finsight-eval verifier` sur les 64 affirmations, avec Ollama sur CPU. La mesure
  a duré environ 2 h, dont environ 17 min de mise en veille du PC.
- **Résultat** : 6 % de faux acceptés (2/34), 33 % de faux rejets (10/30), 18 échecs de format.
  Analyse détaillée en 1.5.

### 2026-10-23 — Évaluation du vérificateur : reprise après interruption et temps restant
- **Quoi** :
  - `src/finsight/verifier_eval.py` : sauvegarde de chaque affirmation dans
    `eval/results/verifier_<nom>.partial.jsonl`, reprise automatique, temps restant estimé ;
  - 2 tests dans `tests/test_verifier_eval.py` ;
  - `.gitignore` : les fichiers `.partial.jsonl` ne sont pas versionnés.
- **Pourquoi** : une évaluation complète dure plus d'une heure sur CPU. Avant, les résultats
  n'étaient écrits qu'à la fin : un arrêt du PC ou du terminal faisait tout perdre. Et rien
  n'indiquait combien de temps il restait.
- **Comment** :
  - chaque ligne est écrite et vidée sur disque (`flush`) dès qu'une affirmation est vérifiée ;
  - au lancement suivant avec le même nom, les affirmations déjà présentes dans le fichier
    partiel sont sautées. Le fichier partiel est supprimé quand le résultat final est écrit ;
  - la ligne de progression `[i/n]` affiche `~X min left`, calculé sur la durée moyenne des
    affirmations de la session en cours.
- **Vérification** : `pytest` → 153 tests passés ; `ruff check` et `ruff format --check` OK.

### 2026-10-23 — Mesure de la qualité de la recherche : `finsight-eval`
- **Quoi** :
  - `src/finsight/evaluation.py` : métriques, `DenseRetriever`, rapport et commande `main` ;
  - `src/finsight/retrieval.py` : `encode_query` extrait de `embed_query`, sans changement de
    comportement ;
  - `pyproject.toml` : nouvelle commande `finsight-eval` ;
  - 3 tests dans `tests/test_evaluation.py` ;
  - `eval/results/retrieval_dense-minilm.json` : premier résultat sauvegardé.
- **Pourquoi** : obtenir un **chiffre de référence** de la recherche actuelle, avant toute
  amélioration. Sans point de départ mesuré, on ne peut pas prouver qu'un changement aide.
- **Comment** :
  - **Hit@k** et **MRR**, au global et par langue, catégorie et année ;
  - recherche **hors ligne** : similarité cosinus exacte sur les fichiers `.npy`. Ce sont les
    mêmes vecteurs que dans Postgres (le hash est vérifié à l'insertion), donc pas besoin de
    Docker et le résultat est reproductible. La production utilise un index HNSW approximatif ;
    l'écart attendu est négligeable ;
  - commande séparée plutôt que sous-commande de `finsight`, pour ne pas casser
    `finsight "question"` ;
  - le type `Retriever` (une fonction question, année, profondeur → textes) permet de brancher
    d'autres variantes (BM25, hybride, reranking) sur le même banc.
- **Vérification** : `pytest` → 144 tests passés ; `ruff` OK ; `finsight-eval` exécuté.

### 2026-10-23 — Jeu d'évaluation
- **Quoi** : `eval/questions.jsonl` (91 questions), `src/finsight/evaluation.py`,
  `tests/test_evaluation.py`.
- **Pourquoi** : sans mesure, impossible de prouver la qualité du système ni de comparer deux
  approches (recherche hybride, reranking, autre modèle d'embedding).
- **Comment** :
  - Chaque question a une année, une langue, une catégorie et des **phrases-preuves** copiées
    mot pour mot du 10-K. Un extrait est jugé pertinent s'il contient l'une d'elles (après
    normalisation : casse, apostrophes typographiques, espaces).
  - Je n'utilise **pas les identifiants de chunks**, parce qu'ils changent dès qu'on modifie le
    découpage. Avec des phrases, le même jeu reste valable pour comparer des stratégies de
    découpage.
  - Répartition : 72 questions sur 2025, 11 sur 2024, 8 sur 2023. 16 questions en **français**
    (reformulations des questions anglaises) mesurent le point faible du modèle MiniLM, entraîné
    surtout en anglais. 5 questions **sans réponse** dans les 10-K (conseil d'investissement,
    autre entreprise…) sont réservées à l'évaluation de l'abstention.
  - Un test vérifie que chaque phrase-preuve existe bien dans les chunks de son année (ignoré si
    `data/processed` est absent, car ce dossier n'est pas versionné).
- **Vérification** : `pytest` → 141 tests passés ; `ruff check` et `ruff format --check` OK.

### 2026-10-22 — Restructuration en package `finsight` testé
- **Quoi** : création de `src/finsight/` (config, llm, facts, retrieval, generation,
  verification, metrics, pipeline, cli) avec modèles pydantic communs ; suite pytest (43 tests,
  faux LLM) ; `pyproject.toml` remplace `requirements.txt` ; suppression du dossier `archive/`
  et des anciens scripts.
- **Pourquoi** : les scripts accumulés étaient difficiles à maintenir et à tester.
- **Comment** : les chiffres sont calculés une seule fois dans `metrics.py` et le LLM ne voit
  que les extraits 10-K ; suppression du repli LLM pour les chiffres (le filtre numérique le
  rejetait toujours) ; les demandes de conseil sont signalées au lieu d'être traitées ;
  vérification en parallèle. L'ancien code reste consultable dans l'historique git.

### 2026-10-22 — Vérification des affirmations par citations exactes
- **Pourquoi** : le LLM pouvait affirmer des choses non présentes dans les sources.
- **Comment** : chaque affirmation est découpée en propositions atomiques et n'est gardée que
  si chacune est SUPPORTED par une citation mot pour mot de l'extrait SEC cité.

### 2026-10-22 — Commit initial
- Structure initiale du pipeline FinSight (ingestion SEC, Postgres/pgvector, RAG).

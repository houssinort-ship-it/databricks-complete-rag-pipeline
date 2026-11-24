# Databricks Notebook source
# MAGIC %md
# MAGIC # Atelier RAG — PDF → Chunks → Embeddings → Vector Search → Llama 3.3 70B
# MAGIC
# MAGIC Notebook original de l'atelier, reconstitué cellule par cellule depuis le
# MAGIC document source *RAG dans Databricks*.
# MAGIC
# MAGIC **Prérequis** (voir `docs/DATABRICKS_SETUP.md`)
# MAGIC * un workspace Databricks avec Unity Catalog et un Volume `workspace.default.rag_docs`
# MAGIC * les 3 PDF de démonstration déposés dans ce Volume
# MAGIC * un endpoint de serving pour l'embedding et un pour le LLM
# MAGIC
# MAGIC Ce fichier est prévu pour être importé dans Databricks (format « source » :
# MAGIC les cellules sont séparées par `# COMMAND ----------`).
# MAGIC
# MAGIC > Aucun secret n'est écrit ici : l'authentification se fait via le contexte
# MAGIC > du workspace (`WorkspaceClient()`), et tous les paramètres peuvent être
# MAGIC > surchargés par des variables d'environnement ou des widgets.

# COMMAND ----------

# MAGIC %pip install -q pypdf langchain-text-splitters
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import os

from databricks.sdk import WorkspaceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

w = WorkspaceClient()
api = w.api_client

# --- Paramètres de l'atelier ------------------------------------------------
CATALOG = os.getenv("CATALOG", "workspace")
SCHEMA = os.getenv("SCHEMA", "default")
VOLUME = os.getenv("VOLUME", "rag_docs")

DOCS_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.rag_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_chunks_index"
VS_ENDPOINT_NAME = "rag_endpoint"  # ignoré si un endpoint existe déjà

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Modèle d'embedding hébergé par Databricks.
# databricks-qwen3-embedding-0-6b est multilingue (français, arabe, anglais...).
# S'il n'est pas disponible sur votre compte, le notebook bascule sur
# databricks-gte-large-en.
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-qwen3-embedding-0-6b")
FALLBACK_EMBEDDING_ENDPOINT = "databricks-gte-large-en"

# LLM de génération (vérifiez le nom exact à l'étape 2)
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

# Mettre True pour supprimer et recréer l'index (ex. après un changement de
# modèle d'embedding)
RECREATE_INDEX = False

print("Catalogue :", CATALOG, "| schéma :", SCHEMA, "| volume :", VOLUME)
print("Table de chunks :", CHUNKS_TABLE)
print("Index :", INDEX_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vérifier les endpoints disponibles
# MAGIC
# MAGIC On liste les endpoints de serving pour confirmer le nom du LLM et vérifier
# MAGIC que le modèle d'embedding est bien accessible. Si
# MAGIC `databricks-qwen3-embedding-0-6b` n'existe pas sur le compte, on retombe
# MAGIC automatiquement sur `databricks-gte-large-en`.

# COMMAND ----------

serving = sorted(ep.name for ep in w.serving_endpoints.list())
print(f"{len(serving)} endpoints disponibles :\n")
for name in serving:
    print(" •", name)

if EMBEDDING_ENDPOINT not in serving:
    print(
        f"\n⚠️  {EMBEDDING_ENDPOINT} indisponible : bascule sur "
        f"{FALLBACK_EMBEDDING_ENDPOINT}"
    )
    EMBEDDING_ENDPOINT = FALLBACK_EMBEDDING_ENDPOINT

if LLM_ENDPOINT not in serving:
    raise ValueError(
        f"{LLM_ENDPOINT} est introuvable. Copiez le nom exact depuis la liste "
        "ci-dessus et mettez à jour LLM_ENDPOINT."
    )

print("\nEndpoint d'embedding utilisé :", EMBEDDING_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingestion des PDF → chunks
# MAGIC
# MAGIC Pour chaque PDF du Volume : extraction page par page avec `pypdf`, nettoyage
# MAGIC des espaces, découpage récursif (1000 caractères, 150 de recouvrement), et
# MAGIC identifiant stable `{fichier}_p{page}_c{index}`.

# COMMAND ----------

import os
import re

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Réduit les espaces multiples sans casser les retours à la ligne."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    keep_separator="end",
    length_function=len,
)

rows = []
pdf_files = sorted(f for f in os.listdir(DOCS_PATH) if f.lower().endswith(".pdf"))
print(f"{len(pdf_files)} PDF trouvés dans {DOCS_PATH}")

for fname in pdf_files:
    reader = PdfReader(f"{DOCS_PATH}/{fname}")
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue  # page vide ou scannée (sans OCR)
        for i, chunk in enumerate(splitter.split_text(text)):
            rows.append(
                {
                    "id": f"{fname}_p{page_num}_c{i}",
                    "source": fname,
                    "page": page_num,
                    "content": chunk,
                }
            )

print(f"{len(rows)} chunks créés")
if rows:
    print("Exemple :", rows[0]["content"][:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Table Delta `rag_chunks`
# MAGIC
# MAGIC Écriture avec un schéma explicite, puis activation du Change Data Feed :
# MAGIC c'est ce qui permet à l'index Delta Sync de suivre les mises à jour.

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("page", IntegerType(), True),
        StructField("content", StringType(), True),
    ]
)

(
    spark.createDataFrame(rows, schema)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CHUNKS_TABLE)
)

spark.sql(
    f"ALTER TABLE {CHUNKS_TABLE} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

display(spark.table(CHUNKS_TABLE).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Endpoint Vector Search
# MAGIC
# MAGIC On réutilise un endpoint existant s'il y en a un, sinon on crée
# MAGIC `rag_endpoint` (type `STANDARD`) puis on attend l'état `ONLINE`.

# COMMAND ----------

import time

endpoints = api.do("GET", "/api/2.0/vector-search/endpoints").get("endpoints", [])

if endpoints:
    endpoint_name = endpoints[0]["name"]
    print(f"Endpoint existant réutilisé : {endpoint_name}")
else:
    api.do(
        "POST",
        "/api/2.0/vector-search/endpoints",
        body={"name": VS_ENDPOINT_NAME, "endpoint_type": "STANDARD"},
    )
    endpoint_name = VS_ENDPOINT_NAME
    print(f"Endpoint créé : {endpoint_name}")

while True:
    ep = api.do("GET", f"/api/2.0/vector-search/endpoints/{endpoint_name}")
    state = ep.get("endpoint_status", {}).get("state")
    print("État :", state)
    if state == "ONLINE":
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Index Delta Sync sur `rag_chunks`
# MAGIC
# MAGIC * `primary_key = "id"` (clé primaire de la table)
# MAGIC * `embedding_source_columns` : l'endpoint calcule lui-même les embeddings de
# MAGIC   la colonne `content` — aucun calcul d'embedding côté notebook
# MAGIC * `pipeline_type = "TRIGGERED"` : synchronisation déclenchée manuellement
# MAGIC
# MAGIC Si l'index existe déjà, on le synchronise ; si `RECREATE_INDEX = True`, on le
# MAGIC supprime d'abord (utile quand on change de modèle d'embedding).

# COMMAND ----------


def get_index():
    """Retourne la description de l'index, ou None s'il n'existe pas."""
    try:
        return api.do("GET", f"/api/2.0/vector-search/indexes/{INDEX_NAME}")
    except Exception:
        return None


index = get_index()

if index and RECREATE_INDEX:
    api.do("DELETE", f"/api/2.0/vector-search/indexes/{INDEX_NAME}")
    print("Ancien index supprimé")
    while get_index():
        time.sleep(10)
    index = None

if index:
    api.do("POST", f"/api/2.0/vector-search/indexes/{INDEX_NAME}/sync")
    print("Index existant : synchronisation lancée")
else:
    api.do(
        "POST",
        "/api/2.0/vector-search/indexes",
        body={
            "name": INDEX_NAME,
            "endpoint_name": VS_ENDPOINT_NAME,
            "primary_key": "id",
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": CHUNKS_TABLE,
                "pipeline_type": "TRIGGERED",
                "embedding_source_columns": [
                    {"name": "content", "embedding_model_endpoint_name": EMBEDDING_ENDPOINT}
                ],
            },
        },
    )
    print("Index en cours de création...")

# Attente de l'indexation
time.sleep(15)
while True:
    status = get_index().get("status", {})
    print(
        "Indexation :",
        status.get("detailed_state"),
        "| lignes indexées :",
        status.get("indexed_row_count"),
    )
    if status.get("ready") and status.get("indexed_row_count", 0) > 0:
        print("Index prêt ✅")
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test de recherche
# MAGIC
# MAGIC La requête ne demande que `id, source, page, content` (pas de `score` : le
# MAGIC notebook affiche donc `0.000` par défaut, c'est le comportement d'origine).

# COMMAND ----------


def retrieve(question: str, k: int = 4):
    """Recherche sémantique dans l'index et retourne les passages."""
    res = api.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{INDEX_NAME}/query",
        body={
            "columns": ["id", "source", "page", "content"],
            "query_text": question,
            "num_results": k,
        },
    )
    cols = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
    data = res.get("result", {}).get("data_array", []) or []
    return [dict(zip(cols, row)) for row in data]


for hit in retrieve("Quels sont les avantages du produit AquaPure X3 ?"):
    print(f"[{hit.get('score', 0):.3f}] {hit['source']} p.{hit['page']}")
    print(hit["content"][:300], "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. RAG : génération avec Llama 3.3 70B
# MAGIC
# MAGIC Le contexte est assemblé au format `[fichier, page]` + texte. Le prompt
# MAGIC système impose de répondre uniquement à partir du contexte, de dire « je ne
# MAGIC sais pas » sinon, de citer les sources et de répondre dans la langue de la
# MAGIC question.

# COMMAND ----------

import time

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni. "
    "Si la réponse n'est pas dans le contexte, dis que tu ne sais pas. "
    "Cite tes sources sous la forme [fichier, page]. Réponds dans la langue de la question."
)


def ask(question: str, k: int = 4, temperature: float = 0.1):
    """Retrieval + génération ; retourne (réponse, passages utilisés)."""
    hits = retrieve(question, k)

    context = "\n\n".join(
        f"[{h['source']}, page {h['page']}]\n{h['content']}" for h in hits
    )

    resp = api.do(
        "POST",
        f"/serving-endpoints/{LLM_ENDPOINT}/invocations",
        body={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"},
            ],
            "temperature": temperature,
            "max_tokens": 800,
        },
    )
    return resp["choices"][0]["message"]["content"], hits


question = "Que faire si le voyant orange clignote sur l'AquaPure X3 ?"
answer, hits = ask(question)

print("=" * 80)
print(answer)
print("=" * 80)
print("\nSources :", sorted({(h["source"], h["page"]) for h in hits}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variante : répondre sur le règlement intérieur

# COMMAND ----------

question = "Quels sont les horaires de travail chez Novatech ?"
answer, hits = ask(question)

print(answer)
print("\nSources :", sorted({(h["source"], h["page"]) for h in hits}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nettoyage (optionnel)
# MAGIC
# MAGIC Supprimer l'index et l'endpoint pour éviter de consommer du crédit DBU :
# MAGIC
# MAGIC ```python
# MAGIC api.do("DELETE", f"/api/2.0/vector-search/indexes/{INDEX_NAME}")
# MAGIC api.do("DELETE", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT_NAME}")
# MAGIC ```

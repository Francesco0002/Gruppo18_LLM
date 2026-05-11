# Gruppo18_LLM

Progetto del corso di *Natural Language Processing and Large Language
Models*: chatbot RAG sulle informazioni ufficiali del DIEM.

## Struttura

```
src/             Codice della pipeline
  discover.py          Orchestratore discovery URL (fase 1)
  discovery_fetch.py   HTTP, robots.txt e sitemap
  discovery_io.py      Config, checkpoint, JSONL e salvataggio HTML
  discovery_models.py  Dataclass condivise della discovery
  html_utils.py        Estrazione link e canonical dagli HTML
  scrape.py            Conversione HTML raw in Markdown
  extract_pdf.py       Download PDF e conversione in Markdown
  ingest.py            Orchestratore pipeline + duplicati + stats
  url_filters.py       Filtri URL (scope DIEM, traversal vs index)
  legacy/              Prototipi non più usati
data/            Dati locali; gli output del crawl sono gitignored
notebooks/       Esperimenti
report/          Documentazione finale
config.example.yaml  Template di configurazione
.env.example         Template variabili d'ambiente
Makefile             Shortcut per i comandi più comuni
```

## Requisiti

- Python **3.11+**
- Connessione a internet per la discovery (raggiunge `*.unisa.it`)
- (Opzionale) `make` su macOS / Linux per usare gli shortcut

## Setup

### 1. Clone e virtual environment

**macOS / Linux**
```bash
git clone <repo-url> Gruppo18_LLM
cd Gruppo18_LLM
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
git clone <repo-url> Gruppo18_LLM
cd Gruppo18_LLM
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Dipendenze

```bash
pip install -r requirements.txt
```

(o `make install` su macOS/Linux)

### 3. Configurazione

Il file `config.yaml` non è in git: ognuno parte dal template.

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

Modifica `config.yaml` per cambiare limiti, domini ammessi e scope.
Il file è documentato sezione per sezione.
Per una descrizione completa del flusso dati, vedi `docs/pipeline.md`.

## Uso

### Discovery (fase 1)

Crawl completo con i parametri di `config.yaml`:
```bash
python src/discover.py
```

Su macOS/Linux è disponibile anche `make discover`.

### Scraping HTML e PDF

```bash
python src/scrape.py
python src/extract_pdf.py
```

Su macOS/Linux:

```bash
make scrape
make extract-pdf
```

### Pipeline ingest

Esegue discovery, scraping HTML, estrazione PDF, marcatura dei duplicati e
generazione di `data/processed/stats.json`:

```bash
python src/ingest.py
```

Se discovery/scraping/PDF sono già stati eseguiti e vuoi solo rigenerare
marcatura duplicati e statistiche:

```bash
python src/ingest.py --stats-only
```

## Output della discovery

| Path | Contenuto |
|---|---|
| `data/discovered_urls.jsonl` | Manifest, una riga per URL processato |
| `data/raw_html/<sh>/<hash>.html` | HTML grezzo dei documenti indicizzabili |
| `data/checkpoint.json` | Stato BFS per riprendere un run interrotto |
| `data/processed/markdown/<sh>/<hash>.md` | Markdown fit da HTML/PDF |
| `data/processed/markdown_raw/<sh>/<hash>.md` | Markdown raw campionato dagli HTML |
| `data/processed/manifest.jsonl` | Manifest dei documenti processati |
| `data/processed/stats.json` | Statistiche finali del corpus e duplicati |

Ogni record di `discovered_urls.jsonl` contiene:
`requested_url`, `final_url`, `canonical_url`, `document_url`,
`mime`, `content_type`, `status`, `indexable`, `raw_path`, `depth`,
`discovered_from`, `domain`, `links_found`.

Tutti questi file sono gitignored.

### Status possibili

| `status` | Significato |
|---|---|
| `ok` | HTML indicizzabile, salvato su disco |
| `not_indexable` | HTML in scope ma escluso dall'indice (es. paginazione) |
| `duplicate_canonical` | Stessa pagina di un altro URL (canonical condiviso) |
| `duplicate_redirect` | Redirect verso una pagina già processata |
| `redirected_out_of_scope` | Redirect porta fuori scope DIEM |
| `non_html` | Risorsa scaricata ma non HTML (e non PDF) |
| `pending_download` | PDF da scaricare nello step successivo |
| `too_large` | HTML oltre `max_html_bytes`, saltato |
| `failed` | Errore HTTP/network |
| `robots_denied` | `robots.txt` blocca l'URL |

## Pipeline completa (roadmap)

La discovery è solo lo step 1. Gli step successivi (in arrivo):

1. **discover** ✅ — popola `discovered_urls.jsonl` e `raw_html/`.
2. **scrape** — converte gli HTML in markdown pulito.
3. **extract_pdf** — scarica i PDF marcati `pending_download` ed estrae il testo.
4. **ingest** ✅ — orchestratore + marcatura duplicati + statistiche.

## Troubleshooting

- *FileNotFoundError: configurazione non trovata*: hai copiato
  `config.example.yaml` in `config.yaml`?
- *`pip install` fallisce su Windows con errore di encoding*: assicurarsi
  che `requirements.txt` sia salvato in UTF-8 senza BOM.
- *Il run sembra fermo*: il rate limit è di 2 req/s per dominio. Aumentare
  `crawler.rate_limit_per_domain_rps` in `config.yaml` se necessario.
- *Voglio ripartire da zero*: `make clean && make discover`.

## Note

Il virtual environment, i dati locali, gli indici, gli output del crawl
e le API key non devono finire in git. Vedi `.gitignore`.

# Pipeline del Corpus DIEM

Questo documento descrive la pipeline che prepara i documenti per la futura
fase di indexing del chatbot RAG.

## Obiettivo

La pipeline trasforma pagine e PDF ufficiali DIEM in file Markdown locali,
misurabili e pronti per chunking, embedding e indicizzazione vettoriale.

```text
discover.py
  -> discovered_urls.jsonl + raw_html/
scrape.py
  -> raw markdown HTML + markdown pulito + manifest.jsonl
extract_pdf.py
  -> raw markdown PDF + markdown pulito + manifest.jsonl
ingest.py
  -> marcatura duplicati + stats.json corrente + storico run
chunking.py
  -> chunks.jsonl + stats.json dei chunk
vector_store.py
  -> embedding dei chunk + Chroma vector store
retrieval.py
  -> BM25 + dense retrieval + fusione RRF dei risultati
rag_chain.py
  -> prompt RAG + generazione risposta tramite API Groq
chatbot_cli.py
  -> interfaccia CLI per interrogare il chatbot
app.py
  -> interfaccia grafica Chainlit
```

## 1. Discovery

Comando:

```bash
python src/discover.py
```

Responsabilità:

- legge i seed da `data/urls.txt`;
- opzionalmente espande i seed tramite sitemap;
- visita URL HTML con BFS;
- applica filtri di scope e query rumorose;
- rispetta `robots.txt` se `respect_robots_txt: true`;
- salva HTML grezzo in `data/raw_html/<sh>/<hash>.html`;
- scrive `data/discovered_urls.jsonl`;
- salva `data/checkpoint.json` per riprendere un crawl interrotto;
- salva `data/discovery_state.json` per mantenere tra run la frontier non ancora
  visitata e la memoria degli URL/documenti già noti;
- applica `refresh_after_days` per ricontrollare periodicamente URL già visti
  senza consumare sempre il budget sui medesimi URL recenti.

`checkpoint.json` è uno stato intra-run: se il processo si interrompe, permette
di riprendere la stessa esecuzione. Salva queue, contatori del run e solo il
delta di memoria rispetto allo stato persistente iniziale (`new_known_urls` e
`new_known_documents`). `discovery_state.json` è invece uno stato inter-run:
conserva la frontier residua e la memoria cumulativa necessaria a proseguire la
copertura in run successive. I limiti `max_total_urls` e `per_domain_limits`
restano budget del singolo run, non contatori globali.
`discovered_urls.jsonl` resta lo snapshot degli URL prodotti dalla run corrente:
serve come input immediato agli step di scraping HTML ed estrazione PDF, mentre
la memoria cumulativa vive in `discovery_state.json`.

### Perché serve `expansion_backlog`

Un URL può essere:

- **visitato**: la pagina è stata scaricata;
- **espanso**: oltre a scaricarla, il crawler ha anche seguito i suoi link per
  popolare la depth successiva.

Queste due cose non coincidono sempre. Esempio:

1. una run usa `max_depth: 1`;
2. il crawler visita una pagina a depth 1;
3. quella pagina viene scaricata, ma i suoi link non vengono seguiti, perché
   produrrebbero URL a depth 2, fuori dal limite della run;
4. se in seguito si passa a `max_depth: 2`, quella stessa pagina va riletta per
   poter scoprire i figli di depth 2.

Prima della modifica, le pagine di bordo venivano considerate solo come
"già visitate". Se si aumentava `max_depth`, il crawler non le riapriva perché
erano ancora dentro `refresh_after_days`; di conseguenza una run a depth 2
poteva terminare senza trovare nessun nuovo URL, anche se la depth 2 non era
stata davvero esplorata.

Per evitare questo problema, oltre alla frontier non ancora visitata,
`discovery_state.json` conserva `expansion_backlog`: l'elenco delle pagine HTML
già scaricate ma non ancora espanse abbastanza rispetto a una futura depth più
alta. Quando una run successiva aumenta `max_depth`, quelle pagine vengono
riaccodate con una **riespansione mirata**: il crawler le rilegge per aprire il
nuovo livello, senza dover fare `make clean` e senza attendere
`refresh_after_days`.

### Concetti da ricordare

- `frontier`: URL già scoperti ma non ancora visitati.
- `espansione`: visita di una pagina HTML e lettura dei suoi link.
- `expansion_backlog`: pagine già visitate che non potevano ancora essere
  espanse perché la run si fermava alla depth corrente.
- `reexpansion`: nuova visita di quelle pagine quando aumenti `max_depth`, così
  possono generare i figli del livello successivo.

Servono a distinguere due situazioni diverse:

- manca ancora lavoro **dentro** una depth;
- la depth è già stata visitata, ma va riaperta per scoprire la depth dopo.

Per i docenti, la discovery mantiene anche `allowed_teacher_profiles`: la
whitelist dei profili `docenti.unisa.it` autorizzati. Un profilo entra in questa
lista solo se viene scoperto dalla pagina `dipartimento/personale` o dal ponte
`rubrica.unisa.it/persone?...` raggiunto da quella pagina. Dopo l'ingresso, il
crawler può navigare solo le sottopagine dello stesso profilo; link verso altri
docenti non autorizzati restano fuori scope.

Le pagine `rubrica.unisa.it/persone?matricola=...` sono indicizzabili solo se
scoperte da `www.diem.unisa.it/dipartimento/personale`, così i contatti dei
professori DIEM entrano nel manifest senza aprire la rubrica a persone esterne.
Il dominio dei consigli didattici (`cd.unisa.it`) è attraversabile solo per i
percorsi corso DIEM configurati, inclusi `commissioni` e `delegati`; i contatti
elencati da quelle pagine non aprono invece la rubrica a persone esterne.

Le query tecniche restano bloccate di default. L'unica eccezione per `archive`
è una allowlist esplicita di pagine informative:

- `/home/eventi?archive=1`
- `/home/eventi?archive=2`
- `/home/news?archive=1`

In questo modo gli archivi utili entrano nel corpus senza rendere attraversabile
qualunque variante parametrica del sito.

Sui progetti finanziati la regola è più selettiva:

- `/ricerca/progetti-finanziati?progetto=...` è attraversabile perché apre il
  dettaglio informativo di un singolo progetto;
- `tip` e `stato` restano bloccati perché producono viste filtrate della stessa
  lista, utili alla navigazione ma ridondanti per il corpus RAG.

Il checkpoint usa `status="in_progress"` durante il run e
`status="completed"` a chiusura. Solo il checkpoint finale riporta anche
`stop_reason`, così si distingue tra arresto per `max_total_urls`, coda
esaurita (`queue_exhausted`) e coda senza URL più eleggibili
(`no_eligible_urls`).

Quando una run non ha URL eleggibili, le statistiche della discovery riportano
anche `skipped_by_reason`: ad esempio `recently_known` segnala URL già noti e
ancora dentro la finestra `refresh_after_days`, quindi esclusi senza fetch nella
run corrente. Gli URL bloccati da `domain_limit`, invece, non vengono più
eliminati dalla frontiera: il limite vale per il run corrente, quindi quegli URL
restano pendenti e potranno essere visitati in una run successiva.

Anche nella costruzione iniziale della coda, seed e sitemap già noti e ancora
recenti non vengono riaccodati inutilmente se non sono dovuti al refresh. In
questo modo una run che riprende da frontier o da `expansion_backlog` non si
porta dietro seed superflui a depth 0 che renderebbero meno leggibile il report
di copertura.

Le statistiche in `data/processed/stats.json` sono organizzate in pochi blocchi:

- `summary`: lettura immediata del run e dello stato corrente del corpus;
- `discovery.run`: cosa è successo nella discovery del run corrente;
- `discovery.found`: quali URL sono stati trovati nel run, per status, tipo e
  dominio;
- `discovery.run_progress`: confronto essenziale tra frontiera prima e dopo il
  run;
- `discovery.remaining_work`: lavoro ancora aperto dopo il run;
- `discovery.coverage`: verdetto di copertura e tabella per depth;
- `discovery.pdfs`: riepilogo dei PDF trovati, bloccati o ammessi dalla regola
  dedicata;
- `extraction`: esito dello scraping HTML e dell'estrazione PDF nel solo run
  corrente;
- `processed`: stato cumulativo del corpus già processato.

Dentro i blocchi più densi, i campi sono raggruppati per scopo invece di stare
tutti sullo stesso piano:

- `discovery.summary`: dimensione del run, copertura e lavoro residuo;
- `discovery.pdfs.summary`: bilancio dei PDF ammessi, bloccati e da rivedere;
- `extraction.pdf.summary`, `volume`, `performance`, `errors`: esito operativo
  della sola fase PDF;
- `processed.summary`, `distribution`, `content`, `errors`: stato corrente del
  corpus, distribuzioni, statistiche testuali e fallimenti PDF ancora attuali.

La tabella `discovery.coverage.depths` concentra in un solo punto le metriche più
utili per ogni livello:

- `html_recorded_at_depth`: record HTML scritti nel run con quella depth;
- `pdf_recorded_at_depth`: record PDF scritti nel run con quella depth, cioè
  figli della depth precedente quando arrivano da link HTML;
- `html_visited_at_depth`: pagine HTML realmente visitate a quella depth;
- `pending_new`: URL nuovi ancora da visitare;
- `pending_reexpansion`: pagine già viste da riespandere per aprire il livello
  successivo;
- `pending_from_lower_depths`: lavoro nelle depth inferiori che può ancora
  generare figli qui;
- `complete`: nessun nuovo URL resta da visitare a quella depth;
- `visited_complete`: la depth è stata visitata davvero fino in fondo.

La frontier usa anche una fairness esplicita tra rami: l'ordine primario resta
sempre la `depth`, ogni URL conserva `origin_seed` e, solo a parità di livello,
la coda alterna i diversi seed in round-robin. Così la semantica BFS resta
corretta anche tra run consecutive, ma con budget limitato un solo seed molto
prolifico non può occupare quasi tutta la capacità del livello. La frontier
persistita mantiene precedenza rispetto al nuovo bootstrap solo quando gli item
hanno la stessa depth. `discovery.coverage.by_seed` mostra quanti URL sono stati
trovati, visitati e restano pendenti per ciascun seed e depth.

`discovery.remaining_work` mantiene solo il riepilogo necessario:

- `new_urls` / `new_urls_by_depth`: URL nuovi ancora da visitare;
- `reexpansions` / `reexpansions_by_depth`: riespansioni ancora pendenti;
- `future_expansion_backlog_by_depth`: pagine già viste che diventeranno utili
  solo se aumenterai ancora `max_depth`.

`failed_by_kind` distingue i fallimenti tra `http_404`, `timeout`, `http_5xx`,
altri errori HTTP e altri casi. `blocking_reasons` spiega perché il verdetto
generale non è ancora `complete`.

Una depth è completa solo quando:

1. non ha URL propri ancora da visitare;
2. non dipende più da depth inferiori ancora aperte;
3. non restano pagine da riespandere che potrebbero generare nuovi figli.

Il verdetto globale è `complete` solo se tutte le depth configurate soddisfano
queste condizioni, risultano anche `visited_complete` e la run termina con
`queue_exhausted`.

### Come leggere l'andamento tra run

Ogni report storico sotto `data/processed/runs/<run_id>/stats.json` conserva il
riepilogo compatto del run. Per capire se stai davvero chiudendo una depth:

- confronta `discovery.run_progress.new_urls_before_by_depth` con
  `new_urls_after_by_depth`;
- usa `new_urls_before_by_seed_and_depth` e
  `new_urls_after_by_seed_and_depth` quando vuoi verificare che nessun seed stia
  restando sistematicamente indietro;
- guarda in `discovery.coverage.depths` se `pending_new` sta scendendo;
- controlla `pending_reexpansion` se hai appena aumentato `max_depth`;
- usa `future_expansion_backlog_by_depth` per sapere quanto lavoro si attiverà
  soltanto quando andrai ancora più in profondità;
- considera una depth davvero chiusa quando `complete` e `visited_complete`
  sono entrambi `true`.
I PDF restano inclusi nelle statistiche generali, ma non riaprono la BFS HTML:
quando sono linkati da una pagina al limite possono comparire a
`depth = max_depth + 1`.

Per ripartire davvero da zero usa `make clean`: il target elimina raw HTML/PDF,
frontier e checkpoint della discovery, manifest, Markdown prodotti, stats
correnti e storico sotto `data/processed/runs/`. Dopo questo reset i filtri
incrementali non vedono più successi precedenti.

I PDF linkati dagli HTML vengono registrati subito in
`discovered_urls.jsonl`.

La regola ordinaria resta semplice: se `robots.txt` consente il download, il
PDF riceve `status="pending_download"`; altrimenti riceve
`status="robots_denied"`.

Esistono però deroghe PDF dedicate e volutamente strette per non perdere
documenti utili alla RAG. Un PDF bloccato da `robots.txt` viene comunque ammesso
solo se rientra in uno di questi casi:

1. documento stabile da sezione centrale (`didattica`, `dipartimento`,
   `ricerca`, `international`) con keyword come `regolamento`, `guida`,
   `linee-guida`, `manifesto`, `piano-di-studi`;
2. materiale didattico operativo da `didattica`, come `calendario`, `schedule`,
   `ofa`, `requirements`;
3. solo per i PDF con keyword `calendario`, una deroga rescue ancora più stretta
   quando il parent DIEM è una pagina di dettaglio informativa con slug esplicito
   come `calendario-prove-in-itinere` o `appelli-di-recupero`;
4. allegati PDF di focus didattici DIEM (`/didattica/focus?id=...`) solo quando
   il path del PDF contiene lo stesso identificativo del parent;
5. materiale internazionale da `international`, come `accordi` ed `erasmus`;
6. documenti di qualità ed esiti del corso, come `SUA-CDS` e `AlmaLaurea`,
   quando sono linkati da corsi già ammessi nello scope;
7. documento principale di opportunità da `home_bandi`, riconosciuto da segnali
   come `bando`, `call`, `premio`, `borsa`, `concorso`, ma non se è solo un
   allegato accessorio come `graduatoria`, `domanda`, `modello`, `locandina`,
   `faq`, `verbale`, `differimento`, `elenco`, `scorrimento`, `presentazione`,
   `comunicato`, `avviso-proroga`;
8. `decreto` proveniente da una sezione centrale e accompagnato da un contesto
   testuale non generico.

Questo include anche le pagine `corsi.unisa.it` già ammesse dallo scope, quando
appartengono a sezioni centrali come `didattica`. In questo modo la pipeline non
perde le opportunità correnti davvero interrogabili dal chatbot, ma continua a
evitare di scaricare in massa risultati, moduli e allegati debolmente utili.

## 2. Scraping HTML

Comando:

```bash
python src/scrape.py
```

Responsabilità:

- legge gli HTML `status="ok"` e `indexable=true` da `discovered_urls.jsonl`;
- legge il file HTML grezzo già salvato, senza riscaricare la pagina;
- genera Markdown raw con Crawl4AI;
- salva il raw in `data/processed/markdown_raw/<sh>/<hash>.md`;
- pulisce boilerplate conservativo e aggiunge front matter YAML;
- salva il Markdown indicizzabile in `data/processed/markdown/<sh>/<hash>.md`;
- aggiunge record a `data/processed/manifest.jsonl`;
- salta URL HTML già processati negli ultimi 7 giorni.

## 3. Estrazione PDF

Comando:

```bash
python src/extract_pdf.py
```

Responsabilità:

- legge PDF `status="pending_download"` da `discovered_urls.jsonl`;
- scarica i PDF in `data/raw_pdf/<sh>/<hash>.pdf`;
- converte i PDF in Markdown raw con `pymupdf4llm`;
- salva il raw in `data/processed/markdown_raw/<sh>/<hash>.md`;
- normalizza artefatti PDF evidenti e aggiunge front matter YAML;
- riconosce i PDF tabellari solo apparentemente pieni, ma privi di vere righe
  dati, e li conserva nel manifest con `text_extracted=false`,
  `indexable=false` e warning `empty_structured_pdf`;
- salva il Markdown indicizzabile in `data/processed/markdown/<sh>/<hash>.md`;
- aggiunge record a `data/processed/manifest.jsonl`;
- salta PDF già processati negli ultimi 7 giorni.

La fase PDF usa una pipeline sovrapposta:

- scarica più PDF in parallelo rispettando `pdf_max_concurrent_downloads` per
  dominio e `pdf_download_delay_seconds`;
- avvia l'estrazione appena un raw valido è disponibile, senza aspettare la fine
  di tutti i download;
- riusa i raw PDF già presenti e validi quando `pdf_force_reextract=false`;
- salva i PDF con scrittura atomica, valida redirect finali, MIME e signature
  PDF, e isola gli errori per singolo documento.

Durante il run mostra una barra `Estrazione PDF` sul lavoro effettivamente
selezionato. Il contatore avanza quando ogni PDF raggiunge uno stato terminale e
il postfix espone pronti per l'estrazione, successi, fallimenti, file troppo
grandi, download di rete e raw riusati.

I parametri effettivi della fase sono configurabili in `config.yaml`:
`pdf_max_bytes`, `pdf_extraction_workers`, `pdf_max_concurrent_downloads`,
`pdf_download_delay_seconds`, `pdf_skip_recent_days`, `pdf_force_reextract` e
`pdf_extraction`.

Nel blocco `extraction.pdf` degli stats vengono riportati anche i parametri
operativi del run: PDF pronti per l'estrazione, raw riusati, download di rete,
byte scaricati, tempi di download/estrazione, throughput PDF/minuto, backend
dell'executor e breakdown dei fallimenti. In questo modo una run lenta distingue
chiaramente rete, parsing e documenti corrotti.

Il blocco `processed.errors.pdf_failures.by_kind` ricostruisce lo stesso tipo di
breakdown dal manifest corrente, quindi resta disponibile anche quando rigeneri
le statistiche con `--stats-only`. Le categorie HTTP vengono rese esplicite
quando il messaggio contiene lo status, ad esempio `http_400`, `http_404` o
`http_5xx`; gli errori di parsing PDF finiscono in `extract_failed`.

Nel report `discovery.pdfs` i nomi sono intenzionalmente espliciti:

- `pdfs_found`: PDF trovati;
- `pdfs_blocked_by_robots`: record PDF ancora bloccati;
- `unique_pdfs_blocked_by_robots`: PDF unici ancora bloccati;
- `blocked_by_robots_by_section` / `blocked_by_robots_by_keyword`: dove si
  concentra la perdita di copertura;
- `pdfs_allowed_by_policy`: PDF unici ammessi dalle deroghe dedicate;
- `pdfs_allowed_by_policy_by_section` /
  `pdfs_allowed_by_policy_by_keyword`: da dove arrivano e perché sono stati
  ammessi;
- `allowed_suspicious_attachments`: ammessi da ricontrollare se in futuro una
  regola lascia passare PDF `home_bandi` che conservano hint da allegato
  accessorio;
- `blocked_review_candidates`: PDF ancora bloccati che meritano attenzione
  perché mostrano segnali documentali espliciti, come documento stabile,
  materiale didattico operativo, programma internazionale, evidenza del corso
  o opportunità principale;
- `blocked_intentionally_excluded`: PDF che la policy lascia fuori di proposito,
  ad esempio perché arrivano da `home_bandi` o sono solo `bando`,
  `graduatoria`, `decreto`;
- `blocked_other`: PDF bloccati che non ricadono in nessuna delle due categorie
  precedenti.

I PDF `robots_denied` che non rispettano la regola dedicata restano tracciati in
`discovered_urls.jsonl`, ma non vengono scaricati.

## 4. Ingest

Comando completo:

```bash
python src/ingest.py
```

Comando per rigenerare solo duplicati e statistiche:

```bash
python src/ingest.py --stats-only
```

Responsabilità:

- coordina discovery, scraping HTML ed estrazione PDF;
- marca documenti duplicati tramite `content_hash`, considerando lo stato
  corrente del manifest;
- genera `data/processed/stats.json`;
- salva una copia storica in `data/processed/runs/<crawl_run_id>/stats.json`.

`ingest.py` non decide i limiti del crawl: i parametri effettivi sono sempre
letti da `config.yaml`.

`data/processed/manifest.jsonl` resta append-only: se un URL fallisce e poi
riesce in una run successiva, conserva entrambe le righe. Le statistiche finali
usano invece solo l'ultimo record disponibile per ogni `source+url`, così
`failed` vecchi non falsano lo stato attuale del corpus.

La futura fase di indexing userà la stessa regola tramite
`pipeline_io.latest_records_by_url()`: leggerà lo storico `manifest.jsonl`,
terrà solo lo stato corrente di ogni documento, poi indicizzerà solo record
`status="ok"`, `indexable=true`, `text_extracted=true`, non duplicati e con
`index_markdown_path` presente.

## 5. Chunking

Comando:

```bash
python src/chunking.py
```

Responsabilità:

- legge data/processed/manifest.jsonl;
- usa `pipeline_io.latest_records_by_url()` per considerare solo lo stato corrente del manifest append-only;
- seleziona solo documenti con status="ok", indexable=true, text_extracted=true, non duplicati e con index_markdown_path presente;
legge i Markdown puliti da data/processed/markdown/;
- rimuove il front matter YAML iniziale, perché i metadati sono già presenti nel manifest;
- rimuove eventuali header contestuali già presenti per evitare duplicazioni;
- normalizza piccoli artefatti testuali prima del chunking;
- divide i documenti prima per sezioni Markdown e poi, quando necessario, con split a dimensione controllata;
- aggiunge a ogni chunk un header contestuale con titolo, breadcrumb e fonte;
- salva i chunk in `data/processed/chunks/chunks.jsonl`;
- salva statistiche del chunking in `data/processed/chunks/stats.json`.

Ogni chunk conserva il testo da indicizzare insieme ai metadati principali del documento sorgente, tra cui chunk_id, document_hash, source_url, title, breadcrumb, chunk_index, text_hash e numero di caratteri.

La fase successiva usa `chunks.jsonl` per generare gli embedding e popolare il vector store.

## 6. Vector Store

Creazione del vector store:

```bash
python src/vector_store.py --reset
```

Query di test:
```bash
python src/vector_store.py --query "Quali corsi di laurea offre il DIEM?"
```
Responsabilità:

- legge `data/processed/chunks/chunks.jsonl`;
- converte ogni chunk in un documento LangChain con testo e metadati;
- genera gli embedding tramite un modello HuggingFace multilingua;
- indicizza i vettori nel database Chroma;
- salva il vector store in `data/vectorstore/chroma/`;
- salva statistiche in `data/vectorstore/stats.json`;
- consente query semantiche di test tramite parametro --query.

Il vector store non deve essere versionato su Git perché è un artefatto generato localmente.

## 7. Retrieval Ibrido

Comando:

```bash
python src/retrieval.py --query "Quali corsi di laurea offre il DIEM?"
```
Comando con numero finale di risultati personalizzato:

```bash
python src/retrieval.py --query "Quali sono gli orari di ricevimento del professor Mario Vento?" --final-k 3
```
Responsabilità:

- legge i chunk da `data/processed/chunks/chunks.jsonl`;
- esegue retrieval lessicale tramite BM25;
- esegue dense retrieval tramite il vector store Chroma creato da `vector_store.py`;
- combina i risultati dei due retriever tramite Reciprocal Rank Fusion;
- rimuove risultati ridondanti provenienti dallo stesso URL;
- applica un rerank leggero basato sui metadati e sul tipo di query;
- stampa i chunk finali con titolo, URL, breadcrumb, chunk ID e anteprima del contenuto.

Il retrieval ibrido migliora la robustezza rispetto alla sola ricerca vettoriale:

- BM25 è utile per nomi propri, docenti, sigle, codici corso, URL e parole chiave esatte;
- il dense retrieval è utile per domande formulate in linguaggio naturale e semanticamente simili ai documenti;
- la fusione RRF evita di confrontare direttamente score eterogenei prodotti da BM25 e Chroma.

Il rerank leggero sui metadati gestisce alcuni casi frequenti:

- domande su corsi di laurea e offerta formativa;
- domande sugli orari di ricevimento di un docente specifico;
- domande generiche sulla didattica;
- penalizzazione di pagine in lingua inglese quando la query è in italiano;
- penalizzazione di pagine relative ad anni accademici vecchi se la query non specifica un anno;
- deduplica dei risultati provenienti dallo stesso URL.

Questa fase non genera ancora la risposta finale: produce i chunk più rilevanti che saranno poi forniti al modello LLM nella fase RAG completa.

## 8. RAG Generation

Comando:

```bash
python src/chatbot_cli.py
```

Comando con modello e numero di chunk personalizzati:

```bash
python src/chatbot_cli.py --model llama-3.3-70b-versatile --final-k 3
```

Interfaccia grafica Chainlit:

```bash
chainlit run src/app.py -w
```

Responsabilità:

- riceve una domanda utente da terminale;
- usa `retrieval.py` per recuperare i chunk più rilevanti dal corpus DIEM;
- costruisce un contesto compatto usando titolo, URL, breadcrumb, chunk ID e contenuto dei chunk;
- genera un prompt RAG vincolato alle fonti recuperate;
- invia il prompt a un modello LLM tramite API Groq;
- restituisce una risposta in italiano insieme alle fonti utilizzate.

La generazione avviene tramite il modulo `src/rag_chain.py`, che implementa la pipeline:

```text
domanda utente
  -> hybrid retrieval
  -> costruzione contesto
  -> prompt RAG
  -> chiamata Groq API
  -> risposta + fonti
```

Il prompt impone al modello di usare esclusivamente il contesto fornito.
Se il contesto non contiene informazioni sufficienti, il chatbot deve dichiarare che l'informazione non è disponibile nelle fonti DIEM indicizzate.
Se la domanda è fuori dominio rispetto al DIEM, il chatbot deve segnalarlo invece di produrre una risposta non fondata.

Le variabili principali sono configurate tramite `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT_SECONDS=60
RAG_FINAL_K=3
RAG_MAX_CONTEXT_CHARS=6000
```

Questa fase completa la pipeline RAG end-to-end: i chunk recuperati dal retrieval ibrido vengono usati come contesto per generare una risposta controllata e accompagnata dalle fonti.

## Output Principali

| File / cartella | Contenuto |
|---|---|
| `data/discovered_urls.jsonl` | Manifest della discovery URL. |
| `data/raw_html/` | HTML grezzo indicizzabile. |
| `data/raw_pdf/` | PDF scaricati. |
| `data/processed/markdown_raw/` | Markdown estratto prima della pulizia. |
| `data/processed/markdown/` | Markdown pulito e indicizzabile da HTML e PDF. |
| `data/processed/manifest.jsonl` | Manifest append-only dei documenti processati. |
| `data/processed/stats.json` | Ultime statistiche generate sullo stato corrente. |
| `data/processed/runs/<crawl_run_id>/stats.json` | Copia storica delle statistiche di un run. |
| `data/checkpoint.json` | Stato per riprendere la discovery. |
| `data/processed/chunks/chunks.jsonl` | Chunk contestuali pronti per embedding e indicizzazione vettoriale. |
| `data/processed/chunks/stats.json` | Statistiche del chunking: numero chunk, lunghezze, domini e documenti più frammentati. |
| `data/vectorstore/chroma/` | Vector store Chroma generato dagli embedding dei chunk. |
| `data/vectorstore/stats.json` | Statistiche del vector store e modello embedding usato. |

## Status Discovery

| Status | Significato |
|---|---|
| `ok` | HTML indicizzabile salvato su disco. |
| `not_indexable` | HTML in scope ma escluso dall'indice. |
| `pending_download` | PDF rilevato e scaricabile nella fase PDF. |
| `robots_denied` | Risorsa rilevata ma bloccata da `robots.txt`. |
| `duplicate_redirect` | Redirect verso documento già visto. |
| `duplicate_canonical` | Canonical già visto. |
| `redirected_out_of_scope` | Redirect fuori scope. |
| `failed` | Errore HTTP o di rete. |

## Scope URL

| Dominio / risorsa | Regola |
|---|---|
| `www.diem.unisa.it` | Ammesso. |
| `docenti.unisa.it` | Solo se scoperto da DIEM o da profilo docente in scope. |
| `corsi.unisa.it` | Solo percorsi corso/codici DIEM configurati. |
| `uploads` PDF | Rilevati; scaricati solo se `robots.txt` lo permette. |

## Note Operative

Ogni record processed conserva `raw_markdown_path` e `index_markdown_path`.
I campi di qualità
(`clean_status`, `clean_warnings`, `raw_markdown_chars`,
`clean_markdown_chars`, `removed_chars_ratio`) descrivono quanto è stata
modificata l'estrazione raw.

I Markdown puliti sotto 100 caratteri restano tracciati nel manifest, ma sono
marcati `text_extracted=false` e `indexable=false` per evitare chunk di sola
navigazione o pagine tecniche quasi vuote. Se invece un PDF non produce alcun
testo estraibile, come nel caso dei PDF scannerizzati senza OCR, il record
processed viene marcato `status="failed"` con `error_kind="no_text_extracted"`;
gli export tabellari privi di dati finiscono in `empty_structured_pdf`.

Per ripartire da zero:

```bash
make clean
python src/ingest.py
```

Per continuare incrementalmente:

```bash
python src/ingest.py
```

Per aggiornare solo statistiche e duplicati:

```bash
python src/ingest.py --stats-only
```

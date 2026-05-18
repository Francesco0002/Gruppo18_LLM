# Makefile per il progetto DIEM-Chatbot.
# Pensato per macOS / Linux. Su Windows usare i comandi mostrati come
# riferimento (ogni target è una singola riga eseguibile a mano).

PY ?= python

.PHONY: help install discover scrape extract-pdf ingest clean

help:
	@echo "Target disponibili:"
	@echo "  install         Installa le dipendenze del progetto."
	@echo "  discover        Lancia la discovery con i parametri di config.yaml."
	@echo "  scrape          Converte gli HTML scoperti in Markdown."
	@echo "  extract-pdf     Scarica ed estrae i PDF scoperti."
	@echo "  ingest          Esegue pipeline completa, marca duplicati e genera stats."
	@echo "  clean           Azzera output, manifest, stats e storico run della pipeline."

install:
	$(PY) -m pip install -r requirements.txt

discover:
	$(PY) src/discover.py

scrape:
	$(PY) src/scrape.py

extract-pdf:
	$(PY) src/extract_pdf.py

ingest:
	$(PY) src/ingest.py

clean:
	$(PY) src/clean_pipeline.py

# Gruppo18_LLM

Project work for the Natural Language Processing and Large Language Models course.

## Goal

The project aims to build an LLM-powered chatbot capable of answering questions about DIEM official information using a Retrieval-Augmented Generation approach.

## Project structure

src/        Source code
data/       Local data and sample datasets
notebooks/  Experiments and prototypes
report/     Final report and documentation
tests/      Test questions and evaluation material

## Setup 

Create and activate a virtual environment:
python -m venv .venv

Install dependencies:
pip install -r requirements.txt

## Notes

The virtual environment, local data, indexes, and API keys must not be committed to GitHub.

Then:
git add .gitignore README.md
git commit -m "Update project documentation and gitignore"
git push origin main

## Run the crawler

The crawler reads the initial list of DIEM URLs from:
data/urls.txt

and downloads the corresponding HTML pages into:
data/raw/

It also generates:
data/metadata.json

Run it with:
python src/crawler.py

Generated files such as data/raw/ and data/metadata.json are ignored by Git.
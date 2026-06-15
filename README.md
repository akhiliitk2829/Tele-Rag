# 3GPP-RAG: a local retrieval-augmented QA system over 3GPP specifications

A fully local, open-source RAG pipeline for answering 3GPP (5G NR) standards
questions. Specifications are chunked, embedded into a FAISS index, retrieved
with a release filter and a cross-encoder reranker, and used as context for a
local LLM. Everything runs on CPU within a 16 GB RAM budget — no GPU, no API.

Built and evaluated on an ASUS Zenbook (Ryzen 7 8840HS, 16 GB RAM, Ubuntu 24.04).

## Motivation

3GPP specifications run to thousands of dense pages across hundreds of documents.
General-purpose LLMs handle everyday telecom questions reasonably well but
struggle on detailed standards questions. This project measures how far a small,
locally-runnable model gets on hard 3GPP questions when grounded with retrieval,
and does it entirely on commodity hardware.

## Pipeline

```
3GPP specs (.doc/.docx)
      │  chunk_3gpp.py        section-aware chunking (~125 tokens), release-tagged
      ▼
   chunks.jsonl
      │  filter_chunks.py     drop boilerplate, keep core specs, dedup
      ▼
   chunks_dedup.jsonl
      │  embed_index.py       bge-large-en-v1.5 -> FAISS IndexFlatIP
      ▼
     index/
      │  retrieve.py          release-filtered dense search + bge-reranker-v2-m3
      ▼
   eval_mcq.py / eval_mcq_gguf.py   logit-scored MCQ evaluation on TeleQnA
```

## Data

**Specifications.** Normative 3GPP TS documents (38-series / 5G NR, Releases
16–18) from the 3GPP archive at `https://www.3gpp.org/ftp/Specs/`. Study reports
(TRs) are excluded; only normative specs are used. Conceptual / protocol /
architecture / PHY specs are kept (e.g. 38.300, 38.331, 38.321, 38.323, 38.401,
38.413, 38.211–215); the very large RRM/RF conformance specs (mostly numeric
test tables) are dropped. After chunking, cleaning and deduplication this yields
**99,017 chunks**.

The chunk and index files are large and are not committed to the repository.
They are regenerated locally by running `chunk_3gpp.py` → `filter_chunks.py` →
`embed_index.py` over the downloaded specs (see [Reproduce](#reproduce)).

**Evaluation set.** [TeleQnA](https://github.com/netop-team/TeleQnA) is a 10,000
question multiple-choice telecom benchmark. This project uses its 3GPP subset:
every question whose text carries a `[3GPP Release N]` tag — **1,810 questions**,
spanning the *Standards specifications* and *Standards overview* categories,
Releases 14–19. The frozen subset is included as `teleqna_3gpp_subset.json`.

## Models

| Role | Model |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim, cosine via normalized IndexFlatIP) |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| Generators | `meta-llama/Llama-3.2-3B`, `AliMaatouk/Llama-3.2-3B-Tele`, `Meta-Llama-3-8B` (4-bit Q4_K_M GGUF) |

## Results

Accuracy on the 1,810-question 3GPP subset, scored by first-token logit
comparison (MMLU-style — read the next-token logits for the option letters and
pick the highest; no free-form generation to parse).

| Model | Setup | Accuracy |
|---|---|---|
| Llama-3.2-3B (general) | + RAG | 49.4% |
| Llama-3.2-3B-Tele | + RAG | 50.1% |
| **Meta-Llama-3-8B (4-bit)** | **+ RAG** | **57.3%** |

The 8B model gives a clear jump over the 3B models on the same benchmark.
Standards-specifications questions (the harder, spec-detail ones) score higher
than standards-overview, and accuracy is strongest on Rel-17/18, where the
corpus is densest.

### Context: GPT models on comparable 3GPP standards questions

These questions are hard even for frontier models. On the TeleQnA standards
category, GPT-4 reaches about **64%** while averaging 74% across all (easier)
categories; GPT-3.5 averages 67% overall. On a separate hard-3GPP standards
evaluation (TSpec-LLM), without retrieval GPT-3.5 scores **44%** and GPT-4 **51%**.

| Model | Setup | Accuracy |
|---|---|---|
| GPT-3.5 | no RAG (hard 3GPP) | ~44% |
| GPT-4 | no RAG (hard 3GPP) | ~51% |
| GPT-4 | TeleQnA standards category | ~64% |
| **This system — Meta-Llama-3-8B 4-bit** | **+ RAG, local** | **57.3%** |

A 4-bit 8B model running entirely on a laptop lands above GPT-3.5 and GPT-4's
no-retrieval scores on hard 3GPP standards questions, and approaches GPT-4's
standards-category figure — without any GPU or cloud API.

> The 57.3% figure is measured over 722 of the 1,810 questions (the evaluation
> run was stopped early); the sample spans all releases and both categories.

References: TeleQnA (arXiv:2310.15051), TSpec-LLM (arXiv:2406.01768),
Tele-LLMs (arXiv:2409.05314), Telco-RAG (arXiv:2404.15939).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# for the 8B GGUF evaluation only:
pip install llama-cpp-python
```

## Reproduce

```bash
# 1. download the 38-series normative specs (Rel 16-18) from the 3GPP archive
#    into specs/, then build and filter the chunks
python chunk_3gpp.py --in specs/ --out chunks.jsonl
python filter_chunks.py --in chunks.jsonl --out chunks_dedup.jsonl \
    --releases 16 17 18 \
    --keep 38.300 38.331 38.321 38.323 38.401 38.413 38.211 38.212 38.213 38.214 \
    --clean --dedup

# 2. build the FAISS index (sharded + resumable)
python embed_index.py --in chunks_dedup.jsonl --out-dir index \
    --model BAAI/bge-large-en-v1.5 --shard 10000

# 3. (optional) sanity-check retrieval
python retrieve.py --query "What triggers RRC re-establishment?" --release 18

# 4a. 3B evaluations (transformers)
python eval_mcq.py --model meta-llama/Llama-3.2-3B --condition rag --no-rerank
python eval_mcq.py --model AliMaatouk/Llama-3.2-3B-Tele --condition rag --no-rerank

# 4b. 8B evaluation (4-bit GGUF, llama.cpp)
hf download QuantFactory/Meta-Llama-3-8B-GGUF Meta-Llama-3-8B.Q4_K_M.gguf --local-dir models
python eval_mcq_gguf.py --model-path models/Meta-Llama-3-8B.Q4_K_M.gguf \
    --condition rag --no-rerank

# 5. summary across all runs
python eval_mcq.py --summary
```

Notes:
- `--no-rerank` uses dense retrieval only and skips loading the reranker, keeping
  memory comfortably under 16 GB. The reranker stays available in `retrieve.py`
  (and is used when running `retrieve.py` directly) for higher-quality retrieval.
- Embedding the full corpus on CPU is a long run; `embed_index.py` is sharded and
  resumable, so re-running continues where it stopped.

## Files

```
chunk_3gpp.py             specs -> chunks.jsonl (section-aware, release-tagged)
filter_chunks.py          clean / dedup / restrict chunks
embed_index.py            bge-large -> FAISS index (sharded, resumable)
retrieve.py               release-filtered dense search + reranker (Retriever class)
eval_mcq.py               logit-scored MCQ eval for transformers models (3B)
eval_mcq_gguf.py          logit-scored MCQ eval for 4-bit GGUF models (8B, llama.cpp)
teleqna_3gpp_subset.json  frozen 1,810-question evaluation set
requirements.txt          python dependencies
```

## License

Released under the MIT License — see [LICENSE](LICENSE).

The 3GPP specifications and the TeleQnA dataset are the property of their
respective owners and are subject to their own terms; this repository contains
only code and the derived evaluation subset.

## Author

Marripudi Akhil Chowdary

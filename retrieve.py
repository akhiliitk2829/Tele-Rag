#!/usr/bin/env python3
# Retrieval over the 3GPP FAISS index. Imported as `Retriever` by eval_mcq.py
# and eval_mcq_gguf.py to build the RAG context; can also be run directly to
# eyeball what a query returns.
#
#   python retrieve.py --query "What triggers RRC re-establishment?" --release 18
#
# search() does: bge-large query encoding -> release-filtered FAISS search ->
# optional cross-encoder rerank -> top chunks with spec/section for citation.

import argparse
import json
import os

import faiss
import numpy as np

# bge-v1.5 wants this prefix on the query only; passages are embedded raw
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, index_dir, reranker_model="BAAI/bge-reranker-v2-m3", device="cpu"):
        cfg = json.load(open(os.path.join(index_dir, "config.json")))
        self.dim = cfg["dim"]
        self.embed_model_name = cfg["model"]
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.meta = [json.loads(l) for l in
                     open(os.path.join(index_dir, "meta.jsonl"), encoding="utf-8")]
        assert self.index.ntotal == len(self.meta), "index/meta count mismatch"

        # release -> row ids, for the release filter
        self.by_release = {}
        for i, m in enumerate(self.meta):
            self.by_release.setdefault(m["release"], []).append(i)
        self.by_release = {r: np.array(ids, dtype="int64")
                           for r, ids in self.by_release.items()}

        from sentence_transformers import SentenceTransformer
        print(f"loading embed model {self.embed_model_name} ...")
        self.embedder = SentenceTransformer(self.embed_model_name, device=device)

        self.reranker = None              # lazy: only loaded when rerank=True
        self.reranker_model = reranker_model
        self.device = device

    def _load_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"loading reranker {self.reranker_model} ...")
            self.reranker = CrossEncoder(self.reranker_model, device=self.device)
        return self.reranker

    def encode_query(self, query):
        v = self.embedder.encode([QUERY_INSTRUCTION + query],
                                 normalize_embeddings=True, convert_to_numpy=True)
        return v.astype("float32")

    def search(self, query, release=None, top_k=50, rerank_k=8, rerank=True):
        qv = self.encode_query(query)

        if release is not None and release in self.by_release:
            ids = self.by_release[release]
            sel = faiss.IDSelectorArray(ids)
            try:
                D, I = self.index.search(qv, top_k, params=faiss.SearchParameters(sel=sel))
            except Exception:
                # older faiss without selector support on a flat index: over-fetch and filter
                D, I = self.index.search(qv, top_k * 4)
                keep = set(ids.tolist())
                pairs = [(d, i) for d, i in zip(D[0], I[0]) if i in keep][:top_k]
                D = np.array([[p[0] for p in pairs]])
                I = np.array([[p[1] for p in pairs]])
        else:
            D, I = self.index.search(qv, top_k)

        cands = [{**self.meta[i], "dense_score": float(d)}
                 for d, i in zip(D[0], I[0]) if i >= 0]

        if not rerank or not cands:
            return cands[:rerank_k]

        ce = self._load_reranker()
        scores = ce.predict([(query, c["text"]) for c in cands])
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
        cands.sort(key=lambda c: c["rerank_score"], reverse=True)
        return cands[:rerank_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--query", required=True)
    ap.add_argument("--release", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rerank-k", type=int, default=8)
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    r = Retriever(a.index)
    hits = r.search(a.query, release=a.release, top_k=a.top_k,
                    rerank_k=a.rerank_k, rerank=not a.no_rerank)
    print(f"\nQuery: {a.query}  (release={a.release})\n")
    for rank, h in enumerate(hits, 1):
        rr = h.get("rerank_score")
        rr = f"{rr:.3f}" if rr is not None else "n/a"
        print(f"#{rank} [{h['spec']} R{h['release']} | {h['section']}] "
              f"dense={h['dense_score']:.3f} rerank={rr}")
        print(f"    {h['text'][:200]}\n")


if __name__ == "__main__":
    main()

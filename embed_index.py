#!/usr/bin/env python3
# Embed the filtered chunks with bge-large and build a FAISS index.
# Sharded + resumable: re-running skips finished shards, so a long CPU run
# survives interruptions. Cosine similarity via IndexFlatIP on normalized vectors.
#
#   python embed_index.py --in chunks_dedup.jsonl --out-dir index \
#       --model BAAI/bge-large-en-v1.5 --batch 32 --shard 10000
#
# Writes index/index.faiss, index/meta.jsonl (one line per vector), index/config.json

import argparse
import glob
import json
import os
import time

import numpy as np


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"loaded {len(recs)} chunks from {path}")
    return recs


def embed_shards(records, model, shards_dir, shard_size, batch):
    os.makedirs(shards_dir, exist_ok=True)
    n = len(records)
    n_shards = (n + shard_size - 1) // shard_size
    for si in range(n_shards):
        out = os.path.join(shards_dir, f"emb_{si:05d}.npy")
        if os.path.exists(out):
            print(f"  shard {si+1}/{n_shards}: exists, skip")
            continue
        lo, hi = si * shard_size, min((si + 1) * shard_size, n)
        texts = [r["text"] for r in records[lo:hi]]
        t0 = time.time()
        vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True).astype("float32")
        np.save(out, vecs)
        dt = time.time() - t0
        print(f"  shard {si+1}/{n_shards}: {hi-lo} vecs in {dt:.0f}s ({(hi-lo)/dt:.1f}/s)")


def build_index(records, shards_dir, out_dir, model_name):
    import faiss
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "emb_*.npy")))
    if not shard_files:
        raise RuntimeError("no shard files found")
    emb = np.vstack([np.load(f) for f in shard_files])
    if emb.shape[0] != len(records):
        raise RuntimeError(f"vector count {emb.shape[0]} != records {len(records)}; "
                           "shards stale -- delete shards/ and re-embed")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps({k: r[k] for k in
                    ("id", "spec", "release", "section", "source_file", "text")},
                    ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "config.json"), "w") as w:
        json.dump({"model": model_name, "dim": dim, "count": int(emb.shape[0]),
                   "normalized": True, "index": "IndexFlatIP"}, w, indent=2)
    print(f"\nindex built: {emb.shape[0]} vectors, dim {dim} -> {out_dir}/index.faiss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", default="index")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=10000)
    a = ap.parse_args()

    records = load_records(a.in_path)
    shards_dir = os.path.join(a.out_dir, "shards")

    from sentence_transformers import SentenceTransformer
    print(f"loading {a.model} (CPU) ...")
    model = SentenceTransformer(a.model, device="cpu")

    t0 = time.time()
    embed_shards(records, model, shards_dir, a.shard, a.batch)
    print(f"embedding done in {(time.time()-t0)/60:.1f} min")
    build_index(records, shards_dir, a.out_dir, a.model)


if __name__ == "__main__":
    main()

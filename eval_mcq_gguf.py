#!/usr/bin/env python3
# Same logit-scoring eval as eval_mcq.py, but for a 4-bit GGUF model via
# llama-cpp-python so an 8B model fits in 16GB RAM on CPU. Reads the last
# token's logits and picks the option letter with the highest one. Writes to
# the same eval_results/ dir, so eval_mcq.py --summary lists it too.
#
# Setup:
#   pip install llama-cpp-python
#   hf download QuantFactory/Meta-Llama-3-8B-GGUF Meta-Llama-3-8B.Q4_K_M.gguf --local-dir models
#
#   python eval_mcq_gguf.py --model-path models/Meta-Llama-3-8B.Q4_K_M.gguf \
#       --condition rag --no-rerank

import argparse
import json
import os
import time

from eval_mcq import load_questions, build_prompt, build_user, LETTERS, RESULTS_DIR


class GGUFScorer:
    def __init__(self, model_path, instruct=False, n_ctx=8192, n_threads=6):
        from llama_cpp import Llama
        import numpy as np
        self.np = np
        print(f"loading GGUF {model_path} (n_ctx={n_ctx}, threads={n_threads}) ...")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         logits_all=True, verbose=False)
        self.instruct = instruct
        self.n_ctx = n_ctx
        self.no_letter = 0
        self.letter_ids = {}
        for L in LETTERS:
            for variant in (f" {L}", L):
                ids = self.llm.tokenize(variant.encode("utf-8"), add_bos=False, special=False)
                if len(ids) == 1:
                    self.letter_ids[L] = ids[0]
                    break
        print(f"  mode: {'instruct' if instruct else 'base'}  (GGUF / llama.cpp)")

    def _prompt(self, q, context):
        if not self.instruct:
            return build_prompt(q, context)
        user = build_user(q, context)
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:")

    def _last_logits(self):
        # llama-cpp-python exposes scores as 1D or 2D depending on version
        arr = self.np.array(self.llm.scores, dtype=self.np.float32)
        nv = self.llm.n_vocab()
        if arr.ndim == 2:
            return arr[self.llm.n_tokens - 1]
        if arr.size == nv:
            return arr
        return arr[(self.llm.n_tokens - 1) * nv: self.llm.n_tokens * nv]

    def score(self, q, context=None):
        toks = self.llm.tokenize(self._prompt(q, context).encode("utf-8"),
                                 add_bos=True, special=True)
        budget = self.n_ctx - 4
        if len(toks) > budget:                  # trim from the left, keep BOS + tail
            toks = toks[:1] + toks[-(budget - 1):]
        self.llm.reset()
        self.llm.eval(toks)
        logits = self._last_logits()
        best, best_lp, found = 0, -1e18, False
        for i in range(len(q["options"])):
            tid = self.letter_ids.get(LETTERS[i])
            if tid is None:
                continue
            lp = float(logits[tid])
            if lp > best_lp:
                best_lp, best, found = lp, i, True
        if not found:
            self.no_letter += 1
            return -1
        return best


def run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = os.path.basename(args.model_path)
    out_path = os.path.join(RESULTS_DIR, f"{tag}__{args.condition}.jsonl")

    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    print(f"{len(done)} already scored; resuming")

    questions = load_questions(args.questions)
    todo = [q for q in questions if q["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} questions to score (of {len(questions)})")

    retriever = None
    if args.condition == "rag":
        from retrieve import Retriever
        retriever = Retriever(args.index)

    scorer = GGUFScorer(args.model_path, instruct=args.instruct,
                        n_ctx=args.n_ctx, n_threads=args.threads)

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as w:
        for n, q in enumerate(todo, 1):
            context = None
            if retriever is not None:
                hits = retriever.search(q["question"], release=q["release"],
                                        top_k=args.top_k, rerank_k=args.rerank_k,
                                        rerank=not args.no_rerank)
                context = "\n\n".join(f"[{h['spec']} {h['section']}] {h['text']}"
                                      for h in hits)
            pred = scorer.score(q, context)
            rec = {"id": q["id"], "pred": pred, "correct": q["correct"],
                   "ok": pred == q["correct"], "release": q["release"],
                   "category": q["category"]}
            w.write(json.dumps(rec) + "\n")
            w.flush()
            if n % 25 == 0:
                print(f"  [{n}/{len(todo)}] {n/(time.time()-t0):.2f} q/s  "
                      f"(no-letter: {scorer.no_letter})")
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min, "
          f"no-letter total: {scorer.no_letter})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--condition", choices=["norag", "rag"], default="rag")
    ap.add_argument("--questions", default="teleqna_3gpp_subset.json")
    ap.add_argument("--index", default="index")
    ap.add_argument("--instruct", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rerank-k", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test the first N")
    run(ap.parse_args())

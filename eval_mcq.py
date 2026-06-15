#!/usr/bin/env python3
# Score the TeleQnA 3GPP subset by first-token logit comparison (MMLU-style):
# build a prompt ending in "Answer:", read the next-token logits for the option
# letters A/B/C/..., pick the highest. One forward pass per question, no
# generation/parsing. Base models get a plain prompt; instruct models (-it /
# -Instruct) get the chat template with a prefilled "Answer:".
#
# Resumable: results append to eval_results/<model>__<condition>.jsonl and a
# re-run skips ids already scored.
#
#   python eval_mcq.py --model meta-llama/Llama-3.2-3B --condition rag --no-rerank
#   python eval_mcq.py --model AliMaatouk/Llama-3.2-3B-Tele --condition rag --no-rerank
#   python eval_mcq.py --summary

import argparse
import json
import os
import re
import time

RELEASE_TAG = re.compile(r"\[3GPP Release (\d+)\]", re.I)
LETTERS = "ABCDEFGH"
RESULTS_DIR = "eval_results"


def load_questions(path):
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for qid, q in data.items():
        keys = sorted([k for k in q if k.startswith("option ")],
                      key=lambda k: int(k.split()[1]))
        opts = [q[k] for k in keys]
        m = re.match(r"option\s+(\d+)", q["answer"].strip(), re.I)
        if not m:
            continue
        correct_idx = int(m.group(1)) - 1
        if correct_idx >= len(opts):
            continue
        rel = RELEASE_TAG.search(q["question"])
        out.append({
            "id": qid,
            "question": RELEASE_TAG.sub("", q["question"]).strip(),
            "options": opts,
            "correct": correct_idx,
            "release": int(rel.group(1)) if rel else None,
            "category": q.get("category", ""),
        })
    return out


def build_prompt(q, context=None):
    # base models: plain text ending in "Answer:"
    lines = []
    if context:
        lines.append("Use the following 3GPP excerpts to answer.\n")
        lines.append(context.strip() + "\n")
    lines.append("Question: " + q["question"] + "\n")
    for i, opt in enumerate(q["options"]):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("\nReply with the single letter of the correct option.")
    lines.append("Answer:")
    return "\n".join(lines)


def build_user(q, context=None):
    # instruct models: this is the user turn, wrapped in the chat template later
    lines = []
    if context:
        lines.append("Use the following 3GPP excerpts to answer.\n")
        lines.append(context.strip() + "\n")
    lines.append("Question: " + q["question"] + "\n")
    for i, opt in enumerate(q["options"]):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("\nAnswer with the single capital letter of the correct option, "
                 "and nothing else.")
    return "\n".join(lines)


class LogitScorer:
    def __init__(self, model_name, device="cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        print(f"loading {model_name} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.model.eval()
        self.device = device
        name = model_name.lower()
        self.instruct = ("-it" in name) or ("instruct" in name)
        print(f"  mode: {'instruct' if self.instruct else 'base'}")
        self.letter_ids = {}
        for L in LETTERS:
            for variant in (f" {L}", L):
                ids = self.tok.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    self.letter_ids[L] = ids[0]
                    break

    def _input_ids(self, q, context):
        if not self.instruct:
            return self.tok(build_prompt(q, context), return_tensors="pt").to(self.device)
        user = build_user(q, context)
        try:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            text = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        text += "Answer:"
        return self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    def score(self, q, context=None):
        torch = self.torch
        ids = self._input_ids(q, context)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        cand = [self.letter_ids[LETTERS[i]] for i in range(len(q["options"]))]
        return int(torch.tensor([logits[c] for c in cand]).argmax().item())


def run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"{args.model.replace('/', '__')}__{args.condition}.jsonl")

    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    print(f"{len(done)} already scored; resuming")

    questions = load_questions(args.questions)
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(todo)} questions to score (of {len(questions)})")

    retriever = None
    if args.condition == "rag":
        from retrieve import Retriever
        retriever = Retriever(args.index)

    scorer = LogitScorer(args.model)

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
                print(f"  [{n}/{len(todo)}] {n/(time.time()-t0):.2f} q/s")
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min)")


def summarise():
    if not os.path.isdir(RESULTS_DIR):
        print("no results yet")
        return
    print(f"{'run':<55} {'N':>6} {'acc':>7}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if not f.endswith(".jsonl"):
            continue
        recs = [json.loads(l) for l in open(os.path.join(RESULTS_DIR, f))]
        if not recs:
            continue
        acc = sum(r["ok"] for r in recs) / len(recs)
        print(f"{f[:-6]:<55} {len(recs):>6} {acc:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--condition", choices=["norag", "rag"])
    ap.add_argument("--questions", default="teleqna_3gpp_subset.json")
    ap.add_argument("--index", default="index")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rerank-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true",
                    help="dense retrieval only; skips the reranker (~2.3GB less RAM)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summarise()
    elif not a.model or not a.condition:
        ap.error("--model and --condition required (or use --summary)")
    else:
        run(a)

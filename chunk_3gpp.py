#!/usr/bin/env python3
# Turn downloaded 3GPP .doc/.docx specs into a JSONL of retrieval chunks.
# Each line: {id, text, spec, release, series, section, source_file}
#
#   python chunk_3gpp.py --in specs/ --out chunks.jsonl
#   python chunk_3gpp.py --in specs/ --out chunks.jsonl --tokens 125 --overlap 20

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def _char_val(c):
    # 3GPP version letter -> release number (g=16, h=17, i=18, ...)
    return int(c) if c.isdigit() else ord(c.lower()) - ord("a") + 10


# 38331-i90  or  38133-id0_s0-7
FNAME = re.compile(r"^(\d{5})-([0-9a-z])([0-9a-z])([0-9a-z])(?:_(.+))?$", re.I)
HEADING_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S")
SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+")
WS = re.compile(r"\s+")

DROP_LINE = re.compile(
    r"^(3GPP\s+T[SR]\s|\bETSI\b|Release\s+\d+\s*$|^\s*\d+\s*$|"
    r"^\s*Figure\s|^\s*Table\s+of\s+Contents)", re.I)


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def parse_name(stem):
    m = FNAME.match(stem)
    if not m:
        return None
    digits, v0, v1, v2, part = m.groups()
    return {
        "spec": f"{digits[:2]}.{digits[2:]}",
        "series": int(digits[:2]),
        "release": _char_val(v0),
        "part": part,
        "base": f"{digits}-{v0}{v1}{v2}",
    }


def part_sort_key(part):
    if part is None:
        return (0, 0)
    p = part.lower()
    if "cover" in p:
        return (-1, 0)
    m = re.search(r"s(\d+)", p)
    return (1, int(m.group(1))) if m else (2, 0)


def ensure_docx(path, tmpdir):
    if path.lower().endswith(".docx"):
        return path
    if not shutil.which("soffice"):
        raise RuntimeError("need LibreOffice for .doc: sudo apt install libreoffice-writer")
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "docx", "--outdir", tmpdir, path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    out = os.path.join(tmpdir, os.path.splitext(os.path.basename(path))[0] + ".docx")
    if not os.path.exists(out):
        raise RuntimeError(f"conversion produced no file for {path}")
    return out


def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def table_text(tbl):
    rows = []
    for row in tbl.rows:
        cells = [WS.sub(" ", c.text).strip() for c in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def is_heading(par):
    name = (par.style.name or "").lower() if par.style else ""
    return name.startswith("heading")


def read_blocks(docx_path, tmpdir):
    doc = Document(ensure_docx(docx_path, tmpdir))
    section = []            # current heading stack, e.g. ['5', '5.2', '5.2.3']
    in_history = False      # change-history annex -> skip
    for block in iter_blocks(doc):
        if isinstance(block, Table):
            if in_history:
                continue
            txt = table_text(block)
            if txt:
                yield ("table", " > ".join(section), txt)
            continue
        text = WS.sub(" ", block.text).strip()
        if not text:
            continue
        if is_heading(block):
            if re.search(r"change history|annex.*history", text, re.I):
                in_history = True
                continue
            in_history = False
            m = HEADING_NUM.match(text)
            level = m.group(1).count(".") if m else 0
            section = section[:level]
            section.append(text)
            yield ("head", " > ".join(section), text)
            continue
        if in_history or DROP_LINE.match(text):
            continue
        yield ("para", " > ".join(section), text)


def chunk_section_text(text, target, overlap):
    sents = SENT_SPLIT.split(text)
    chunks, buf, buf_tok = [], [], 0
    for s in sents:
        st = count_tokens(s)
        if buf and buf_tok + st > target:
            chunks.append(" ".join(buf))
            tail, tok = [], 0
            for prev in reversed(buf):
                pt = count_tokens(prev)
                if tail and tok + pt > overlap:
                    break
                tail.insert(0, prev)
                tok += pt
                if tok >= overlap:
                    break
            buf, buf_tok = tail[:], tok
        buf.append(s)
        buf_tok += st
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def build(in_dir, out_path, target, overlap):
    # group files by (spec, release); multi-part specs get read together in order
    groups = defaultdict(list)
    for root, _, files in os.walk(in_dir):
        for f in files:
            if not f.lower().endswith((".doc", ".docx")):
                continue
            meta = parse_name(os.path.splitext(f)[0])
            if not meta:
                continue
            key = (meta["spec"], meta["release"])
            groups[key].append((part_sort_key(meta["part"]), os.path.join(root, f), meta))

    print(f"{len(groups)} specs across {sum(len(v) for v in groups.values())} files")

    n_chunks = 0
    tmpdir = tempfile.mkdtemp(prefix="docx_")
    try:
        with open(out_path, "w", encoding="utf-8") as out:
            for (spec, release), parts in sorted(groups.items()):
                parts.sort(key=lambda x: x[0])
                meta = parts[0][2]
                sec_buffer = {}          # ordered dict: section -> list of paragraph texts
                for _, path, _ in parts:
                    try:
                        for kind, sec, text in read_blocks(path, tmpdir):
                            sec_buffer.setdefault(sec, [])
                            if kind != "head":
                                sec_buffer[sec].append(text)
                    except Exception as e:
                        print(f"  ! {os.path.basename(path)}: {type(e).__name__} {e}")
                for sec, texts in sec_buffer.items():
                    body = " ".join(texts)
                    if not body.strip():
                        continue
                    for piece in chunk_section_text(body, target, overlap):
                        rec = {
                            "id": f"{spec}_r{release}_{n_chunks}",
                            "text": piece,
                            "spec": spec,
                            "release": release,
                            "series": meta["series"],
                            "section": sec or "(none)",
                            "source_file": os.path.basename(parts[0][1]),
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_chunks += 1
                print(f"  {spec} R{release}: {len(sec_buffer)} sections")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\ndone. {n_chunks} chunks -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_path", default="chunks.jsonl")
    ap.add_argument("--tokens", type=int, default=125)
    ap.add_argument("--overlap", type=int, default=20)
    a = ap.parse_args()
    build(a.in_dir, a.out_path, a.tokens, a.overlap)

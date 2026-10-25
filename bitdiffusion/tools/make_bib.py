import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "docs" / n for n in ("lit-review.md", "paper-draft.md", "handoff.md", "reading-notes.md")]
ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)")
ATOM = "{http://www.w3.org/2005/Atom}"
EXTRA_DOI = {
    "yayla2022dac": "10.1145/3489517.3530496",
    "anastassiou1989ed": "10.1109/31.34663",
}
EXTRA_ARXIV = ["2106.05203"]
SKIP_IDS = {"1901.09847"}
OPENREVIEW = {}
S2_QUERIES = {}
MANUAL = {
    "hohfeld1992probabilistic": (
        "@article{hohfeld1992probabilistic,\n"
        '  author = {H{\\"o}hfeld, Markus and Fahlman, Scott E.},\n'
        "  title = {Probabilistic rounding in neural network learning with limited precision},\n"
        "  journal = {Neurocomputing},\n  volume = {4},\n  number = {6},\n  pages = {291--299},\n"
        "  year = {1992},\n  doi = {10.1016/0925-2312(92)90014-g},\n"
        "  note = {metadata verified against Crossref on 2026-07-15}\n}"
    ),
    "karimireddy2019ef": (
        "@inproceedings{karimireddy2019ef,\n"
        "  title = {Error Feedback Fixes SignSGD and other Gradient Compression Schemes},\n"
        "  author = {Karimireddy, Sai Praneeth and Rebjock, Quentin and Stich, Sebastian U. "
        "and Jaggi, Martin},\n"
        "  booktitle = {ICML},\n  year = {2019},\n  eprint = {1901.09847},\n"
        "  note = {metadata verified against arxiv.org/abs/1901.09847 on 2026-07-15}\n}"
    ),
    "lee2022hysteresis": (
        "@inproceedings{lee2022hysteresis,\n"
        "  title = {Toward Efficient Low-Precision Training: Data Format Optimization "
        "and Hysteresis Quantization},\n"
        "  author = {Lee, Sunwoo and Park, Jeongwoo and Jeon, Dongsuk},\n"
        "  booktitle = {ICLR},\n  year = {2022},\n"
        "  url = {https://openreview.net/forum?id=3HJOA-1hb0e},\n"
        "  note = {metadata verified against DBLP on 2026-07-15}\n}"
    ),
    "shibuya2024binaryspace": (
        "@misc{shibuya2024binaryspace,\n"
        "  title = {Training Binary Neural Networks in a Binary Weight Space},\n"
        "  author = {Shibuya, Tatsukichi and Inoue, Nakamasa and Kawakami, Rei and Sato, Ikuro},\n"
        "  year = {2023},\n  howpublished = {OpenReview},\n"
        "  url = {https://openreview.net/forum?id=Dm4qrBuFKH},\n"
        "  note = {ICLR 2024 submission; metadata verified against openreview.net/forum?id=Dm4qrBuFKH "
        "on 2026-07-15}\n}"
    ),
    "seide2014onebit": (
        "@inproceedings{seide2014onebit,\n"
        "  title = {1-bit stochastic gradient descent and its application to data-parallel "
        "distributed training of speech {DNNs}},\n"
        "  author = {Seide, Frank and Fu, Hao and Droppo, Jasha and Li, Gang and Yu, Dong},\n"
        "  booktitle = {Interspeech},\n  pages = {1058--1062},\n  year = {2014},\n"
        "  note = {metadata verified against the ISCA archive (Interspeech 2014, paper seide14) "
        "on 2026-07-15}\n}"
    ),
}


def harvest_ids():
    ids = set()
    for doc in DOCS:
        if doc.exists():
            ids |= set(ARXIV_RE.findall(doc.read_text(encoding="utf-8")))
    return sorted(ids)


def fetch(url, headers=None, retries=6):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "bitdiffusion-bib/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < retries - 1:
                time.sleep(60 * (i + 1))
                continue
            if i == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(5)
        except Exception as e:
            if i == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(5)


def arxiv_batch(ids):
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        url = f"https://export.arxiv.org/api/query?id_list={','.join(chunk)}&max_results={len(chunk)}"
        raw = fetch(url)
        if raw.startswith("__ERROR__"):
            for x in chunk:
                out[x] = None
            continue
        root = ET.fromstring(raw)
        for entry in root.findall(f"{ATOM}entry"):
            eid = entry.findtext(f"{ATOM}id") or ""
            m = ARXIV_RE.search(eid)
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            if not m or not title or title.lower().startswith("error"):
                continue
            authors = [a.findtext(f"{ATOM}name") for a in entry.findall(f"{ATOM}author")]
            year = (entry.findtext(f"{ATOM}published") or "????")[:4]
            out[m.group(1)] = {"title": title, "authors": authors, "year": year}
        time.sleep(3)
    for x in ids:
        out.setdefault(x, None)
    return out


def bib_arxiv(aid, meta):
    key = "arxiv" + aid.replace(".", "_")
    if "$" not in meta["title"]:
        meta["title"] = meta["title"].replace("μ", "{$\\mu$}").replace("ΔΣ", "{$\\Delta\\Sigma$}")
    auth = " and ".join(meta["authors"])
    return (
        f"@misc{{{key},\n  title = {{{meta['title']}}},\n  author = {{{auth}}},\n"
        f"  year = {{{meta['year']}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n"
        f"  url = {{https://arxiv.org/abs/{aid}}}\n}}"
    )


def bib_doi(key, doi):
    raw = fetch(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"})
    if raw.startswith("__ERROR__") or not raw.strip().startswith("@"):
        return None
    return re.sub(r"@(\w+)\{[^,]+,", f"@\\1{{{key},", raw.strip(), count=1)


def bib_openreview(key, note_id):
    for api in ("https://api2.openreview.net", "https://api.openreview.net"):
        raw = fetch(f"{api}/notes?id={note_id}")
        if raw.startswith("__ERROR__"):
            continue
        try:
            notes = json.loads(raw).get("notes", [])
        except json.JSONDecodeError:
            continue
        if not notes:
            continue
        c = notes[0].get("content", {})
        title = c.get("title", "")
        title = title.get("value", title) if isinstance(title, dict) else title
        authors = c.get("authors", [])
        authors = authors.get("value", authors) if isinstance(authors, dict) else authors
        if not title:
            continue
        stamp = notes[0].get("pdate") or notes[0].get("cdate") or 0
        year = time.gmtime(stamp / 1000).tm_year if stamp else "????"
        return (
            f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{' and '.join(authors)}}},\n"
            f"  year = {{{year}}},\n  url = {{https://openreview.net/forum?id={note_id}}}\n}}"
        )
    return None


def sem_scholar(key, query, must_contain):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        + urllib.parse.quote(query)
        + "&fields=title,authors,year,venue,externalIds&limit=3"
    )
    raw = fetch(url)
    if raw.startswith("__ERROR__"):
        return None
    try:
        papers = json.loads(raw).get("data", [])
    except json.JSONDecodeError:
        return None
    for p in papers:
        if must_contain.lower() in (p.get("title") or "").lower():
            auth = " and ".join(a["name"] for a in p.get("authors", []))
            venue = p.get("venue") or "n/a"
            return (
                f"@misc{{{key},\n  title = {{{p['title']}}},\n  author = {{{auth}}},\n"
                f"  year = {{{p.get('year')}}},\n  howpublished = {{{venue}}},\n"
                f"  note = {{verified via Semantic Scholar API}}\n}}"
            )
    return None


def main():
    ids = sorted((set(harvest_ids()) | set(EXTRA_ARXIV)) - SKIP_IDS)
    meta = arxiv_batch(ids)
    entries, report = [], []
    for aid in ids:
        if meta[aid]:
            entries.append(bib_arxiv(aid, meta[aid]))
            report.append(f"OK    arXiv:{aid}  {meta[aid]['year']}  {meta[aid]['title'][:80]}")
        else:
            report.append(f"FAIL  arXiv:{aid}  -- did not resolve; REMOVE OR FIX before submission")
    for key, doi in EXTRA_DOI.items():
        b = bib_doi(key, doi)
        if b:
            entries.append(b)
            report.append(f"OK    doi:{doi}  ({key})")
        else:
            report.append(f"FAIL  doi:{doi}  ({key}) -- resolve manually")
    for key, nid in OPENREVIEW.items():
        b = bib_openreview(key, nid)
        if not b and key in S2_QUERIES:
            q, mc = S2_QUERIES[key]
            b = sem_scholar(key, q, mc)
            src = "semanticscholar"
        else:
            src = f"openreview:{nid}"
        if b:
            entries.append(b)
            report.append(f"OK    {src}  ({key})")
        else:
            report.append(f"FAIL  {src}  ({key}) -- resolve manually")
    for key, b in MANUAL.items():
        entries.append(b)
        report.append(f"MANUAL {key} -- entered by hand, verify against publisher")
    (ROOT / "docs" / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    (ROOT / "docs" / "bib-report.md").write_text(
        "# BibTeX Verification Report\n\nGenerated by tools/make_bib.py; every OK line was resolved "
        "against the arXiv API / doi.org / OpenReview at generation time.\n\n```\n"
        + "\n".join(report)
        + "\n```\n",
        encoding="utf-8",
    )
    ok = sum(1 for r in report if r.startswith("OK"))
    fail = sum(1 for r in report if r.startswith("FAIL"))
    print(f"{ok} verified, {fail} failed, {len(MANUAL)} manual -> docs/references.bib")


if __name__ == "__main__":
    main()

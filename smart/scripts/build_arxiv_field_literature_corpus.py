#!/usr/bin/env python3
"""Build a strict 3x100 full-text corpus for DeAL study and visual design."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from audit_literature_visual_systems import VISUAL_PATTERNS, categories, contact_sheet
from build_literature_evidence_corpus import EVIDENCE_PATTERNS, evidence_from_text, visual_captions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus/arxiv_field_corpus"
ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
USER_AGENT = "DeAL-field-literature-audit/1.0 (mailto:parsa.vatani99@gmail.com)"
NS = {"atom": "http://www.w3.org/2005/Atom", "open": "http://a9.com/-/spec/opensearch/1.1/"}

AREAS = {
    "neural_operator_surrogates": {
        "query": '(all:"neural operator" OR all:"operator learning" OR all:DeepONet) AND (all:PDE OR all:physics OR all:simulation)',
        "required": ("neural operator", "operator learning", "deeponet", "fourier neural operator", "graph neural operator"),
        "context": ("pde", "partial differential", "physics", "simulation", "flow", "field", "operator"),
        "exclude": ("visual tracking", "proximal operator", "reinforcement learning", "quantum operator", "optical operator"),
    },
    "nonuniform_point_cloud_learning": {
        "query": 'all:"point cloud" AND (all:sampling OR all:density OR all:nonuniform OR all:convolution OR all:robustness)',
        "required": ("point cloud", "point clouds", "point set", "pointnet"),
        "context": ("sampling", "density", "nonuniform", "non-uniform", "convolution", "robust", "network", "learning"),
        "exclude": ("language model", "gaussian splatting", "autonomous driving survey", "compression standard"),
    },
    "mesh_discretization_robustness": {
        "query": '(all:"unstructured mesh" OR all:"mesh-based" OR all:discretization) AND (all:"neural operator" OR all:"graph neural network" OR all:PDE OR all:surrogate)',
        "required": ("mesh", "unstructured", "discretization", "grid", "resolution"),
        "context": ("neural operator", "graph neural", "pde", "partial differential", "physics", "simulation", "surrogate", "learning"),
        "exclude": ("image segmentation", "mesh segmentation", "face recognition", "language model", "molecular"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates-per-area", type=int, default=350)
    parser.add_argument("--papers-per-area", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--crossref-workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=95)
    parser.add_argument("--skip-visual-pages", action="store_true")
    return parser.parse_args()


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def request_bytes(url: str, attempts: int = 4, timeout: int = 120) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def arxiv_candidates(name: str, config: dict, limit: int) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": config["query"], "start": 0, "max_results": limit,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    root = ET.fromstring(request_bytes(url))
    rows = []
    for entry in root.findall("atom:entry", NS):
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=NS)).strip()
        abstract = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=NS)).strip()
        combined = (title + " " + abstract).lower()
        if any(term in title.lower() for term in config["exclude"]):
            continue
        required_hits = sum(term in combined for term in config["required"])
        context_hits = sum(term in combined for term in config["context"])
        if required_hits == 0 or context_hits == 0:
            continue
        arxiv_url = entry.findtext("atom:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"v\d+$", "", arxiv_url.rsplit("/", 1)[-1])
        authors = "; ".join(author.findtext("atom:name", default="", namespaces=NS) for author in entry.findall("atom:author", NS))
        categories_found = [category.attrib.get("term", "") for category in entry.findall("atom:category", NS)]
        rows.append({
            "area": name, "arxiv_id": arxiv_id, "title": title, "abstract": abstract,
            "authors": authors, "published": entry.findtext("atom:published", default="", namespaces=NS),
            "categories": ";".join(categories_found),
            "relevance_score": 4 * required_hits + context_hits,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        })
    deduplicated = {normalize_title(row["title"]): row for row in rows}
    return list(deduplicated.values())


def known_citations() -> dict[str, int]:
    path = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus/paper_evidence_matrix.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {normalize_title(str(row.title)): int(row.citations) for _, row in frame.iterrows()}


def crossref_citations(title: str) -> tuple[int, str]:
    url = CROSSREF_API + "?" + urllib.parse.urlencode({
        "query.title": title, "rows": 4, "select": "DOI,title,is-referenced-by-count,URL",
        "mailto": "parsa.vatani99@gmail.com",
    })
    try:
        payload = json.loads(request_bytes(url, attempts=3, timeout=45))["message"]["items"]
    except Exception:
        return 0, ""
    target = normalize_title(title); best = (0.0, 0, "")
    for item in payload:
        candidate = normalize_title((item.get("title") or [""])[0])
        ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
        if ratio > best[0]:
            best = (ratio, int(item.get("is-referenced-by-count") or 0), str(item.get("DOI") or ""))
    return (best[1], best[2]) if best[0] >= 0.82 else (0, "")


def rank_candidates(candidates: list[dict], workers: int) -> list[dict]:
    known = known_citations(); pending = []
    for index, row in enumerate(candidates):
        citation = known.get(normalize_title(row["title"]))
        if citation is None:
            pending.append((index, row["title"]))
        else:
            row["citation_proxy"] = citation; row["citation_source"] = "OpenAlex"; row["doi"] = ""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(crossref_citations, title): index for index, title in pending}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]; citations, doi = future.result()
            candidates[index]["citation_proxy"] = citations
            candidates[index]["citation_source"] = "Crossref"
            candidates[index]["doi"] = doi
            if done % 50 == 0 or done == len(futures):
                print(f"Resolved citation proxy {done}/{len(futures)}.", flush=True)
    return sorted(candidates, key=lambda row: (row.get("citation_proxy", 0), row["relevance_score"]), reverse=True)


def process_paper(row: dict, output: Path, dpi: int, render_pages: bool) -> dict:
    stem = f"{row['area']}_{row['rank']:03d}_{row['arxiv_id'].replace('/', '_')}"
    pdf_dir = output / "pdfs"; text_dir = output / "full_text"; page_dir = output / "visual_pages" / stem
    pdf_dir.mkdir(parents=True, exist_ok=True); text_dir.mkdir(parents=True, exist_ok=True); page_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"; text_path = text_dir / f"{stem}.txt"
    try:
        if not pdf.exists():
            pdf.write_bytes(request_bytes(row["pdf_url"], timeout=180))
        if pdf.read_bytes()[:5] != b"%PDF-":
            raise ValueError("not a PDF")
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        evidence, snippets = evidence_from_text(text)
        captions = visual_captions(text)
        pages = []
        if render_pages:
            info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True, timeout=30).stdout
            match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE); count = int(match.group(1)) if match else 0
            for page in range(1, count + 1):
                page_text = subprocess.run(
                    ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
                    check=True, capture_output=True, text=True, timeout=30,
                ).stdout
                if not re.search(r"\b(?:fig(?:ure)?\.?|table)\s*[a-z]?\d+", page_text, flags=re.IGNORECASE):
                    continue
                image_stem = page_dir / f"page_{page:03d}"
                subprocess.run(
                    ["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-jpeg", "-jpegopt", "quality=90", "-r", str(dpi), str(pdf), str(image_stem)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
                )
                pages.append({"page": page, "image_path": str(image_stem.with_suffix('.jpg')), "categories": ";".join(categories(page_text))})
        return {"status": "extracted", "words": len(text.split()), "evidence": evidence, "snippets": snippets, "captions": captions, "pages": pages}
    except Exception as error:
        return {"status": f"failed:{type(error).__name__}", "words": 0, "evidence": {}, "snippets": [], "captions": [], "pages": []}


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for name, config in AREAS.items():
        candidates = arxiv_candidates(name, config, args.candidates_per_area)
        if len(candidates) < args.papers_per_area:
            raise RuntimeError(f"Only {len(candidates)} strict candidates for {name}")
        print(f"{name}: {len(candidates)} strict candidates.", flush=True)
        ranked = rank_candidates(candidates, args.crossref_workers)[:args.papers_per_area]
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        selected.extend(ranked)
        time.sleep(2)
    pd.DataFrame(selected).to_csv(args.output_dir / "selected_3x100_papers.csv", index=False)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_paper, row, args.output_dir, args.dpi, not args.skip_visual_pages): index for index, row in enumerate(selected)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]; results[index] = future.result()
            if done % 20 == 0 or done == len(futures):
                print(f"Read and audited {done}/{len(futures)} papers.", flush=True)

    paper_rows = []; snippet_rows = []; caption_rows = []; page_rows = []
    for index, row in enumerate(selected):
        result = results[index]
        paper_rows.append({**row, "full_text_status": result["status"], "full_text_words": result["words"], "figure_captions": sum(kind == "figure" for kind, _ in result["captions"]), "table_captions": sum(kind == "table" for kind, _ in result["captions"]), **result["evidence"]})
        for snippet in result["snippets"]:
            label, context = snippet.split(": ", 1); snippet_rows.append({"area": row["area"], "rank": row["rank"], "title": row["title"], "evidence_type": label, "context": context})
        for kind, caption in result["captions"]:
            caption_rows.append({"area": row["area"], "rank": row["rank"], "title": row["title"], "kind": kind, "categories": ";".join(categories(caption)), "caption": caption})
        for page in result["pages"]:
            page_rows.append({"area": row["area"], "rank": row["rank"], "title": row["title"], **page})
    papers = pd.DataFrame(paper_rows); papers.to_csv(args.output_dir / "paper_evidence_matrix.csv", index=False)
    pd.DataFrame(snippet_rows).to_csv(args.output_dir / "evidence_contexts.csv", index=False)
    pd.DataFrame(caption_rows).to_csv(args.output_dir / "visual_caption_audit.csv", index=False)
    pages = pd.DataFrame(page_rows); pages.to_csv(args.output_dir / "visual_page_index.csv", index=False)

    evidence_summary = []
    for area, frame in papers.groupby("area"):
        for metric in EVIDENCE_PATTERNS:
            evidence_summary.append({"area": area, "practice": metric, "papers": len(frame), "papers_reporting": int(frame[metric].fillna(False).sum()), "prevalence_percent": 100 * frame[metric].fillna(False).mean()})
    pd.DataFrame(evidence_summary).to_csv(args.output_dir / "evidence_practice_summary.csv", index=False)
    visual_summary = []
    if not pages.empty:
        for area, frame in pages.groupby("area"):
            for category in (*VISUAL_PATTERNS, "other"):
                mask = frame.categories.str.split(";").map(lambda values: category in values)
                visual_summary.append({"area": area, "category": category, "visual_pages": int(mask.sum()), "papers": int(frame.loc[mask, "title"].nunique())})
    pd.DataFrame(visual_summary).to_csv(args.output_dir / "visual_practice_summary.csv", index=False)
    if not pages.empty:
        for area, frame in pages.sort_values(["area", "rank", "page"]).groupby("area"):
            records = frame.to_dict("records")
            for start in range(0, len(records), 20):
                contact_sheet(records[start:start + 20], args.output_dir / "contact_sheets" / f"{area}_{start // 20 + 1:03d}.jpg", f"{area}: visual pages {start + 1}-{min(start + 20, len(records))}")
    summary = {
        "papers": len(papers), "full_text_extracted": int(papers.full_text_status.eq("extracted").sum()),
        "full_text_words": int(papers.full_text_words.sum()), "visual_pages": len(pages),
        "by_area": papers.groupby("area").agg(papers=("title", "size"), full_text_extracted=("full_text_status", lambda x: int((x == "extracted").sum())), words=("full_text_words", "sum")).to_dict("index"),
    }
    (args.output_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote strict full-text corpus to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

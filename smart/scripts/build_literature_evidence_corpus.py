#!/usr/bin/env python3
"""Build a traceable literature corpus for DeAL evaluation design.

The script ranks relevant OpenAlex works by citation count, keeps 100 works in
each research area, downloads every accessible open PDF, extracts its text, and
records paper-level evidence for common experimental practices. Closed or
unreachable works remain in the corpus with abstract-only status.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus"
OPENALEX = "https://api.openalex.org/works"
USER_AGENT = "DeAL-literature-audit/1.0 (mailto:research@example.org)"

AREAS = {
    "neural_operator_surrogates": {
        "queries": (
            "neural operator", "operator learning", "DeepONet", "Fourier neural operator",
            "graph neural operator", "geometric neural operator", "PDE surrogate",
            "physics informed operator", "scientific machine learning PDE", "wavelet neural operator",
            "latent neural operator", "multipole graph neural operator", "geometry informed neural operator",
            "Geo-FNO", "GINO neural operator", "PDE operator learning", "neural operator simulation",
        ),
        "core": (
            "neural operator", "neural operators", "deeponet", "operator learning", "operator network",
            "operator approximation", "operator inference", "fourier neural", "wavelet neural operator",
            "graph neural operator", "physics-informed neural operator", "physics informed neural operator",
        ),
        "scientific": (
            "partial differential", "pde", "physics", "scientific", "flow", "fluid", "weather",
            "climate", "mechanic", "field", "simulation", "equation", "dynamics", "heat", "wave",
            "geometry", "turbulence", "elastic", "material", "function space",
        ),
        "support": (
            "resolution", "discretization", "mesh", "geometry", "surrogate", "operator", "field",
            "generalization", "invariant", "simulation", "solver",
        ),
        "exclude_title": (
            "visual tracking", "operator selection", "proximal operator", "image operator",
            "optical learning operator", "implantation", "vascular experience", "quantum machine",
            "selection operator", "sigmoidal function", "breast cancer", "learning curve",
        ),
        "from_year": 2015,
    },
    "nonuniform_point_cloud_learning": {
        "queries": (
            "point cloud", "point set network", "3D point cloud", "point cloud sampling",
            "point cloud convolution", "point cloud density", "point cloud registration",
            "point cloud representation learning", "point cloud robustness", "PointNet", "PointNet++",
            "PointConv", "KPConv point cloud", "Monte Carlo convolution point cloud", "PointCNN",
            "Point Transformer point cloud", "PointNeXt", "density adaptive point cloud",
            "nonuniform point cloud", "irregular point sampling",
        ),
        "core": ("point cloud", "point clouds", "point set", "pointnet", "3d points"),
        "scientific": (
            "deep learning", "neural", "network", "convolution", "transformer", "representation learning",
            "classification", "segmentation", "sampling", "density", "robust", "geometry",
        ),
        "support": (
            "sampling", "density", "nonuniform", "non-uniform", "irregular", "convolution", "geometry",
            "robust", "neighborhood", "local feature", "invariant", "upsampling", "downsampling",
        ),
        "exclude_title": (
            "object detection", "proposal generation", "place recognition", "robust grasps", "lidar odometry",
            "tree segmentation", "building information", "mosaic", "autonomous vehicle", "road-object",
            "registration using", "point cloud registration", "point cloud library",
        ),
        "from_year": 2010,
    },
    "mesh_discretization_robustness": {
        "queries": (
            "unstructured mesh neural", "mesh neural network", "remeshing learning",
            "mesh generation machine learning", "discretization neural operator",
            "graph mesh PDE", "geometric deep learning mesh", "finite element neural",
            "mesh invariant learning", "resolution invariant operator", "MeshGraphNets",
            "learning mesh based simulation", "graph neural operator PDE", "Geo-FNO",
            "geometry informed neural operator", "multipole graph neural operator", "neural operator mesh",
            "arbitrary geometry neural operator", "unstructured grid surrogate", "mesh independent learning",
            "discretization invariant neural operator", "resolution generalization PDE",
        ),
        "core": ("mesh", "remesh", "discretization", "unstructured", "finite element", "resolution", "grid"),
        "scientific": (
            "neural operator", "operator learning", "graph network", "graph neural", "deep learning",
            "neural network", "pde", "partial differential", "physics", "simulation", "surrogate",
        ),
        "support": (
            "mesh", "remesh", "discretization", "unstructured", "resolution", "geometry", "invariant",
            "generalization", "simulation", "operator", "pde", "solver",
        ),
        "exclude_title": (
            "face detection", "face recognition", "mesh segmentation", "mesh labeling", "shape representation",
            "bone remodelling", "bone remodeling", "shipbuilding steel", "porous beams", "damage detection",
            "optical processor", "double lap joints", "indentation", "pavement", "polymer nanocomposites",
            "training finite element neural network", "fuel cell", "crystal plasticity",
        ),
        "from_year": 2010,
    },
}

EVIDENCE_PATTERNS = {
    "relative_l2": (r"relative\s+(?:l2|l_?2|ell[- ]?2)", r"relative\s+error"),
    "absolute_error": (r"mean\s+absolute\s+error", r"\bmae\b", r"root\s+mean\s+square", r"\brmse\b"),
    "tail_or_worst_case": (r"worst[- ]case", r"90th\s+percentile", r"95th\s+percentile", r"maximum\s+error", r"tail\s+risk"),
    "uncertainty_or_multiple_seeds": (r"multiple\s+(?:random\s+)?seeds", r"standard\s+deviation", r"confidence\s+interval", r"error\s+bars?", r"bootstrap"),
    "resolution_or_discretization": (r"resolution\s+(?:invariance|generalization|study)", r"discretization", r"mesh\s+resolution", r"grid\s+resolution", r"convergence\s+study"),
    "nonuniform_sampling": (r"non[- ]?uniform\s+sampling", r"sampling\s+density", r"density\s+variation", r"irregular\s+sampling", r"point\s+density"),
    "out_of_distribution": (r"out[- ]of[- ]distribution", r"\bood\b", r"distribution\s+shift", r"unseen\s+(?:geometry|condition|resolution)"),
    "ablation": (r"ablation\s+stud", r"ablat(?:e|ed|ion)", r"component\s+analysis"),
    "compute_reporting": (r"number\s+of\s+parameters", r"parameter\s+count", r"inference\s+time", r"training\s+time", r"memory\s+(?:usage|cost)", r"flops"),
    "physics_metric": (r"drag\s+coefficient", r"lift\s+coefficient", r"conservation\s+(?:error|law)", r"residual\s+error", r"energy\s+error", r"flux\s+error"),
    "fieldwise_reporting": (r"field[- ]wise", r"per[- ]channel", r"pressure\s+error", r"velocity\s+error", r"temperature\s+error"),
    "geometry_fidelity": (r"chamfer\s+distance", r"hausdorff\s+distance", r"normal\s+(?:error|deviation|consistency)", r"surface\s+area", r"geometric\s+distortion"),
    "statistical_test": (r"wilcoxon", r"significance\s+test", r"p[- ]?value", r"paired\s+test", r"effect\s+size"),
}

VISUAL_PATTERNS = {
    "method_or_architecture": (r"architect", r"framework", r"pipeline", r"overview", r"schematic", r"workflow"),
    "qualitative_field": (r"pressure", r"velocity", r"temperature", r"vorticity", r"stress", r"field", r"solution"),
    "geometry_or_dataset": (r"geometr", r"mesh", r"point cloud", r"dataset", r"sample", r"domain"),
    "ablation_or_sensitivity": (r"ablation", r"sensitivity", r"hyperparameter", r"component"),
    "accuracy_comparison": (r"error", r"accuracy", r"comparison", r"performance", r"baseline"),
    "resolution_or_scaling": (r"resolution", r"grid", r"discret", r"mesh size", r"number of points", r"scal"),
    "uncertainty_or_distribution": (r"confidence", r"standard deviation", r"distribution", r"boxplot", r"violin", r"histogram"),
    "training_or_convergence": (r"training", r"validation", r"convergence", r"epoch", r"loss curve"),
    "compute_or_efficiency": (r"runtime", r"speedup", r"memory", r"throughput", r"parameter", r"flops"),
    "geometry_fidelity": (r"hausdorff", r"chamfer", r"normal deviation", r"surface area", r"mesh quality"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--papers-per-area", type=int, default=100)
    parser.add_argument("--query-results", type=int, default=200)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--max-pdf-mib", type=int, default=40)
    parser.add_argument("--keep-pdfs", action="store_true")
    parser.add_argument("--render-visual-pages", action="store_true")
    parser.add_argument("--visual-dpi", type=int, default=105)
    parser.add_argument("--skip-fulltext", action="store_true")
    return parser.parse_args()


def invert_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(words))


def api_request(params: dict[str, str | int], attempts: int = 5) -> dict:
    url = OPENALEX + "?" + urllib.parse.urlencode(params)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def paper_text(work: dict) -> str:
    return " ".join((work.get("title") or "", invert_abstract(work.get("abstract_inverted_index")))).lower()


def relevance_score(work: dict, area: dict) -> float:
    title = (work.get("title") or "").lower()
    abstract = invert_abstract(work.get("abstract_inverted_index")).lower()
    combined = title + " " + abstract
    if any(term in title for term in area["exclude_title"]):
        return -math.inf
    core = sum(4 if term in title else 1.5 for term in area["core"] if term in combined)
    scientific = sum(2 if term in title else 0.75 for term in area["scientific"] if term in combined)
    support = sum(1.0 if term in title else 0.25 for term in area["support"] if term in combined)
    if core <= 0 or scientific <= 0:
        return -math.inf
    return core + scientific + support


def collect_area(area_name: str, area: dict, query_results: int, limit: int) -> list[dict]:
    candidates: dict[str, dict] = {}
    select = ",".join(
        (
            "id", "doi", "title", "publication_year", "cited_by_count", "type", "authorships",
            "primary_location", "best_oa_location", "locations", "open_access", "abstract_inverted_index",
            "keywords", "primary_topic",
        )
    )
    for query in area["queries"]:
        payload = api_request(
            {
                "filter": f"title.search:{query},from_publication_date:{area['from_year']}-01-01",
                "sort": "cited_by_count:desc",
                "per-page": min(query_results, 200),
                "select": select,
            }
        )
        for work in payload.get("results", []):
            score = relevance_score(work, area)
            if not math.isfinite(score):
                continue
            work["area"] = area_name
            work["relevance_score"] = score
            candidates[work["id"]] = work
        time.sleep(0.12)
    ranked = sorted(
        candidates.values(),
        key=lambda item: (item.get("cited_by_count") or 0, item["relevance_score"]),
        reverse=True,
    )
    if len(ranked) < limit:
        raise RuntimeError(f"Only {len(ranked)} relevant works found for {area_name}; need {limit}")
    return ranked[:limit]


def author_text(work: dict) -> str:
    authors = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    return "; ".join(authors)


def normalize_pdf_url(url: str) -> list[str]:
    candidates = []
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("/abs/", 1)[1].split("?", 1)[0].rstrip("/")
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    if "openreview.net/forum" in url:
        candidates.append(url.replace("/forum", "/pdf"))
    if "openreview.net/pdf" in url:
        candidates.append(url)
    if "proceedings.mlr.press/" in url and url.endswith(".html"):
        candidates.append(url.rsplit("/", 1)[0] + "/" + url.rsplit("/", 1)[1].replace(".html", ".pdf"))
    if "papers.nips.cc/paper_files/paper/" in url and "/Abstract-" in url:
        candidates.append(url.replace("/Abstract-", "/Paper-").replace(".html", ".pdf"))
    candidates.append(url)
    return candidates


def candidate_pdf_urls(work: dict) -> list[str]:
    urls = []
    locations = [work.get("best_oa_location"), work.get("primary_location"), *(work.get("locations") or [])]
    for location in locations:
        if location and location.get("pdf_url"):
            urls.append(location["pdf_url"])
        if location and location.get("landing_page_url"):
            landing = location["landing_page_url"]
            if any(host in landing for host in ("arxiv.org", "openreview.net", "proceedings.mlr.press", "papers.nips.cc")):
                urls.append(landing)
    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url:
        urls.append(oa_url)
    expanded = [candidate for url in urls for candidate in normalize_pdf_url(url)]
    return list(dict.fromkeys(expanded))


def safe_stem(work: dict) -> str:
    identifier = work["id"].rsplit("/", 1)[-1]
    title = re.sub(r"[^a-z0-9]+", "-", (work.get("title") or "paper").lower()).strip("-")[:70]
    return f"{identifier}_{title}"


def download_pdf(work: dict, destination: Path, max_bytes: int) -> tuple[str, str]:
    for url in candidate_pdf_urls(work):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("PDF exceeds configured size limit")
                    output.write(block)
            with destination.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise ValueError("response is not a PDF")
            return "downloaded", url
        except Exception:
            destination.unlink(missing_ok=True)
    return ("no_open_pdf" if not candidate_pdf_urls(work) else "download_failed"), ""


def evidence_from_text(text: str) -> tuple[dict[str, bool], list[str]]:
    normalized = re.sub(r"\s+", " ", text).lower()
    evidence = {
        key: any(re.search(pattern, normalized) for pattern in patterns)
        for key, patterns in EVIDENCE_PATTERNS.items()
    }
    snippets = []
    for key, patterns in EVIDENCE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                start = max(0, match.start() - 170); end = min(len(normalized), match.end() + 240)
                snippets.append(f"{key}: {normalized[start:end].strip()}")
                break
    return evidence, snippets


def visual_captions(text: str) -> list[tuple[str, str]]:
    captions = []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        match = re.match(r"^(fig(?:ure)?\.?|table)\s*([a-z]?\d+[a-z]?)\s*[:.\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        kind = "table" if match.group(1).lower().startswith("table") else "figure"
        caption = match.group(3)
        cursor = index + 1
        while len(caption) < 500 and cursor < len(lines) and lines[cursor] and not re.match(r"^(fig(?:ure)?\.?|table|\d+(?:\.\d+)*\s+)", lines[cursor], flags=re.IGNORECASE):
            caption += " " + lines[cursor]
            cursor += 1
        if len(caption) >= 20:
            captions.append((kind, caption[:800]))
    return captions


def visual_categories(kind: str, caption: str) -> list[str]:
    normalized = caption.lower()
    categories = [
        category for category, patterns in VISUAL_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    ]
    if kind == "table" and not categories:
        categories.append("quantitative_table")
    if kind == "figure" and not categories:
        categories.append("other_figure")
    return categories


def render_visual_pages(pdf_path: Path, destination: Path, dpi: int) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True, timeout=30
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    page_count = int(match.group(1)) if match else 0
    rendered = []
    for page in range(1, page_count + 1):
        page_text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf_path), "-"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        if not re.search(r"\b(?:fig(?:ure)?\.?|table)\s*[a-z]?\d+", page_text, flags=re.IGNORECASE):
            continue
        output_stem = destination / f"page_{page:03d}"
        subprocess.run(
            [
                "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-jpeg",
                "-jpegopt", "quality=88", "-r", str(dpi), str(pdf_path), str(output_stem),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
        )
        rendered.append({"page": page, "path": str(output_stem.with_suffix(".jpg")), "text": page_text[:2000]})
    return rendered


def process_fulltext(
    work: dict,
    pdf_dir: Path,
    text_dir: Path,
    visual_dir: Path,
    max_bytes: int,
    keep_pdf: bool,
    render_pages: bool,
    visual_dpi: int,
) -> dict:
    stem = safe_stem(work); pdf_path = pdf_dir / f"{stem}.pdf"; text_path = text_dir / f"{stem}.txt"
    status, source = download_pdf(work, pdf_path, max_bytes)
    if status != "downloaded":
        abstract = invert_abstract(work.get("abstract_inverted_index"))
        evidence, snippets = evidence_from_text(abstract)
        return {"full_text_status": status, "full_text_source": source, "full_text_words": 0, "evidence": evidence, "snippets": snippets, "visual_captions": [], "visual_pages": []}
    try:
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        evidence, snippets = evidence_from_text(text)
        pages = render_visual_pages(pdf_path, visual_dir / stem, visual_dpi) if render_pages else []
        result = {"full_text_status": "extracted", "full_text_source": source, "full_text_words": len(text.split()), "evidence": evidence, "snippets": snippets, "visual_captions": visual_captions(text), "visual_pages": pages}
    except Exception:
        abstract = invert_abstract(work.get("abstract_inverted_index"))
        evidence, snippets = evidence_from_text(abstract)
        result = {"full_text_status": "extract_failed", "full_text_source": source, "full_text_words": 0, "evidence": evidence, "snippets": snippets, "visual_captions": [], "visual_pages": []}
    if not keep_pdf:
        pdf_path.unlink(missing_ok=True)
    return result


def write_csv(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.output_dir / "pdfs"; text_dir = args.output_dir / "full_text"; visual_dir = args.output_dir / "visual_pages"
    pdf_dir.mkdir(exist_ok=True); text_dir.mkdir(exist_ok=True); visual_dir.mkdir(exist_ok=True)
    all_works = []
    for name, area in AREAS.items():
        works = collect_area(name, area, args.query_results, args.papers_per_area)
        for rank, work in enumerate(works, 1):
            work["citation_rank_within_area"] = rank
        all_works.extend(works)
        print(f"Collected {len(works)} works for {name}.", flush=True)

    processed = {}
    if not args.skip_fulltext:
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {
                pool.submit(
                    process_fulltext, work, pdf_dir, text_dir, visual_dir,
                    args.max_pdf_mib * 1024 * 1024, args.keep_pdfs,
                    args.render_visual_pages, args.visual_dpi,
                ): work
                for work in all_works
            }
            for index, future in enumerate(as_completed(futures), 1):
                work = futures[future]
                try:
                    processed[work["id"]] = future.result()
                except Exception as error:
                    evidence, snippets = evidence_from_text(invert_abstract(work.get("abstract_inverted_index")))
                    processed[work["id"]] = {"full_text_status": f"error:{type(error).__name__}", "full_text_source": "", "full_text_words": 0, "evidence": evidence, "snippets": snippets, "visual_captions": [], "visual_pages": []}
                if index % 25 == 0 or index == len(futures):
                    print(f"Processed full text {index}/{len(futures)}.", flush=True)
    else:
        for work in all_works:
            evidence, snippets = evidence_from_text(invert_abstract(work.get("abstract_inverted_index")))
            processed[work["id"]] = {"full_text_status": "skipped", "full_text_source": "", "full_text_words": 0, "evidence": evidence, "snippets": snippets, "visual_captions": [], "visual_pages": []}

    rows = []
    snippet_rows = []
    visual_rows = []
    visual_page_rows = []
    for work in all_works:
        result = processed[work["id"]]
        topic = work.get("primary_topic") or {}
        row = {
            "area": work["area"],
            "rank": work["citation_rank_within_area"],
            "openalex_id": work["id"],
            "doi": work.get("doi") or "",
            "title": work.get("title") or "",
            "authors": author_text(work),
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count", 0),
            "type": work.get("type") or "",
            "topic": topic.get("display_name") or "",
            "relevance_score": work["relevance_score"],
            "abstract_available": bool(work.get("abstract_inverted_index")),
            "full_text_status": result["full_text_status"],
            "full_text_source": result["full_text_source"],
            "full_text_words": result["full_text_words"],
            "figure_captions": sum(kind == "figure" for kind, _ in result["visual_captions"]),
            "table_captions": sum(kind == "table" for kind, _ in result["visual_captions"]),
            **result["evidence"],
        }
        rows.append(row)
        for snippet in result["snippets"]:
            label, text = snippet.split(": ", 1)
            snippet_rows.append({"area": work["area"], "rank": work["citation_rank_within_area"], "openalex_id": work["id"], "title": work.get("title") or "", "evidence_type": label, "context": text})
        for kind, caption in result["visual_captions"]:
            categories = visual_categories(kind, caption)
            visual_rows.append({"area": work["area"], "rank": work["citation_rank_within_area"], "openalex_id": work["id"], "title": work.get("title") or "", "kind": kind, "categories": ";".join(categories), "caption": caption})
        for page in result["visual_pages"]:
            visual_page_rows.append({"area": work["area"], "rank": work["citation_rank_within_area"], "openalex_id": work["id"], "title": work.get("title") or "", "page": page["page"], "image_path": page["path"]})

    columns = list(rows[0])
    write_csv(args.output_dir / "paper_evidence_matrix.csv", rows, columns)
    write_csv(args.output_dir / "evidence_contexts.csv", snippet_rows, ["area", "rank", "openalex_id", "title", "evidence_type", "context"])
    write_csv(args.output_dir / "visual_caption_audit.csv", visual_rows, ["area", "rank", "openalex_id", "title", "kind", "categories", "caption"])
    write_csv(args.output_dir / "visual_page_index.csv", visual_page_rows, ["area", "rank", "openalex_id", "title", "page", "image_path"])
    summary = {}
    for area in AREAS:
        subset = [row for row in rows if row["area"] == area]
        summary[area] = {
            "papers": len(subset),
            "full_text_extracted": sum(row["full_text_status"] == "extracted" for row in subset),
            "abstract_available": sum(bool(row["abstract_available"]) for row in subset),
            "figure_captions": sum(int(row["figure_captions"]) for row in subset),
            "table_captions": sum(int(row["table_captions"]) for row in subset),
            "visual_pages_rendered": sum(1 for page in visual_page_rows if page["area"] == area),
            "total_citations": sum(int(row["citations"]) for row in subset),
            "practice_prevalence_percent": {
                key: 100 * sum(bool(row[key]) for row in subset) / len(subset)
                for key in EVIDENCE_PATTERNS
            },
        }
    summary["visual_practice_counts"] = {
        category: sum(category in row["categories"].split(";") for row in visual_rows)
        for category in VISUAL_PATTERNS
    }
    (args.output_dir / "literature_survey_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote literature evidence corpus to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

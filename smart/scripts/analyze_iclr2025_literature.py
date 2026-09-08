#!/usr/bin/env python3
"""Audit ICLR 2025 accepted papers and field-relevant rejected submissions."""

from __future__ import annotations

import argparse
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from audit_literature_visual_systems import VISUAL_PATTERNS, categories, contact_sheet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus/iclr2025_reviews"
USER_AGENT = "DeAL-ICLR-review-audit/1.0 (mailto:parsa.vatani99@gmail.com)"

CONCERNS = {
    "novelty / significance": (r"lack(?:s|ing)? (?:of )?novelty", r"(?:appears?|seems?) incremental", r"limited novelty", r"significance.*(?:unclear|limited|weak)", r"contribution.*(?:weak|limited|unclear)"),
    "missing or weak baselines": (r"(?:missing|lack(?:s|ing)?|insufficient).*baseline", r"baseline.*(?:weak|limited|insufficient|outdated)", r"compare.*(?:stronger|more recent)", r"additional baseline"),
    "insufficient ablation": (r"(?:missing|lack(?:s|ing)?|insufficient).*ablation", r"ablation.*(?:missing|limited|insufficient)", r"(?:cannot|does not).*isolate.*effect"),
    "insufficient uncertainty": (r"(?:missing|lack(?:s|ing)?|report).*standard deviation", r"(?:missing|lack(?:s|ing)?).*error bars?", r"confidence interval.*(?:missing|not)", r"(?:single|one) (?:random )?seed", r"statistical significance.*(?:missing|unclear|not)"),
    "metric ambiguity": (r"metric.*(?:unclear|undefined|not defined|questionable)", r"evaluation metric.*(?:unclear|missing)", r"how.*(?:metric|error).*computed", r"normalization.*(?:unclear|missing|not specified)"),
    "dataset or split ambiguity": (r"dataset.*(?:unclear|limited|small|not described)", r"train.*test split.*(?:unclear|missing)", r"data split.*(?:unclear|missing)", r"data leakage", r"benchmark.*(?:limited|missing|weak)"),
    "limited scale or coverage": (r"small.*dataset", r"limited.*experiment", r"(?:need|requires?|would benefit from|lack(?:s|ing)?).*more experiments", r"additional experiments.*(?:needed|required|necessary)", r"experimental.*(?:scale|coverage).*(?:limited|small)"),
    "generalization / OOD": (r"generalization.*(?:unclear|limited|missing|weak)", r"(?:lack|missing).*out.of.distribution", r"\bood.*(?:missing|limited|not evaluated)", r"distribution shift.*(?:missing|not evaluated)", r"unseen.*(?:not evaluated|unclear)"),
    "reproducibility / details": (r"reproduc.*(?:concern|unclear|limited|difficult)", r"(?:missing|lack(?:s|ing)?|insufficient).*implementation details", r"hyperparameter.*(?:missing|unclear|not specified)", r"code.*(?:not available|missing)", r"missing details"),
    "compute / efficiency": (r"computational cost.*(?:high|unclear|missing|concern)", r"training time.*(?:missing|unclear|high)", r"inference time.*(?:missing|unclear|high)", r"memory.*(?:cost|overhead|high)", r"efficiency.*(?:unclear|limited|concern)"),
    "theory / justification": (r"(?:lack|missing|limited|insufficient).*theor", r"theoretical.*(?:missing|weak|limited|unclear)", r"(?:not|insufficiently).*justify", r"justification.*(?:missing|weak|unclear)", r"intuition.*(?:missing|limited|unclear)"),
    "writing / clarity": (r"(?:is|are|remains?) unclear", r"hard to follow", r"confusing", r"poorly written", r"presentation.*(?:unclear|weak|needs|could be improved)", r"readability.*(?:poor|limited|improve)"),
    "visual evidence": (r"figure.*(?:unclear|hard to read|missing|poor|illegible)", r"(?:lack|missing).*qualitative", r"visualization.*(?:unclear|poor|missing)", r"plot.*(?:unclear|hard to read|missing)", r"table.*(?:unclear|hard to read|missing)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT / "iclr2025_full.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "audit")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--skip-pdfs", action="store_true")
    return parser.parse_args()


def relevance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    title = df.title.fillna("").str.lower()
    abstract = df.abstract.fillna("").str.lower()
    keywords = df.keywords.fillna("").astype(str).str.lower()
    area = df.primary_area.fillna("").str.lower()
    combined = title + " " + abstract + " " + keywords
    score = pd.Series(0.0, index=df.index)
    components = [
        (r"neural operator|operator learning|deeponet|fourier neural|graph neural operator", 7.0),
        (r"partial differential equation|\bpdes?\b|physics.informed", 4.0),
        (r"point cloud|pointnet|point set", 5.0),
        (r"unstructured mesh|mesh.?based|remesh|discretization|finite element", 5.0),
        (r"fluid|flow|simulation|surrogate|scientific machine learning|physical field|geometry", 1.0),
    ]
    for pattern, weight in components:
        score += combined.str.contains(pattern, regex=True).astype(float) * weight
    score += area.str.contains("applications to physical sciences", regex=False).astype(float) * 3.0
    score += area.str.contains(r"graphs and other geometries|time series and dynamical", regex=True).astype(float)
    for pattern, weight in (
        (r"language model|retrieval|recommend|speech|text-to|reinforcement", -7.0),
        (r"gaussian splat|image generation|video generation|medical imaging|molecule|protein", -4.0),
    ):
        score += title.str.contains(pattern, regex=True).astype(float) * weight
    label = pd.Series("adjacent", index=df.index)
    label[combined.str.contains(r"point cloud|pointnet|point set", regex=True)] = "point-cloud learning"
    label[combined.str.contains(r"unstructured mesh|mesh.?based|remesh|discretization|finite element", regex=True)] = "mesh / discretization"
    label[combined.str.contains(r"neural operator|operator learning|deeponet|fourier neural|graph neural operator|partial differential equation|\bpdes?\b", regex=True)] = "operator / PDE surrogate"
    return score, label


def abstract_statistics(text: str) -> dict:
    words = re.findall(r"\b\w+[\w'-]*\b", text or "")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
    return {
        "abstract_words": len(words),
        "sentences": len(sentences),
        "words_per_sentence": len(words) / max(len(sentences), 1),
        "contains_numeric_result": bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|times|x\b)", text or "", flags=re.IGNORECASE)),
        "contains_problem_move": bool(re.search(r"\b(?:however|yet|despite|challenge|problem|limitation)\b", text or "", flags=re.IGNORECASE)),
        "contains_method_move": bool(re.search(r"\b(?:we propose|we introduce|we present|our method|our approach)\b", text or "", flags=re.IGNORECASE)),
        "contains_evidence_move": bool(re.search(r"\b(?:experiments|results|demonstrate|outperform|improve|reduce)\b", text or "", flags=re.IGNORECASE)),
    }


def weakness_sections(review: str) -> str:
    sections = re.findall(
        r"weaknesses:\s*(.*?)(?=\n\s*(?:questions|rating|confidence|summary|strengths):|</content>|</thread_object>)",
        review or "", flags=re.IGNORECASE | re.DOTALL,
    )
    return " ".join(sections) if sections else review or ""


def download_pdf(row: pd.Series, destination: Path) -> str:
    urls = [f"https://openreview.net/pdf?id={row.paper_id}"]
    arxiv = str(row.arxiv_id or "").strip()
    if arxiv and arxiv.lower() != "nan":
        urls.append(f"https://arxiv.org/pdf/{arxiv}.pdf")
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if destination.read_bytes()[:5] == b"%PDF-":
                return url
        except Exception:
            pass
        destination.unlink(missing_ok=True)
    return ""


def paper_audit(row: pd.Series, output: Path, dpi: int) -> dict:
    stem = f"{row.selection_group}_{int(row.selection_rank):03d}_{row.paper_id}"
    pdf_dir = output / "pdfs"; text_dir = output / "full_text"; page_dir = output / "visual_pages" / stem
    pdf_dir.mkdir(parents=True, exist_ok=True); text_dir.mkdir(parents=True, exist_ok=True); page_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"; text_path = text_dir / f"{stem}.txt"
    source = download_pdf(row, pdf)
    if not source:
        return {"status": "download_failed", "source": "", "words": 0, "pages": []}
    try:
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True, timeout=30).stdout
        match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE); count = int(match.group(1)) if match else 0
        pages = []
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
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
            )
            pages.append({"page": page, "image_path": str(image_stem.with_suffix('.jpg')), "categories": ";".join(categories(page_text))})
        return {"status": "extracted", "source": source, "words": len(text.split()), "pages": pages}
    except Exception:
        return {"status": "extract_failed", "source": source, "words": 0, "pages": []}


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.dataset)
    data["relevance_score"], data["field"] = relevance(data)
    accepted = data[data.decision.str.startswith("accept", na=False)].copy()
    rejected = data[data.decision.eq("reject") & data.relevance_score.ge(5)].copy()
    high_impact = accepted.sort_values(["citations_serper", "num_reviewers"], ascending=False).head(100).copy()
    field_accepted = accepted[accepted.relevance_score.ge(5)].sort_values(["relevance_score", "citations_serper", "num_reviewers"], ascending=False).head(100).copy()
    field_rejected = rejected.assign(review_chars=rejected.reviews.fillna("").str.len()).sort_values(
        ["relevance_score", "num_reviewers", "review_chars", "citations_serper"], ascending=False
    ).head(100).copy()
    groups = []
    for name, frame in (("high_impact_accepted", high_impact), ("field_relevant_accepted", field_accepted), ("field_relevant_rejected", field_rejected)):
        frame = frame.copy(); frame["selection_group"] = name; frame["selection_rank"] = np.arange(1, len(frame) + 1); groups.append(frame)
    selected = pd.concat(groups, ignore_index=True)
    selected.to_csv(args.output_dir / "selected_papers.csv", index=False)

    style_rows = []
    for _, row in selected.iterrows():
        style_rows.append({
            "selection_group": row.selection_group, "selection_rank": row.selection_rank,
            "paper_id": row.paper_id, "title": row.title, "decision": row.decision,
            "field": row.field, "citations": row.citations_serper, **abstract_statistics(str(row.abstract or "")),
        })
    style = pd.DataFrame(style_rows)
    style.to_csv(args.output_dir / "abstract_style_by_paper.csv", index=False)
    style.groupby("selection_group").agg(
        papers=("paper_id", "size"), abstract_words_mean=("abstract_words", "mean"),
        sentences_mean=("sentences", "mean"), words_per_sentence_mean=("words_per_sentence", "mean"),
        numeric_result_percent=("contains_numeric_result", lambda x: 100 * x.mean()),
        problem_move_percent=("contains_problem_move", lambda x: 100 * x.mean()),
        method_move_percent=("contains_method_move", lambda x: 100 * x.mean()),
        evidence_move_percent=("contains_evidence_move", lambda x: 100 * x.mean()),
    ).reset_index().to_csv(args.output_dir / "abstract_style_summary.csv", index=False)

    concern_rows = []
    evidence_rows = []
    for _, row in selected.iterrows():
        review = re.sub(r"\s+", " ", weakness_sections(str(row.reviews or ""))).lower()
        for concern, patterns in CONCERNS.items():
            matches = [match for pattern in patterns for match in re.finditer(pattern, review)]
            present = bool(matches)
            concern_rows.append({"selection_group": row.selection_group, "paper_id": row.paper_id, "concern": concern, "present": present})
            if present:
                match = min(matches, key=lambda item: item.start()); start = max(0, match.start() - 170); end = min(len(review), match.end() + 260)
                evidence_rows.append({"selection_group": row.selection_group, "paper_id": row.paper_id, "title": row.title, "concern": concern, "context": review[start:end]})
    concerns = pd.DataFrame(concern_rows)
    concerns.to_csv(args.output_dir / "reviewer_concerns_by_paper.csv", index=False)
    concerns.groupby(["selection_group", "concern"]).present.agg(["count", "sum", "mean"]).reset_index().assign(
        prevalence_percent=lambda frame: 100 * frame["mean"]
    ).to_csv(args.output_dir / "reviewer_concern_prevalence.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(args.output_dir / "reviewer_concern_contexts.csv", index=False)

    if args.skip_pdfs:
        print(f"Wrote textual ICLR audit to {args.output_dir}")
        return 0

    unique = selected.sort_values(["selection_group", "selection_rank"]).drop_duplicates("paper_id")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(paper_audit, row, args.output_dir, args.dpi): index for index, row in unique.iterrows()}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]; results[index] = future.result()
            if done % 20 == 0 or done == len(futures):
                print(f"Audited papers {done}/{len(futures)}.", flush=True)
    paper_rows = []; page_rows = []
    for index, row in unique.iterrows():
        result = results[index]
        paper_rows.append({
            "selection_group": row.selection_group, "rank": int(row.selection_rank), "paper_id": row.paper_id,
            "title": row.title, "decision": row.decision, "field": row.field, "status": result["status"],
            "full_text_words": result["words"], "visual_pages": len(result["pages"]), "source": result["source"],
        })
        for page in result["pages"]:
            page_rows.append({
                "selection_group": row.selection_group, "rank": int(row.selection_rank), "paper_id": row.paper_id,
                "title": row.title, "decision": row.decision, "field": row.field, **page,
            })
    pd.DataFrame(paper_rows).to_csv(args.output_dir / "paper_fulltext_visual_coverage.csv", index=False)
    pages = pd.DataFrame(page_rows); pages.to_csv(args.output_dir / "visual_page_index.csv", index=False)
    summary_rows = []
    for group, frame in pages.groupby("selection_group"):
        for category in (*VISUAL_PATTERNS, "other"):
            has_category = frame.categories.str.split(";").map(lambda values: category in values)
            summary_rows.append({"selection_group": group, "category": category, "pages": int(has_category.sum()), "papers": int(frame.loc[has_category, "paper_id"].nunique())})
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "visual_practice_summary.csv", index=False)
    for group, frame in pages.sort_values(["selection_group", "rank", "page"]).groupby("selection_group"):
        records = frame.to_dict("records")
        for start in range(0, len(records), 20):
            contact_sheet(records[start:start + 20], args.output_dir / "contact_sheets" / f"{group}_{start // 20 + 1:02d}.jpg", f"{group}: visual pages {start + 1}-{min(start + 20, len(records))}")
    print(f"Wrote complete ICLR audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthesize paper-, evidence-, and image-level literature audit outputs.

The input audits retain every rendered figure/table page.  This script adds
paper-balanced summaries so long appendices cannot dominate visual-practice
statistics, and records simple image-layout measurements for every page.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from audit_literature_visual_systems import VISUAL_PATTERNS, categories, contact_sheet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literature-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def image_statistics(path: str) -> dict[str, float]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((640, 900), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    white = np.all(array >= 245, axis=2)
    ink = ~white
    chroma = array.max(axis=2).astype(np.int16) - array.min(axis=2).astype(np.int16)
    colored = chroma >= 18
    if ink.any():
        ys, xs = np.where(ink)
        bbox_fraction = ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / ink.size
    else:
        bbox_fraction = 0.0
    return {
        "white_fraction": float(white.mean()),
        "ink_fraction": float(ink.mean()),
        "colored_fraction": float(colored.mean()),
        "colored_ink_fraction": float((colored & ink).sum() / max(ink.sum(), 1)),
        "ink_bbox_fraction": float(bbox_fraction),
    }


def category_matrix(pages: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    category_names = [*VISUAL_PATTERNS, "other"]
    for keys, frame in pages.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        values = frame.categories.fillna("").str.split(";")
        for category in category_names:
            row[category] = bool(values.map(lambda items: category in items).any())
        row["visual_pages"] = len(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def representative_pages(pages: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    weights = {
        "qualitative field": 3.0,
        "method / architecture": 2.0,
        "accuracy comparison": 1.5,
        "geometry / dataset": 1.0,
        "uncertainty / distribution": 0.75,
        "ablation / sensitivity": 0.75,
        "geometry fidelity": 0.75,
    }

    def score(row: pd.Series) -> float:
        items = set(str(row.categories).split(";"))
        semantic = sum(weights.get(item, 0.0) for item in items)
        if items == {"other"}:
            semantic -= 4.0
        # Large colored regions are a useful proxy for an actual diagram or
        # field panel, while semantic labels distinguish them from decoration.
        return semantic + 65.0 * float(row.colored_fraction) + 2.0 * float(row.ink_fraction)

    chosen = []
    for _, frame in pages.groupby(group_columns, dropna=False, sort=False):
        ranked = frame.copy()
        ranked["selection_score"] = ranked.apply(score, axis=1)
        ranked = ranked.sort_values(["selection_score", "page"], ascending=[False, True])
        chosen.append(ranked.iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def summarize_visuals(
    pages: pd.DataFrame,
    paper_columns: list[str],
    collection: str,
    output: Path,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pages = pages.copy()
    pages["collection"] = collection
    paths = pages.image_path.astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(image_statistics, paths))
    pages = pd.concat([pages.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    matrix = category_matrix(pages, paper_columns)
    representatives = representative_pages(pages, paper_columns)

    sheet_dir = output / "paper_balanced_contact_sheets" / collection
    sort_columns = [column for column in ("selection_group", "area", "rank", "page") if column in representatives]
    representatives = representatives.sort_values(sort_columns)
    for group, frame in representatives.groupby(
        "selection_group" if "selection_group" in representatives else "area", sort=False
    ):
        records = frame.to_dict("records")
        for start in range(0, len(records), 20):
            contact_sheet(
                records[start : start + 20],
                sheet_dir / f"{group}_{start // 20 + 1:02d}.jpg",
                f"{group}: one representative visual page per paper",
            )
    return pages, matrix, representatives


def practice_summary(matrix: pd.DataFrame, collection: str) -> pd.DataFrame:
    group = "selection_group" if "selection_group" in matrix else "area"
    rows = []
    for name, frame in matrix.groupby(group):
        for category in (*VISUAL_PATTERNS, "other"):
            rows.append(
                {
                    "collection": collection,
                    "group": name,
                    "category": category,
                    "papers": len(frame),
                    "papers_with_category": int(frame[category].sum()),
                    "paper_prevalence_percent": 100 * float(frame[category].mean()),
                }
            )
    return pd.DataFrame(rows)


def caption_practice_summary(field_root: Path) -> pd.DataFrame:
    """Measure visual practices from captions, using all selected papers as the denominator."""
    captions = pd.read_csv(field_root / "visual_caption_audit.csv")
    papers = pd.read_csv(field_root / "paper_evidence_matrix.csv")
    rows = []
    for area, selected in papers.groupby("area"):
        area_captions = captions[captions.area.eq(area)].copy()
        area_captions["derived_categories"] = area_captions.caption.fillna("").map(
            lambda value: categories(value)
        )
        for category in (*VISUAL_PATTERNS, "other"):
            matching = area_captions[
                area_captions.derived_categories.map(lambda values: category in values)
            ]
            rows.append({
                "area": area,
                "category": category,
                "selected_papers": len(selected),
                "papers_with_category": matching["rank"].nunique(),
                "caption_count": len(matching),
                "paper_prevalence_percent": 100 * matching["rank"].nunique() / len(selected),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output = args.literature_root / "synthesis"
    output.mkdir(parents=True, exist_ok=True)
    all_pages = []
    all_matrices = []
    all_representatives = []
    all_practices = []

    iclr_root = args.literature_root / "iclr2025_reviews/audit"
    iclr_path = iclr_root / "visual_page_index.csv"
    if iclr_path.exists():
        frame = pd.read_csv(iclr_path)
        pages, matrix, representatives = summarize_visuals(
            frame, ["selection_group", "paper_id", "title"], "iclr2025", output, args.workers
        )
        all_pages.append(pages)
        all_matrices.append(matrix.assign(collection="iclr2025"))
        all_representatives.append(representatives)
        all_practices.append(practice_summary(matrix, "iclr2025"))

    field_root = args.literature_root / "arxiv_field_corpus"
    field_path = field_root / "visual_page_index.csv"
    if field_path.exists():
        frame = pd.read_csv(field_path)
        pages, matrix, representatives = summarize_visuals(
            frame, ["area", "rank", "title"], "arxiv_field", output, args.workers
        )
        all_pages.append(pages)
        all_matrices.append(matrix.assign(collection="arxiv_field"))
        all_representatives.append(representatives)
        all_practices.append(practice_summary(matrix, "arxiv_field"))

    if not all_pages:
        raise FileNotFoundError("No completed visual-page audit was found.")

    pages = pd.concat(all_pages, ignore_index=True)
    pages.to_csv(output / "all_visual_page_statistics.csv", index=False)
    pd.concat(all_matrices, ignore_index=True).to_csv(output / "paper_visual_category_matrix.csv", index=False)
    pd.concat(all_representatives, ignore_index=True).to_csv(output / "representative_visual_pages.csv", index=False)
    practices = pd.concat(all_practices, ignore_index=True)
    practices.to_csv(output / "paper_level_visual_practices.csv", index=False)

    if (field_root / "visual_caption_audit.csv").exists():
        caption_practices = caption_practice_summary(field_root)
        caption_practices.to_csv(output / "caption_visual_practices.csv", index=False)

    layout = pages.groupby("collection").agg(
        visual_pages=("image_path", "size"),
        white_fraction_median=("white_fraction", "median"),
        ink_fraction_median=("ink_fraction", "median"),
        colored_ink_fraction_median=("colored_ink_fraction", "median"),
        ink_bbox_fraction_median=("ink_bbox_fraction", "median"),
    ).reset_index()
    layout.to_csv(output / "visual_layout_summary.csv", index=False)
    print(f"Audited {len(pages):,} visual pages and wrote synthesis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

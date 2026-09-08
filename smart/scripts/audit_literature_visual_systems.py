#!/usr/bin/env python3
"""Render and catalogue every accessible figure/table page in a paper corpus."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "results/final/deal_comprehensive_evaluation/literature_corpus"
USER_AGENT = "DeAL-literature-visual-audit/1.0 (mailto:parsa.vatani99@gmail.com)"

VISUAL_PATTERNS = {
    "method / architecture": (r"architect", r"framework", r"pipeline", r"overview", r"schematic", r"workflow"),
    "qualitative field": (r"pressure", r"velocity", r"temperature", r"vorticity", r"stress", r"physical field", r"solution field"),
    "geometry / dataset": (r"geometr", r"mesh", r"point cloud", r"dataset", r"sample", r"domain"),
    "ablation / sensitivity": (r"ablation", r"sensitivity", r"hyperparameter", r"component"),
    "accuracy comparison": (r"error", r"accuracy", r"comparison", r"performance", r"baseline"),
    "resolution / scaling": (r"resolution", r"grid", r"discret", r"mesh size", r"number of points", r"scal"),
    "uncertainty / distribution": (r"confidence", r"standard deviation", r"distribution", r"boxplot", r"violin", r"histogram"),
    "training / convergence": (r"training", r"validation", r"convergence", r"epoch", r"loss curve"),
    "compute / efficiency": (
        r"runtime",
        r"speedup",
        r"memory (?:usage|cost|footprint)",
        r"throughput",
        r"number of parameters",
        r"parameter count",
        r"model size",
        r"flops",
    ),
    "geometry fidelity": (r"hausdorff", r"chamfer", r"normal deviation", r"surface area", r"mesh quality"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--max-pdf-mib", type=int, default=50)
    return parser.parse_args()


def download(url: str, path: Path, max_bytes: int) -> bool:
    if path.exists() and path.stat().st_size > 1024:
        return True
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
        with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as stream:
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError("PDF too large")
                stream.write(block)
        with path.open("rb") as stream:
            valid = stream.read(5) == b"%PDF-"
        if not valid:
            path.unlink(missing_ok=True)
        return valid
    except Exception:
        path.unlink(missing_ok=True)
        return False


def safe_stem(row: pd.Series) -> str:
    identifier = str(row.openalex_id).rsplit("/", 1)[-1]
    title = re.sub(r"[^a-z0-9]+", "-", str(row.title).lower()).strip("-")[:58]
    return f"{identifier}_{title}"


def categories(text: str) -> list[str]:
    normalized = text.lower()
    found = [name for name, patterns in VISUAL_PATTERNS.items() if any(re.search(pattern, normalized) for pattern in patterns)]
    return found or ["other"]


def process(row: pd.Series, corpus: Path, dpi: int, max_bytes: int) -> dict:
    stem = safe_stem(row)
    pdf_dir = corpus / "visual_audit" / "pdfs"
    page_dir = corpus / "visual_audit" / "pages" / stem
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"
    url = str(row.full_text_source or "")
    if not url or not download(url, pdf, max_bytes):
        return {"status": "download_failed", "pages": [], "pdf": ""}
    try:
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True, timeout=30).stdout
        match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
        total_pages = int(match.group(1)) if match else 0
        pages = []
        for page in range(1, total_pages + 1):
            text = subprocess.run(
                ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
                check=True, capture_output=True, text=True, timeout=30,
            ).stdout
            if not re.search(r"\b(?:fig(?:ure)?\.?|table)\s*[a-z]?\d+", text, flags=re.IGNORECASE):
                continue
            output = page_dir / f"page_{page:03d}"
            subprocess.run(
                [
                    "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-jpeg",
                    "-jpegopt", "quality=90", "-r", str(dpi), str(pdf), str(output),
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
            )
            pages.append({"page": page, "image_path": str(output.with_suffix(".jpg")), "categories": ";".join(categories(text))})
        return {"status": "rendered", "pages": pages, "pdf": str(pdf)}
    except Exception:
        return {"status": "render_failed", "pages": [], "pdf": str(pdf)}


def contact_sheet(rows: list[dict], output: Path, title: str, columns: int = 4) -> None:
    if not rows:
        return
    thumb_width, thumb_height, header = 410, 560, 50
    count = len(rows); grid_rows = (count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, grid_rows * (thumb_height + header) + 42), "white")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    draw.text((12, 12), title, fill=(20, 45, 60), font=font)
    for index, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((thumb_width - 20, thumb_height - 20), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width + (thumb_width - image.width) // 2
        y0 = 42 + (index // columns) * (thumb_height + header)
        y = y0 + header + (thumb_height - image.height) // 2
        sheet.paste(image, (x, y))
        label = f"{row['rank']:02d} | p.{row['page']} | {row['title'][:48]}"
        draw.text((index % columns * thumb_width + 8, y0 + 8), label, fill=(20, 35, 45), font=font)
        draw.text((index % columns * thumb_width + 8, y0 + 25), row["categories"][:62], fill=(65, 85, 95), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94, subsampling=0)


def main() -> int:
    args = parse_args()
    matrix = pd.read_csv(args.corpus_dir / "paper_evidence_matrix.csv")
    matrix = matrix[matrix.full_text_status.eq("extracted") & matrix.full_text_source.notna()].copy()
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, row, args.corpus_dir, args.dpi, args.max_pdf_mib * 1024 * 1024): index
            for index, row in matrix.iterrows()
        }
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            results[index] = future.result()
            if done % 10 == 0 or done == len(futures):
                print(f"Rendered {done}/{len(futures)} papers.", flush=True)

    page_rows = []
    paper_rows = []
    for index, row in matrix.iterrows():
        result = results[index]
        paper_rows.append({
            "area": row.area, "rank": int(row["rank"]), "openalex_id": row.openalex_id,
            "title": row.title, "status": result["status"], "visual_pages": len(result["pages"]),
            "pdf_path": result["pdf"],
        })
        for page in result["pages"]:
            page_rows.append({
                "area": row.area, "rank": int(row["rank"]), "openalex_id": row.openalex_id,
                "title": row.title, **page,
            })
    audit = args.corpus_dir / "visual_audit"
    pd.DataFrame(paper_rows).to_csv(audit / "paper_visual_coverage.csv", index=False)
    page_frame = pd.DataFrame(page_rows)
    page_frame.to_csv(audit / "visual_page_index.csv", index=False)

    category_rows = []
    for area, group in page_frame.groupby("area"):
        for category in (*VISUAL_PATTERNS, "other"):
            category_rows.append({
                "area": area, "category": category,
                "visual_pages": int(group.categories.str.split(";").map(lambda values: category in values).sum()),
                "papers": int(group[group.categories.str.contains(re.escape(category), regex=True)].openalex_id.nunique()),
            })
    pd.DataFrame(category_rows).to_csv(audit / "visual_practice_summary.csv", index=False)

    contact_dir = audit / "contact_sheets"
    chunk_size = 20
    for area, group in page_frame.sort_values(["area", "rank", "page"]).groupby("area"):
        records = group.to_dict("records")
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            contact_sheet(chunk, contact_dir / f"{area}_{start // chunk_size + 1:02d}.jpg", f"{area}: visual pages {start + 1}-{start + len(chunk)}")
    print(f"Wrote visual audit to {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

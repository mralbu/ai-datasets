"""Images for the paper pages, kept in the repository (both are small and change rarely).

    uv run scripts/papers.py pages [ids…]    first page of each paper -> public/papers/<id>.webp
    uv run scripts/papers.py figures         data for the figures drawn from site datasets
                                             -> src/paper-figures/<name>.json

pages: downloads the PDF (arXiv for papers with an `arxiv` id, else the `pdf` field of
papers/<id>.yaml), renders page 1 with poppler's pdftoppm and saves a small WebP. Papers that
already have an image are skipped unless --force. A paper with no reachable PDF just has no image.

figures: fetches whole datasets from the Hugging Face dataset viewer and computes what a figure
shows (Fisher's discriminant on all 150 iris flowers, Yule's AR(2) fit on yearly sunspot means).
The figures themselves are Astro components in src/components/figures/.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers"
PAGES_OUT = ROOT / "public" / "papers"
FIGURES_OUT = ROOT / "src" / "paper-figures"
PAGE_WIDTH = 520  # px; shown at up to 260 px wide, so sharp on 2x screens
VIEWER = "https://datasets-server.huggingface.co"


def pdf_url(meta: dict) -> str | None:
    if meta.get("arxiv"):
        return f"https://arxiv.org/pdf/{meta['arxiv']}"
    return meta.get("pdf")


def first_page(client: httpx.Client, pid: str, url: str) -> Path:
    r = client.get(url)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise RuntimeError(f"not a PDF ({r.headers.get('content-type')})")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "paper.pdf"
        src.write_bytes(r.content)
        subprocess.run(
            ["pdftoppm", "-f", "1", "-l", "1", "-png", "-scale-to-x", str(PAGE_WIDTH), "-scale-to-y", "-1",
             str(src), str(Path(tmp) / "page")],
            check=True,
        )
        png = next(Path(tmp).glob("page*.png"))
        img = Image.open(png).convert("RGB")
    out = PAGES_OUT / f"{pid}.webp"
    img.save(out, "WEBP", quality=72, method=6)
    return out


def cmd_pages(ids: list[str], force: bool) -> int:
    PAGES_OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(PAPERS.glob("*.yaml"))
    if ids:
        files = [f for f in files if f.stem in ids]
    failed = 0
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "ai-datasets-site/0.1"}) as client:
        for f in files:
            pid = f.stem
            out = PAGES_OUT / f"{pid}.webp"
            if out.exists() and not force:
                print(f"- {pid}: exists")
                continue
            url = pdf_url(yaml.safe_load(f.read_text()))
            if not url:
                print(f"- {pid}: no PDF source, skipped")
                continue
            try:
                first_page(client, pid, url)
                print(f"- {pid}: {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
            except Exception as e:  # keep going: one unreachable PDF should not stop the rest
                failed += 1
                print(f"- {pid}: FAILED {url}: {e}", file=sys.stderr)
    return 1 if failed else 0


# ---- figures -------------------------------------------------------------------------------

def viewer_rows(client: httpx.Client, dataset: str, config: str, split: str) -> pl.DataFrame:
    """All rows of a small split, page by page."""
    rows, offset = [], 0
    while True:
        r = client.get(f"{VIEWER}/rows", params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
        r.raise_for_status()
        page = r.json()
        rows += [x["row"] for x in page["rows"]]
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"]:
            return pl.DataFrame(rows)


def viewer_range(client: httpx.Client, dataset: str, config: str, split: str, n: int) -> pl.DataFrame:
    """The first n rows of a split, fetched 100 at a time two requests at a time (the viewer rate-limits)."""
    def page(offset: int) -> list[dict]:
        for attempt in range(8):
            try:
                r = client.get(f"{VIEWER}/rows", params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
                if r.status_code == 200:
                    return [x["row"] for x in r.json()["rows"]]
            except httpx.TransportError:
                pass  # dropped connection: retry like a 429
            time.sleep(5 * 2 ** min(attempt, 4))
        raise RuntimeError(f"rows at offset {offset} failed after retries")
    cache = ROOT / ".cache" / f"{dataset.replace('/', '__')}-{config}-{split}-{n}.json"
    if cache.exists():
        return pl.DataFrame(json.loads(cache.read_text()))
    with ThreadPoolExecutor(2) as pool:
        pages = list(pool.map(page, range(0, n, 100)))
    rows = [row for p in pages for row in p][:n]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(rows))
    return pl.DataFrame(rows)


def fisher(client: httpx.Client) -> dict:
    """Fisher's linear discriminant on all 150 flowers: the two discriminant axes of the
    three species (classic multi-class LDA), as Fisher's single function separates them."""
    df = viewer_rows(client, "scikit-learn/iris", "default", "train")
    feats = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df.select(feats).to_numpy().astype(float)
    y = df["Species"].to_list()
    classes = sorted(set(y))
    mu = X.mean(0)
    Sw = np.zeros((4, 4))
    Sb = np.zeros((4, 4))
    for c in classes:
        Xc = X[[i for i, v in enumerate(y) if v == c]]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    vals, vecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-vals.real)
    W = vecs[:, order[:2]].real
    W /= np.linalg.norm(W, axis=0)
    Z = (X - mu) @ W
    # Orient the first axis so setosa is on the left.
    setosa = [i for i, v in enumerate(y) if v == classes[0]]
    if Z[setosa, 0].mean() > 0:
        Z[:, 0] *= -1
        W[:, 0] *= -1
    share = vals.real[order[:2]] / vals.real[order].sum()
    return {
        "features": feats,
        "weights": [[round(float(v), 4) for v in W[:, k]] for k in range(2)],
        "share": [round(float(s), 4) for s in share],
        "points": [
            {"ld1": round(float(Z[i, 0]), 3), "ld2": round(float(Z[i, 1]), 3), "species": y[i].removeprefix("Iris-"),
             "petal_length": float(X[i, 2]), "petal_width": float(X[i, 3])}
            for i in range(len(y))
        ],
    }


def yule(client: httpx.Client) -> dict:
    """Yearly mean sunspot numbers from 1818 (the start of the daily record) to 1924, the end of
    Yule's data, with his AR(2) model:
    x_t = a + b1 x_{t-1} + b2 x_{t-2} + e_t, fitted by least squares, and its one-step predictions."""
    # Daily rows from 1818-01-01; only the years up to 1924 are needed, so page through the
    # dataset viewer (its Parquet files are served from a CDN this project does not need).
    days = (date(1925, 1, 1) - date(1818, 1, 1)).days
    df = viewer_range(client, "juliensimon/silso-sunspot-number", "default", "train", days)
    yearly = (
        df.select(pl.col("date").str.to_datetime().dt.year().alias("year"), pl.col("sunspot_number").cast(pl.Float64).alias("v"))
        .filter(pl.col("v") >= 0)
        .group_by("year").agg(pl.col("v").mean(), pl.len().alias("days"))
        .filter((pl.col("year") <= 1924) & (pl.col("days") >= 180))
        .sort("year")
    )
    # Years with too few observed days (1835–1837) are left out; the model is fitted only on
    # years whose two previous years are both present, so it never spans a gap.
    by_year = dict(zip(yearly["year"].to_list(), yearly["v"].to_list()))
    years = sorted(by_year)
    fit_years = [t for t in years if t - 1 in by_year and t - 2 in by_year]
    A = np.array([[1.0, by_year[t - 1], by_year[t - 2]] for t in fit_years])
    target = np.array([by_year[t] for t in fit_years])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = dict(zip(fit_years, A @ coef))
    resid = target - A @ coef
    # The damped "pendulum": the period implied by the AR(2) roots.
    roots = np.roots([1, -coef[1], -coef[2]])
    period = float(2 * np.pi / abs(np.angle(roots[0]))) if np.iscomplex(roots[0]) else None
    return {
        "source": "juliensimon/silso-sunspot-number: daily values averaged per year, 1818–1924",
        "coef": {"a": round(float(coef[0]), 2), "b1": round(float(coef[1]), 3), "b2": round(float(coef[2]), 3)},
        "period": round(period, 1) if period else None,
        "r2": round(float(1 - resid.var() / target.var()), 3),
        "missing": sorted(set(range(years[0], years[-1] + 1)) - set(years)),
        "points": [{"year": t, "actual": round(float(by_year[t]), 1), "fit": round(float(pred[t]), 1) if t in pred else None} for t in years],
    }


FIGURES = {"fisher-1936": fisher, "yule-1927": yule}


def cmd_figures(names: list[str]) -> int:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, fn in FIGURES.items():
            if names and name not in names:
                continue
            data = fn(client)
            (FIGURES_OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
            print(f"- {name}: {len(data['points'])} points")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("ids", nargs="*")
    p.add_argument("--force", action="store_true")
    f = sub.add_parser("figures")
    f.add_argument("names", nargs="*")
    a = ap.parse_args()
    return cmd_pages(a.ids, a.force) if a.cmd == "pages" else cmd_figures(a.names)


if __name__ == "__main__":
    sys.exit(main())

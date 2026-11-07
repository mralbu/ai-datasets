"""Fetch a small, reproducible sample of every dataset in ``datasets/*.yaml``.

Rows come from the Hugging Face dataset viewer API, so nothing large is downloaded.

Outputs (both git-ignored, regenerated on every build):
  public/samples/<id>/<row_idx>.<ext>   images referenced by the samples
  src/generated/<id>.json               features, sampled rows, optional chart spec

Usage:
  uv run scripts/sample.py            # all datasets
  uv run scripts/sample.py mnist iris # only these ids
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import httpx
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "datasets"
OUT_JSON = ROOT / "src" / "generated"
OUT_MEDIA = ROOT / "public" / "samples"

API = "https://datasets-server.huggingface.co"
PAGE = 100  # max rows per /rows request
FETCH_ALL_LIMIT = 5_000  # small tabular datasets are fetched whole (for charts)


def get(client: httpx.Client, url: str, **params) -> httpx.Response:
    """GET with simple exponential backoff (the viewer API rate-limits)."""
    for attempt in range(5):
        try:
            r = client.get(url, params=params or None)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if attempt == 4:
                raise
            wait = 2**attempt
            print(f"    retry in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_rows(client: httpx.Client, src: dict, offset: int, length: int) -> dict:
    return get(
        client,
        f"{API}/rows",
        dataset=src["hf"],
        config=src.get("config", "default"),
        split=src.get("split", "train"),
        offset=offset,
        length=length,
    ).json()


def sample_dataset(client: httpx.Client, meta: dict) -> dict:
    ds_id = meta["id"]
    src = meta["source"]
    n = int(src.get("n", 48))
    rng = random.Random(src.get("seed", 0))

    # One cheap request to learn the schema and the total row count.
    head = fetch_rows(client, src, 0, 1)
    total = head["num_rows_total"]
    features = [{"name": f["name"], "type": f["type"]} for f in head["features"]]

    all_rows: list[dict] | None = None
    if src.get("fetch_all") and total <= FETCH_ALL_LIMIT:
        all_rows = []
        for off in range(0, total, PAGE):
            all_rows += fetch_rows(client, src, off, PAGE)["rows"]
        picked = sorted(rng.sample(range(total), min(n, total)))
        rows = [all_rows[i] for i in picked]
    else:
        # Random offsets, one row each: many datasets are sorted by label
        # (IMDB is all negatives, then all positives), so a contiguous slice lies.
        picked = sorted(rng.sample(range(total), min(n, total)))
        rows = []
        for i in picked:
            rows += fetch_rows(client, src, i, 1)["rows"]

    # Fetched in index order; present in a (seeded) random order so the first
    # rows on the page are not, say, all from one class.
    rng.shuffle(rows)

    media_dir = OUT_MEDIA / ds_id
    image_cols = [f["name"] for f in features if f["type"].get("_type") == "Image"]
    if image_cols:
        media_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        for col in image_cols:
            cell = r["row"].get(col)
            if not cell:
                continue
            ext = Path(urlparse(cell["src"]).path).suffix or ".jpg"
            name = f"{r['row_idx']}-{col}{ext}"
            (media_dir / name).write_bytes(get(client, cell["src"]).content)
            # Signed URLs expire; point at the local copy instead.
            r["row"][col] = {
                "src": f"samples/{ds_id}/{name}",
                "width": cell.get("width"),
                "height": cell.get("height"),
            }

    out = {
        "id": ds_id,
        "source": {k: src.get(k) for k in ("hf", "config", "split")},
        "num_rows_total": total,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": features,
        "rows": [
            {"row_idx": r["row_idx"], "row": r["row"], "truncated": r.get("truncated_cells", [])}
            for r in rows
        ],
    }

    chart_cfg = (meta.get("display") or {}).get("chart")
    if chart_cfg and all_rows:
        out["chart"] = build_chart(all_rows, chart_cfg)
    return out


def build_chart(all_rows: list[dict], cfg: dict) -> dict:
    """A small Altair chart over the whole (small) dataset, as a Vega-Lite spec."""
    df = pl.DataFrame([r["row"] for r in all_rows])
    cols = [c for c in (cfg["x"], cfg["y"], cfg.get("color")) if c]
    df = df.select(cols)
    enc = {
        "x": alt.X(cfg["x"], type="quantitative", scale=alt.Scale(zero=False)),
        "y": alt.Y(cfg["y"], type="quantitative", scale=alt.Scale(zero=False)),
        "tooltip": [f"{cfg['x']}:Q", f"{cfg['y']}:Q"] + ([f"{cfg['color']}:N"] if cfg.get("color") else []),
    }
    if cfg.get("color"):
        enc["color"] = alt.Color(cfg["color"], type="nominal")
    chart = (
        alt.Chart(alt.Data(values=df.to_dicts()))
        .mark_circle(size=60, opacity=0.75)
        .encode(**enc)
        .properties(width="container", height=320, title=cfg.get("title", ""))
    )
    return chart.to_dict()


def main(argv: list[str]) -> int:
    only = set(argv)
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    failures = []
    headers = {"User-Agent": "ai-datasets-sampler/0.1 (static educational site)"}
    with httpx.Client(timeout=60, headers=headers, follow_redirects=True) as client:
        for path in sorted(CATALOG.glob("*.yaml")):
            meta = yaml.safe_load(path.read_text())
            meta["id"] = path.stem
            if only and meta["id"] not in only:
                continue
            if not meta.get("source", {}).get("hf"):
                print(f"- {meta['id']}: no source, skipped")
                continue
            print(f"- {meta['id']}: {meta['source']['hf']}")
            try:
                data = sample_dataset(client, meta)
            except Exception as e:  # keep going; the site renders without samples
                print(f"    FAILED: {e}", file=sys.stderr)
                failures.append(meta["id"])
                continue
            (OUT_JSON / f"{meta['id']}.json").write_text(json.dumps(data, ensure_ascii=False))
            print(f"    {len(data['rows'])} rows of {data['num_rows_total']:,}")
    if failures:
        print(f"failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

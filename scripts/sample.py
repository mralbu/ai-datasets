"""Fetch a small, reproducible sample of every dataset in ``datasets/*.yaml``.

Rows come from the Hugging Face dataset viewer API, so nothing large is downloaded.

Samples live in a snapshot folder (default ``.samples/``; in CI it is a worktree of
the ``samples`` branch), so they are fetched once and reused:

  .samples/manifest.json            fingerprint, upstream revision, fetch time per dataset
  .samples/generated/<id>.json      features, sampled rows, optional chart spec
  .samples/samples/<id>/<file>      images and audio referenced by the rows

A dataset is re-fetched only when its fingerprint (its ``source`` block, its chart
config and SAMPLER_VERSION) changes. Datasets removed from the catalog are pruned.
The snapshot is then copied to where the site reads it (src/generated, public/samples).

Usage:
  uv run scripts/sample.py              # fetch what is missing or changed, then install
  uv run scripts/sample.py mnist iris   # only consider these ids (no pruning)
  uv run scripts/sample.py --force mnist  # re-fetch even if unchanged
  uv run scripts/sample.py --strict     # exit 1 if any dataset fails (used in CI)
  uv run scripts/sample.py --check      # only check every source still resolves
  uv run scripts/sample.py --snapshot DIR
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import base64
import io
import tarfile

import altair as alt
import httpx
import numpy as np
import polars as pl
import yaml
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "datasets"
SITE_JSON = ROOT / "src" / "generated"
SITE_MEDIA = ROOT / "public" / "samples"
DEFAULT_SNAPSHOT = ROOT / ".samples"

# Bump to re-fetch everything after a change to how samples are produced.
SAMPLER_VERSION = 1

API = "https://datasets-server.huggingface.co"
PAGE = 100  # max rows per /rows request
FETCH_ALL_LIMIT = 5_000  # small tabular datasets are fetched whole (for charts)
WINDOW_LIMIT = 10_000  # most consecutive rows a time-series window may fetch (source.window)
WINDOW_ROWS_KEPT = 12  # rows of a window kept as rows (raw JSON); the whole window is kept as columns
ROWS_PER_PAGE_USED = 8  # rows taken per fetched page; lower it (source.rows_per_page) for datasets sorted by a many-valued label
PAUSE = 0.15  # seconds between requests, to stay under the rate limit


def get(client: httpx.Client, url: str, **params) -> httpx.Response:
    """GET with backoff. The viewer API rate-limits bursts (429) and sometimes 5xx."""
    attempts = 8
    for attempt in range(attempts):
        time.sleep(PAUSE)
        try:
            r = client.get(url, params=params or None)
        except httpx.TransportError as e:
            reason = f"{type(e).__name__}"
            retry_after = None
        else:
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r
            reason = f"HTTP {r.status_code}"
            retry_after = r.headers.get("retry-after")
        if attempt == attempts - 1:
            raise RuntimeError(f"giving up after {attempts} attempts ({reason})")
        wait = min(60.0, 2.0 ** (attempt + 1)) + random.random()
        if retry_after and retry_after.isdigit():
            wait = max(wait, float(retry_after))
        print(f"    {reason}, retry in {wait:.0f}s", file=sys.stderr)
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


def long_enough(row: dict, src: dict) -> bool:
    """source.min_length: {field: n} drops rows whose field is shorter (e.g. blank lines)."""
    return all(len(str(row["row"].get(f) or "").strip()) >= n for f, n in (src.get("min_length") or {}).items())


def unpack(rows: list[dict], src: dict) -> list[dict]:
    """source.unpack: {column, files: {name: path}} reads files out of a .tar.gz binary column.

    Terminal-Bench packs each task (instruction, Dockerfile, solution, tests) as one archive;
    this turns the files worth showing into ordinary text columns. Members are only read, never
    extracted to disk. A file missing from an archive becomes None (pair with min_length to skip it).
    """
    spec = src.get("unpack")
    if not spec:
        return rows
    for r in rows:
        cell = r["row"].get(spec["column"])
        found: dict[str, str] = {}
        if isinstance(cell, str) and cell:  # the viewer returns binary cells as base64
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(cell)), mode="r:gz") as tar:
                members = {m.name.removeprefix("./"): m for m in tar.getmembers() if m.isfile()}
                for name, path in spec["files"].items():
                    m = members.get(path)
                    if m is not None and m.size <= 200_000:
                        found[name] = tar.extractfile(m).read().decode("utf-8", errors="replace")
        for name in spec["files"]:
            r["row"][name] = found.get(name)
    return rows


def stratified_sample(client: httpx.Client, src: dict, total: int, n: int, rng: random.Random) -> list[dict]:
    """Sample n rows with few requests, without trusting the dataset's order.

    Many datasets are sorted by label (IMDB is all negatives, then all
    positives), so one contiguous slice would mislead. Instead split the index
    range into k equal strata, fetch one random page inside each, and take
    about n/k random rows from every page. k pages instead of n single-row calls.
    """
    n = min(n, total)
    per_page = int(src.get("rows_per_page", ROWS_PER_PAGE_USED))
    # source.page: fewer rows per request, for rows that are megabytes each (NYU's raw arrays).
    page_len = int(src.get("page", PAGE))
    k = max(1, min(-(-n // per_page), total // page_len or 1))
    bounds = [round(total * j / k) for j in range(k + 1)]
    rows: list[dict] = []
    for j in range(k):
        lo, hi = bounds[j], bounds[j + 1]
        length = min(page_len, hi - lo)
        offset = rng.randint(lo, hi - length)
        page = unpack(fetch_rows(client, src, offset, length)["rows"], src)
        # Very large cells (long texts, big grids) come back truncated; avoid them.
        page = [r for r in page if not r.get("truncated_cells")] or page
        page = [r for r in page if long_enough(r, src)] or page
        want = n // k + (1 if j < n % k else 0)
        rows += rng.sample(page, min(want, len(page)))
    return rows


def round_sig(v, digits: int = 5):
    """Round floats to a few significant digits: series are drawn, not computed with."""
    if isinstance(v, float):
        return float(f"{v:.{digits}g}")
    if isinstance(v, list):
        return [round_sig(x, digits) for x in v]
    return v


def trim_lists(rows: list[dict], src: dict, rng: random.Random) -> None:
    """source.max_points: keep a contiguous stretch of N points of each list column (a series).

    Lists of the same length (timestamps and values) are cut at the same place. source.trim
    picks where: "end" keeps the last N points (what a forecaster sees last), "random" a
    seeded random stretch. The original length is kept per column in r["points"].
    """
    n = src.get("max_points")
    if not n:
        return
    for r in rows:
        starts: dict[int, int] = {}
        points = {}
        for col, v in r["row"].items():
            if not isinstance(v, list) or len(v) <= n:
                continue
            if len(v) not in starts:
                starts[len(v)] = len(v) - n if src.get("trim", "random") == "end" else rng.randint(0, len(v) - n)
            a = starts[len(v)]
            points[col] = {"total": len(v), "from": a}
            r["row"][col] = v[a:a + n]
        if points:
            r["points"] = points


def sample_window(client: httpx.Client, src: dict, rng: random.Random) -> dict:
    """source.window: N consecutive rows of a long-format series (one row per time step).

    A random sample of rows would scatter timestamps; a series needs a contiguous stretch.
    The stretch starts at window.start, or at a seeded random row inside window.within
    (default: the whole split). It comes back as columns, plus the first few rows as rows.
    """
    w = src["window"]
    head = fetch_rows(client, src, 0, 1)
    total = head["num_rows_total"]
    features = [{"name": f["name"], "type": f["type"]} for f in head["features"]]
    lo, hi = w.get("within") or [0, total]
    hi = min(hi, total)
    n = min(int(w["n"]), hi - lo, WINDOW_LIMIT)
    start = w["start"] if "start" in w else rng.randint(lo, hi - n)
    rows: list[dict] = []
    for off in range(start, start + n, PAGE):
        rows += fetch_rows(client, src, off, min(PAGE, start + n - off))["rows"]
    rows.sort(key=lambda r: r["row_idx"])
    drop = set(src.get("drop") or [])
    cols = [f["name"] for f in features if f["name"] not in drop]
    columns = {c: round_sig([r["row"].get(c) for r in rows]) for c in cols}
    return {
        "total": total,
        "features": features,
        "rows": rows[:WINDOW_ROWS_KEPT],
        "all_rows": None,
        "window": {"start": start, "n": len(rows), "columns": columns},
    }


def sample_part(client: httpx.Client, src: dict, n: int, rng: random.Random) -> dict:
    """Sample one (config, split) of a dataset."""
    if src.get("window"):
        return sample_window(client, src, rng)
    head = fetch_rows(client, src, 0, 1)  # cheap: schema + total row count
    total = head["num_rows_total"]
    features = [{"name": f["name"], "type": f["type"]} for f in head["features"]]
    features += [{"name": name, "type": {"dtype": "string", "_type": "Value"}}
                 for name in (src.get("unpack") or {}).get("files", {})]
    all_rows: list[dict] | None = None
    if src.get("fetch_all") and total <= FETCH_ALL_LIMIT:
        all_rows = []
        for off in range(0, total, PAGE):
            all_rows += unpack(fetch_rows(client, src, off, PAGE)["rows"], src)
        pool = [r for r in all_rows if long_enough(r, src)] or all_rows
        rows = rng.sample(pool, min(n, len(pool)))
    else:
        rows = stratified_sample(client, src, total, n, rng)
    # Fetched in index order; present in a seeded random order so the first
    # rows on the page are not, say, all from one class.
    rng.shuffle(rows)
    trim_lists(rows, src, rng)
    if src.get("max_points"):
        for r in rows:
            r["row"] = {k: round_sig(v) if isinstance(v, list) else v for k, v in r["row"].items()}
    return {"total": total, "features": features, "rows": rows, "all_rows": all_rows}


def shrink(data: bytes, max_px: int) -> bytes:
    """Downscale a large photo so its longest side is max_px (JPEG). Small images pass through."""
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_px:
        return data
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    out = io.BytesIO()
    img.convert("RGB").save(out, "JPEG", quality=85, optimize=True)
    return out.getvalue()


def to_png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def rasterize(client: httpx.Client, cell, how: str, max_px: int | None) -> tuple[bytes, str, dict]:
    """source.dense: turn a per-pixel annotation into a small image the site can draw and read back.

    The cell is a viewer Image (a mask image, like Cityscapes') or a raw array (NYU's Array3D).
      image    float array, channels first, 0..1                    -> JPEG photo
      labels   class id per pixel (-1 = void)                       -> grayscale PNG, id as the value, 255 = void
      mask     a black-and-white mask image (ISIC's lesions)         -> labels with 0 = outside, 1 = inside
      depth    meters per pixel, channels first (0 = no reading)    -> grayscale PNG, 1..255 spans 0..max, 0 = missing
      normals  unit vectors, channels first, -1..1                  -> RGB PNG, the usual (n + 1) / 2 coloring
    Returns the file bytes, its extension and metadata for the row (size; class shares; depth range).
    """
    if isinstance(cell, dict):  # an image from the viewer: download it
        a = np.array(Image.open(io.BytesIO(get(client, cell["src"]).content)))
    else:
        a = np.asarray(cell, dtype=np.float32)
    meta: dict = {}
    if how == "image":
        if a.dtype != np.uint8:
            a = (np.clip(a.transpose(1, 2, 0) if a.shape[0] == 3 else a, 0, 1) * 255).round().astype(np.uint8)
        img = Image.fromarray(a)
        if max_px and max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        out = io.BytesIO()
        img.convert("RGB").save(out, "JPEG", quality=88, optimize=True)
        data, ext = out.getvalue(), ".jpg"
    elif how in ("labels", "mask"):
        if how == "mask":
            a = (a[..., 0] if a.ndim == 3 else a) > 127
        ids = np.where(a < 0, 255, a).astype(np.uint8)
        img = Image.fromarray(ids, "L")
        if isinstance(cell, dict):
            # Mask images come back from the viewer as JPEG, which smears ids along edges.
            # A mode filter restores clean regions before the (nearest-neighbor) downscale.
            img = img.filter(ImageFilter.ModeFilter(5)).filter(ImageFilter.ModeFilter(5))
        if max_px and max(img.size) > max_px:
            w, h = img.size
            f = max_px / max(w, h)
            img = img.resize((round(w * f), round(h * f)), Image.NEAREST)
        v, c = np.unique(np.array(img), return_counts=True)
        meta["classes"] = {str(int(i)): round(float(n) / c.sum(), 4) for i, n in zip(v, c) if i != 255}
        data, ext = to_png(img), ".png"
    elif how == "depth":
        d = a[0] if a.ndim == 3 else a
        top = float(d.max()) or 1.0
        q = np.where(d > 0, 1 + np.round(d / top * 254), 0).astype(np.uint8)
        img = Image.fromarray(q, "L")
        meta |= {"min": round(float(d[d > 0].min()), 3) if (d > 0).any() else 0, "max": round(top, 3)}
        data, ext = to_png(img), ".png"
    elif how == "normals":
        n = a.transpose(1, 2, 0) if a.shape[0] == 3 else a
        img = Image.fromarray(np.round((np.clip(n, -1, 1) + 1) / 2 * 255).astype(np.uint8), "RGB")
        data, ext = to_png(img), ".png"
    else:
        raise ValueError(f"unknown dense kind {how!r}")
    meta |= {"width": img.size[0], "height": img.size[1]}
    return data, ext, meta


def save_dense(client: httpx.Client, media_dir: Path, ds_id: str, tag: str, features: list[dict], rows: list[dict],
               dense: dict[str, str], max_px: int | None) -> None:
    """Rasterize the source.dense columns of every row (see rasterize) and point the rows at the files."""
    media_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        for col, how in dense.items():
            data, ext, meta = rasterize(client, r["row"][col], how, max_px)
            name = f"{tag}{r['row_idx']}-{col}{ext}"
            (media_dir / name).write_bytes(data)
            r["row"][col] = {"src": f"samples/{ds_id}/{name}"} | meta
    for f in features:
        if f["name"] in dense:  # now a file, not an array or a viewer image: keep save_media off it
            f["type"] = {"_type": "Dense", "kind": dense[f["name"]], "was": f["type"].get("_type")}


def save_media(client: httpx.Client, media_dir: Path, ds_id: str, tag: str, features: list[dict], rows: list[dict],
               max_px: int | None = None) -> None:
    """Download images/audio next to the site; signed viewer URLs expire.

    With max_px, large images are downscaled; width/height keep the original size,
    because annotations (polygons, boxes) are in original pixel coordinates."""
    media_cols = {f["name"]: f["type"].get("_type") for f in features if f["type"].get("_type") in ("Image", "Audio")}
    if not media_cols:
        return
    media_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        for col, kind in media_cols.items():
            cell = r["row"].get(col)
            if isinstance(cell, list):  # Audio cells are a list of encodings
                cell = cell[0] if cell else None
            if not cell:
                continue
            ext = Path(urlparse(cell["src"]).path).suffix or (".jpg" if kind == "Image" else ".wav")
            data = get(client, cell["src"]).content
            if kind == "Image" and max_px:
                small = shrink(data, max_px)
                if small is not data:
                    data, ext = small, ".jpg"
            name = f"{tag}{r['row_idx']}-{col}{ext}"
            (media_dir / name).write_bytes(data)
            local = {"src": f"samples/{ds_id}/{name}"}
            if kind == "Image":
                local |= {"width": cell.get("width"), "height": cell.get("height")}
            else:
                local |= {"type": cell.get("type")}
            r["row"][col] = local


def sample_dataset(client: httpx.Client, meta: dict, media_dir: Path) -> dict:
    ds_id = meta["id"]
    src = meta["source"]
    n = int(src.get("n", 48))
    rng = random.Random(src.get("seed", 0))

    # A benchmark suite like GLUE is several configs; sample a few rows of each.
    configs = src.get("configs") or [{"config": src.get("config", "default"), "split": src.get("split", "train")}]
    configs = [c if isinstance(c, dict) else {"config": c} for c in configs]
    multi = len(configs) > 1

    parts, per_part_rows = [], []
    for c in configs:
        part_src = src | {"config": c["config"], "split": c.get("split", src.get("split", "train"))}
        part = sample_part(client, part_src, n, rng)
        # source.drop: columns not worth keeping (ADE20K's per-object mask images, say).
        drop = set(src.get("drop") or [])
        if drop:
            part["features"] = [f for f in part["features"] if f["name"] not in drop]
            for r in part["rows"]:
                r["row"] = {k: v for k, v in r["row"].items() if k not in drop}
        tag = f"{c['config']}-" if multi else ""
        if src.get("dense"):
            save_dense(client, media_dir, ds_id, tag, part["features"], part["rows"], src["dense"], src.get("max_image_px"))
        save_media(client, media_dir, ds_id, tag, part["features"], part["rows"], src.get("max_image_px"))
        parts.append({
            "config": part_src["config"],
            "split": part_src["split"],
            "num_rows_total": part["total"],
            "features": part["features"],
            "all_rows": part["all_rows"],
            "window": part.get("window"),
        })
        per_part_rows.append([
            {"row_idx": r["row_idx"], "config": part_src["config"], "row": r["row"],
             "truncated": r.get("truncated_cells", [])} | ({"points": r["points"]} if r.get("points") else {})
            for r in part["rows"]
        ])

    # Interleave configs so the first rows on the page show one of each.
    rows = [r for group in zip_longest(*per_part_rows) for r in group if r]

    out = {
        "id": ds_id,
        "source": {"hf": src["hf"], "config": parts[0]["config"], "split": parts[0]["split"]},
        "num_rows_total": sum(p["num_rows_total"] for p in parts),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": parts[0]["features"],
        "parts": [{k: v for k, v in p.items() if k not in ("all_rows", "window")} for p in parts],
        "rows": rows,
    }
    if parts[0]["window"]:
        out["window"] = parts[0]["window"]

    chart_cfg = (meta.get("display") or {}).get("chart")
    if chart_cfg and parts[0]["all_rows"]:
        out["chart"] = build_chart(parts[0]["all_rows"], chart_cfg)
    return out


def build_chart(all_rows: list[dict], cfg: dict) -> dict:
    """A small Altair chart over the whole (small) dataset, as a Vega-Lite spec.

    type "scatter": x vs y, colored by color.
    type "rate":    mean of y (e.g. a 0/1 target) for each x, split by color.
    """
    df = pl.DataFrame([r["row"] for r in all_rows])
    x, y, color = cfg["x"], cfg["y"], cfg.get("color")
    title = cfg.get("title", "")
    if cfg.get("type", "scatter") == "rate":
        keys = [k for k in (x, color) if k]
        agg = (
            df.group_by(keys)
            .agg(pl.col(y).mean().alias("rate"), pl.len().alias("count"))
            .sort(keys)
        )
        enc = {
            "x": alt.X(f"{x}:N", title=x, axis=alt.Axis(labelAngle=0)),
            "y": alt.Y("rate:Q", title=cfg.get("y_title", f"mean {y}"), axis=alt.Axis(format="%")),
            "tooltip": [f"{k}:N" for k in keys] + [alt.Tooltip("rate:Q", format=".0%"), "count:Q"],
        }
        if color:
            enc |= {"color": alt.Color(f"{color}:N", legend=alt.Legend(orient="top")), "xOffset": alt.XOffset(f"{color}:N")}
        chart = alt.Chart(alt.Data(values=agg.to_dicts())).mark_bar().encode(**enc)
    else:
        df = df.select([c for c in (x, y, color) if c])
        enc = {
            "x": alt.X(f"{x}:Q", scale=alt.Scale(zero=False)),
            "y": alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
            "tooltip": [f"{x}:Q", f"{y}:Q"] + ([f"{color}:N"] if color else []),
        }
        if color:
            enc["color"] = alt.Color(f"{color}:N", legend=alt.Legend(orient="top"))
        chart = alt.Chart(alt.Data(values=df.to_dicts())).mark_circle(size=60, opacity=0.75).encode(**enc)
    return chart.properties(width="container", height=320, title=title).to_dict()


SNAPSHOT_README = """# Sample snapshot (generated, do not edit)

This branch holds the dataset samples shown on the site. It is written by
`scripts/sample.py` in the deploy workflow on `main`; do not open pull requests
against it.

- `manifest.json`: per dataset, the fingerprint of its config, the Hugging Face
  revision it was sampled from and when.
- `generated/<id>.json`: sampled rows.
- `samples/<id>/`: images and audio those rows reference.

To change a sample, edit `datasets/<id>.yaml` on `main` (or run the deploy
workflow with "force refetch"). Samples remain under each dataset's own license.
"""


def fingerprint(meta: dict) -> str:
    """What the sample depends on. Display tweaks (scale, labels) don't need a re-fetch."""
    key = {
        "v": SAMPLER_VERSION,
        "source": meta.get("source"),
        "chart": (meta.get("display") or {}).get("chart"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def upstream_revision(client: httpx.Client, hf_id: str) -> str | None:
    """The dataset repo's commit on the Hub, to record which version was sampled."""
    try:
        return get(client, f"https://huggingface.co/api/datasets/{hf_id}").json().get("sha")
    except Exception:
        return None


def load_catalog() -> tuple[dict[str, dict], list[str]]:
    catalog, bad = {}, []
    for path in sorted(CATALOG.glob("*.yaml")):
        try:
            meta = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            print(f"- {path.stem}: invalid YAML: {e}", file=sys.stderr)
            bad.append(path.stem)
            continue
        meta["id"] = path.stem
        catalog[path.stem] = meta
    return catalog, bad


def install(snap: Path) -> None:
    """Copy the snapshot to where Astro reads it."""
    for src, dst in ((snap / "generated", SITE_JSON), (snap / "samples", SITE_MEDIA)):
        shutil.rmtree(dst, ignore_errors=True)
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)


def check_sources(client: httpx.Client, catalog: dict[str, dict]) -> list[str]:
    """Cheap health check: can the viewer still serve one row of every source?"""
    broken = []
    for ds_id, meta in catalog.items():
        src = meta.get("source", {})
        configs = src.get("configs") or [{"config": src.get("config", "default"), "split": src.get("split", "train")}]
        for c in configs:
            c = c if isinstance(c, dict) else {"config": c}
            part = src | {"config": c["config"], "split": c.get("split", src.get("split", "train"))}
            try:
                fetch_rows(client, part, 0, 1)
                print(f"  ok      {ds_id} ({part['config']}/{part['split']})")
            except Exception as e:
                print(f"  BROKEN  {ds_id} ({part['config']}/{part['split']}): {e}")
                broken.append(ds_id)
    return broken


def warn(msg: str) -> None:
    print(f"::warning::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"warning: {msg}", file=sys.stderr)


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    snap = DEFAULT_SNAPSHOT
    if "--snapshot" in argv:
        i = argv.index("--snapshot")
        snap = Path(argv[i + 1]).resolve()
        args.remove(argv[i + 1])
    only = set(args)
    strict, force = "--strict" in flags, "--force" in flags

    catalog, failures = load_catalog()
    headers = {"User-Agent": "ai-datasets-sampler/0.1 (static educational site)"}
    with httpx.Client(timeout=180, headers=headers, follow_redirects=True) as client:
        if "--check" in flags:
            broken = check_sources(client, catalog)
            if broken:
                warn(f"broken sources: {', '.join(sorted(set(broken)))}")
            return 1 if broken or failures else 0

        manifest_path = snap / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        entries: dict = manifest.get("datasets", {})
        (snap / "generated").mkdir(parents=True, exist_ok=True)

        for ds_id, meta in catalog.items():
            if only and ds_id not in only:
                continue
            if not meta.get("source", {}).get("hf"):
                print(f"- {ds_id}: no source, skipped")
                continue
            fp = fingerprint(meta)
            json_path = snap / "generated" / f"{ds_id}.json"
            if not force and entries.get(ds_id, {}).get("fingerprint") == fp and json_path.exists():
                print(f"- {ds_id}: unchanged, reusing snapshot")
                continue
            print(f"- {ds_id}: fetching {meta['source']['hf']}", flush=True)
            # Fetch into a scratch dir and swap in only on success, so a failed
            # fetch leaves the previous (stale but complete) sample in place.
            tmp_media = snap / ".tmp" / ds_id
            shutil.rmtree(tmp_media, ignore_errors=True)
            try:
                data = sample_dataset(client, meta, tmp_media)
                data["revision"] = upstream_revision(client, meta["source"]["hf"])
            except Exception as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                failures.append(ds_id)
                continue
            media_dir = snap / "samples" / ds_id
            shutil.rmtree(media_dir, ignore_errors=True)
            if tmp_media.exists():
                media_dir.parent.mkdir(parents=True, exist_ok=True)
                tmp_media.rename(media_dir)
            # Stable, indented JSON so changes read well as diffs on the samples branch.
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            entries[ds_id] = {
                "fingerprint": fp,
                "hf": meta["source"]["hf"],
                "revision": data["revision"],
                "fetched_at": data["fetched_at"],
                "rows": len(data["rows"]),
            }
            print(f"    {len(data['rows'])} rows of {data['num_rows_total']:,}")
        shutil.rmtree(snap / ".tmp", ignore_errors=True)

        # Prune datasets that left the catalog (only on a full run).
        if not only:
            for ds_id in sorted(set(entries) - set(catalog)):
                print(f"- {ds_id}: removed from catalog, pruning")
                entries.pop(ds_id)
                (snap / "generated" / f"{ds_id}.json").unlink(missing_ok=True)
                shutil.rmtree(snap / "samples" / ds_id, ignore_errors=True)

        (snap / "README.md").write_text(SNAPSHOT_README)
        manifest = {"sampler_version": SAMPLER_VERSION, "datasets": dict(sorted(entries.items()))}
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    install(snap)
    if failures:
        warn(f"no fresh samples for: {', '.join(failures)} (previous snapshot kept if there was one)")
        return 1 if strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

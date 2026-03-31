#!/mnt/c/xampp/htdocs/website_rebuild/chatbot/bin/python3
"""
Two-stage cube separator report against the full service history Excel.

Stage 1 — Detection (cube_separator_detector.pt):
  "Is this photo showing a cube separator?"
  → NO  → NOISE

Stage 2 — Quality (cube_separator_quality.pt   ← V2  /  cube_separator_model.pkl  ← V1):
  "Is this good, bad, or cleaning?"
  → BAD      → written to CSV (top rows, with uar_link)
  → CLEANING → written to CSV (mid-service state, not a fault)
  → GOOD     → written to CSV

False-positive controls (applied after Stage 2):
  --min-confidence  If max(softmax) < threshold → UNCERTAIN  (default: 0.80)
  --tta             Test-time augmentation: average 5 crops for more stable predictions
  CONFLICT          Post-processing: units with BAD on one visit and GOOD on
                    another are flagged CONFLICT for manual review.

quality_margin is written to the CSV as an info column but does not affect results.

Every photo URL is written to the CSV — MISSING if the image couldn't be loaded.

Result sort order: BAD → UNCERTAIN → CONFLICT → GOOD → CLEANING → NOISE → MISSING

Reads:  hoz_cares_unit_visits.xlsx
        cube_separator_detector.pt     (Stage 1)
        cube_separator_quality.pt      (Stage 2 V2 — preferred)
        cube_separator_model.pkl       (Stage 2 V1 — fallback if .pt not found)

Outputs: cube_separator_report.csv

Usage:
    python run_cube_separator_report.py
    python run_cube_separator_report.py --output cube_separator_report_v3.csv
    python run_cube_separator_report.py --unit-type "Ice Machine"
    python run_cube_separator_report.py --detect-confidence 0.85
    python run_cube_separator_report.py --min-confidence 0.635
    python run_cube_separator_report.py --tta
"""

import argparse
import csv
import io
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image

DEFAULT_EXCEL    = Path(__file__).parent / "hoz_cares_unit_visits.xlsx"
DEFAULT_DETECTOR = Path(__file__).parent / "cube_separator_detector.pt"
DEFAULT_QUALITY  = Path(__file__).parent / "cube_separator_quality.pt"
DEFAULT_QUALITY_V1 = Path(__file__).parent / "cube_separator_model.pkl"
DEFAULT_REPORT   = Path(__file__).parent / "cube_separator_report.csv"

RESULT_SORT = {"BAD": 0, "UNCERTAIN": 1, "CONFLICT": 2, "GOOD": 3, "CLEANING": 4, "NOISE": 5, "MISSING": 6}

# Photos with these descriptions are skipped before hitting either model —
# confirmed non-cube-separator photo types based on field description analysis.
# These are pre-filtered to save download time and avoid false positives.
SKIP_DESCRIPTIONS = {
    # Labels / identification
    "qr code", "qr", "qr codes",
    "data plate", "data sticker", "nameplate",
    "label", "serial number", "model number",

    # Electrical / diagnostics
    "last error code log", "last error code log / no applicable",
    "error code", "error codes", "error log",
    "disconnect", "disconnect tag", "secured unit",

    # Probes / sensors / hoses
    "ice thickness probe position after adjustment",
    "ice thickness probe", "thickness probe",
    "water line", "ice machine hose", "hose",

    # Bin / drop zone (not the cube separator area)
    "before cleaning bin", "empty bin after cleaning",
    "bin control", "bin", "ice in bin", "ice level",
    "drop zone wiped down", "before cleaning drop zone",

    # Exterior / unit-level
    "unit on", "unit off", "unit exterior", "outside unit",
    "unit front", "no head clearance",
    "air filter rinsing", "air filter", "filter",

    # Miscellaneous non-CS
    "photo", "before cleaning (dirty photo)",
}

REPORT_HEADERS = [
    "result", "quality_confidence", "quality_margin", "detect_confidence",
    "region", "site_id", "site_num",
    "unit_id", "unit_type", "model", "serial", "qrcode",
    "unit_history_link",
    "first_visit_date", "first_visit_cube_separator_photo",
    "current_visit_date", "total_visits",
    "visit_num", "uar_link", "current_visit_cube_separator_photo", "photo_description",
]

UNIT_HISTORY_BASE = "https://filtrexone.filtrexservicegroup.com/unit/history/"

# V1 fallback only (DINOv2 + Isolation Forest uses fixed 224px)
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


# ── Model loading ──────────────────────────────────────────────────────────────

def _make_efficientnet(num_classes: int):
    net = efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    return net


def _build_transform(input_size: int):
    """Build inference transform for the given model input size."""
    return T.Compose([
        T.Resize(input_size + 32),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _build_tta_transforms(input_size: int):
    """Five-crop TTA transforms for the given model input size."""
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        T.Compose([T.Resize(input_size + 32), T.CenterCrop(input_size), T.ToTensor(), norm]),
        T.Compose([T.Resize(input_size + 32), T.RandomCrop(input_size), T.ToTensor(), norm]),
        T.Compose([T.Resize(input_size + 32), T.RandomCrop(input_size), T.ToTensor(), norm]),
        T.Compose([T.Resize(input_size + 32), T.RandomCrop(input_size), T.ToTensor(), norm]),
        T.Compose([T.Resize(input_size + 32), T.RandomCrop(input_size), T.ToTensor(), norm]),
    ]


def load_detector(path: Path):
    if not path.exists():
        print(f"Detector not found: {path} — falling back to single-stage mode.")
        return None, None, None, None, None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    net = _make_efficientnet(num_classes=2)
    net.load_state_dict(bundle["model_state_dict"])
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    cs_idx = bundle["cs_class_idx"]
    version = bundle.get("version", 1)
    input_size = bundle.get("input_size", 224)
    arch = bundle.get("architecture", "efficientnet_b0")
    print(f"Detector loaded  | v{version} | {arch} | input={input_size}px | val_acc={bundle.get('val_acc','?')} | cs_idx={cs_idx}")
    return net, cs_idx, device, _build_transform(input_size), _build_tta_transforms(input_size)


def load_quality_v2(path: Path):
    """V2/V3: EfficientNet 3-class (good/bad/cleaning)"""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    num_classes = bundle.get("num_classes", 3)
    net = _make_efficientnet(num_classes=num_classes)
    net.load_state_dict(bundle["model_state_dict"])
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    idx_to_class = bundle["idx_to_class"]
    input_size = bundle.get("input_size", 224)
    arch = bundle.get("architecture", "efficientnet_b0")
    version = bundle.get("version", 2)
    print(f"Quality model loaded | v{version} | {arch} | input={input_size}px | val_acc={bundle.get('val_acc','?')} | classes={bundle.get('classes')}")
    return net, idx_to_class, device, "v2", _build_transform(input_size), _build_tta_transforms(input_size)


def load_quality_v1(path: Path):
    """V1 fallback: DINOv2 + Isolation Forest"""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Quality model loaded  | v1 (Isolation Forest) | trained on {bundle['good_count']} good photos | threshold={bundle['threshold']:.4f}")
    return bundle["clf"], bundle["threshold"], bundle["dinov2_model"], "v1"


def load_dinov2(model_name: str):
    print(f"Loading {model_name} (first run ~330MB, cached after)...")
    dino = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    dino.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = dino.to(device)
    print(f"  Device: {device}")
    return dino, device


# ── Inference helpers ──────────────────────────────────────────────────────────

def fetch_image(url: str):
    clean_url = url.split("?")[0]
    try:
        resp = SESSION.get(clean_url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def is_cube_separator(detector, cs_idx, device, transform, img: Image.Image, min_confidence: float):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(detector(tensor), dim=1).squeeze()
    conf = probs[cs_idx].item()
    return conf >= min_confidence, round(conf, 4)


def quality_score_v2(net, idx_to_class, device, transform, tta_transforms, img: Image.Image, use_tta: bool = False):
    """V2/V3: EfficientNet 3-class — returns (result, confidence, margin).

    margin = top-1 probability minus top-2 probability.
    When use_tta=True, averages predictions over 5 crops for more stable results.
    """
    if use_tta:
        all_probs = []
        with torch.no_grad():
            for tf in tta_transforms:
                tensor = tf(img).unsqueeze(0).to(device)
                all_probs.append(torch.softmax(net(tensor), dim=1).squeeze())
        probs = torch.stack(all_probs).mean(0)
    else:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(net(tensor), dim=1).squeeze()

    sorted_probs, _ = torch.sort(probs, descending=True)
    class_idx  = probs.argmax().item()
    result     = idx_to_class[class_idx].upper()
    confidence = round(sorted_probs[0].item(), 4)
    margin     = round((sorted_probs[0] - sorted_probs[1]).item(), 4)
    return result, confidence, margin


def quality_score_v1(dino, dino_device, clf, threshold, img: Image.Image):
    """V1 fallback: DINOv2 + Isolation Forest — returns (BAD|GOOD, score, None)."""
    tensor = TRANSFORM(img).unsqueeze(0).to(dino_device)
    with torch.no_grad():
        emb = dino(tensor)
    score = clf.decision_function([emb.squeeze().cpu().numpy()])[0]
    result = "BAD" if score < threshold else "GOOD"
    return result, round(float(score), 6), None


# ── Per-unit conflict detection ────────────────────────────────────────────────

def flag_conflicts(rows: list[dict]) -> int:
    """Mark rows CONFLICT where a unit has both BAD and GOOD results across visits.

    Ignores UNCERTAIN, NOISE, MISSING, and CLEANING when deciding conflict —
    CLEANING is a mid-service state, not an opposing quality signal.
    Returns the number of rows changed.
    """
    # Collect decisive results per unit (qrcode)
    unit_results: dict[str, set] = defaultdict(set)
    for row in rows:
        if row["result"] in ("BAD", "GOOD"):
            unit_results[row["qrcode"]].add(row["result"])

    # A unit conflicts when it has scored both BAD and GOOD across visits
    conflict_units = {qr for qr, results in unit_results.items()
                      if "BAD" in results and "GOOD" in results}

    changed = 0
    for row in rows:
        if row["qrcode"] in conflict_units and row["result"] == "GOOD":
            row["result"] = "CONFLICT"
            changed += 1

    return changed


# ── Excel parsing ──────────────────────────────────────────────────────────────

def parse_photos(json_str, source_label: str):
    if not json_str or not isinstance(json_str, str):
        return []
    try:
        photos = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []
    return [
        {"photo_url": (p.get("photo") or "").strip(),
         "photo_description": p.get("description", ""),
         "photo_source": source_label}
        for p in photos if (p.get("photo") or "").strip()
    ]


def load_visits(excel_path: Path, unit_type_filter: str | None):
    import pandas as pd
    df = pd.read_excel(excel_path)

    # Compute equipment history per unit across ALL visits (before filtering)
    df["completed time"] = pd.to_datetime(df["completed time"], errors="coerce")

    # Build first-visit cube-separator photo URL per unit.
    # Only returns a URL if the description explicitly names the cube separator
    # or its components. Returns "" rather than falling back to a bin/evap photo.
    CS_DESCRIPTION_KEYWORDS = (
        "cube separator",           # explicit
        "ice machine components",   # CS components being cleaned/reassembled
        "components in cleaning",   # CS components in cleaning solution
    )

    def best_cs_photo_from_row(row):
        """Return the first cube-separator photo URL from a single visit row,
        matched by description. Returns '' if none found."""
        for field in ("photo json", "problem photos json"):
            raw = row.get(field)
            if not raw or not isinstance(raw, str):
                continue
            try:
                photos = json.loads(raw)
            except Exception:
                continue
            for p in photos:
                desc = (p.get("description") or "").strip().lower()
                url  = (p.get("photo") or "").strip().split("?")[0]
                if not url:
                    continue
                if any(kw in desc for kw in CS_DESCRIPTION_KEYWORDS):
                    return url
        return ""

    df_sorted = df.sort_values("completed time")
    first_visit_rows = df_sorted.groupby("qrcode", sort=False).first().reset_index()
    first_photo_map = {
        str(row["qrcode"]): best_cs_photo_from_row(row)
        for _, row in first_visit_rows.iterrows()
    }

    unit_history = (
        df.groupby("qrcode")["completed time"]
        .agg(first_visit_date="min", total_visits="count")
        .reset_index()
    )
    unit_history["first_visit_date"] = unit_history["first_visit_date"].dt.strftime("%Y-%m-%d")
    history_map = unit_history.set_index("qrcode").to_dict("index")

    if unit_type_filter:
        df = df[df["unit_type"].str.contains(unit_type_filter, case=False, na=False)]
        print(f"Filtered to '{unit_type_filter}': {len(df)} visits")
    else:
        print(f"Loaded {len(df)} visits (all unit types)")

    all_photos = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        photos = (
            parse_photos(row.get("photo json"),          "visit_photo") +
            parse_photos(row.get("problem photos json"), "problem_photo")
        )
        qrcode   = str(row.get("qrcode", ""))
        hist     = history_map.get(qrcode, {})
        raw_date = row.get("completed time")
        current_date = raw_date.strftime("%Y-%m-%d") if pd.notna(raw_date) else ""

        meta = {
            "unit_id":           str(row.get("Unit ID",       "")),
            "unit_type":         str(row.get("unit_type",     "")),
            "serial":            str(row.get("serial",        "")),
            "model":             str(row.get("model",         "")),
            "qrcode":            qrcode,
            "region":            str(row.get("region",        "")),
            "site_id":           str(row.get("site ID",       "")),
            "site_num":          str(row.get("site #",        "")),
            "visit_num":         str(row.get("visit #",       "")),
            "uar_link":          str(row.get("uar Link",      "")),
            "first_visit_date":      hist.get("first_visit_date", ""),
            "first_visit_photo_url": first_photo_map.get(qrcode, ""),
            "current_visit_date":    current_date,
            "total_visits":      str(hist.get("total_visits", "")),
        }
        for photo in photos:
            url = photo.get("photo_url", "")
            if not url:
                continue
            # Deduplicate on qrcode + url — catches same photo across
            # duplicate visit rows or repeated in both photo/problem_photo fields
            dedup_key = f"{qrcode}|{url}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            all_photos.append({**meta, **photo})

    return all_photos


def make_row(p, result, quality_confidence="", quality_margin="", detect_confidence=""):
    qrcode = p["qrcode"]
    return {
        "result":              result,
        "quality_confidence":  quality_confidence,
        "quality_margin":      quality_margin,
        "detect_confidence":   detect_confidence,
        "region":              p["region"],
        "site_id":             p["site_id"],
        "site_num":            p["site_num"],
        "unit_id":             p["unit_id"],
        "unit_type":           p["unit_type"],
        "model":               p["model"],
        "serial":              p["serial"],
        "qrcode":              qrcode,
        "unit_history_link":                f"{UNIT_HISTORY_BASE}{qrcode}" if qrcode else "",
        "first_visit_date":                 p["first_visit_date"],
        "first_visit_cube_separator_photo": p["first_visit_photo_url"],
        "current_visit_date":               p["current_visit_date"],
        "total_visits":                     p["total_visits"],
        "visit_num":                        p["visit_num"],
        "uar_link":                         p["uar_link"],
        "current_visit_cube_separator_photo": p["photo_url"],
        "photo_description":                p["photo_description"],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel",             default=str(DEFAULT_EXCEL))
    parser.add_argument("--detector",          default=str(DEFAULT_DETECTOR))
    parser.add_argument("--quality-model",     default=str(DEFAULT_QUALITY),
                        help="V2 quality model .pt (default). Falls back to .pkl if not found.")
    parser.add_argument("--output",            default=str(DEFAULT_REPORT))
    parser.add_argument("--unit-type",         default="Ice Machine")
    parser.add_argument("--detect-confidence", type=float, default=0.85,
                        help="Min confidence to pass Stage 1 (default: 0.85)")
    parser.add_argument("--min-confidence",    type=float, default=0.635,
                        help="Min quality confidence to trust result — below this becomes UNCERTAIN (default: 0.635)")
    parser.add_argument("--tta",               action="store_true",
                        help="Test-time augmentation: average 5 crops per image (~5x slower but more accurate)")
    args = parser.parse_args()

    if not Path(args.excel).exists():
        print(f"Excel not found: {args.excel}")
        raise SystemExit(1)

    # Load Stage 1 detector
    detector, cs_idx, det_device, det_transform, det_tta = load_detector(Path(args.detector))
    single_stage = detector is None

    # Load Stage 2 quality model (V2/V3 preferred, V1 fallback)
    quality_path = Path(args.quality_model)
    if quality_path.exists():
        quality_net, idx_to_class, qual_device, qual_version, qual_transform, qual_tta = load_quality_v2(quality_path)
        dino = None
    elif DEFAULT_QUALITY_V1.exists():
        print(f"V2 quality model not found at {quality_path} — falling back to V1 pkl")
        clf, threshold, dinov2_name = load_quality_v1(DEFAULT_QUALITY_V1)[:3]
        qual_version = "v1"
        quality_net = idx_to_class = qual_device = qual_transform = qual_tta = None
        dino, dino_device = load_dinov2(dinov2_name)
    else:
        print(f"No quality model found. Checked:")
        print(f"  V2: {quality_path}")
        print(f"  V1: {DEFAULT_QUALITY_V1}")
        raise SystemExit(1)

    print(f"\nQuality model version : {qual_version}")
    if single_stage:
        print("Stage 1              : DISABLED (no detector found)")
    else:
        print(f"Stage 1 confidence   : >= {args.detect_confidence}")
    print(f"Min confidence       : {args.min_confidence}")
    print(f"TTA                  : {'ON (5 crops averaged)' if args.tta else 'OFF'}")

    unit_filter = args.unit_type.strip() or None
    photos = load_visits(Path(args.excel), unit_filter)
    print(f"Total photo URLs     : {len(photos)}\n")

    rows = []
    t0 = time.time()

    # Tracks the earliest confirmed cube separator photo per unit.
    # Built during the run using Stage 1 — 100% accurate, no description guessing.
    # key: qrcode  value: (visit_date, photo_url)
    first_cs_photo: dict[str, tuple[str, str]] = {}

    for i, p in enumerate(photos, 1):
        # Skip known non-cube-separator photo types before fetching
        if (p.get("photo_description") or "").strip().lower() in SKIP_DESCRIPTIONS:
            rows.append(make_row(p, "NOISE", detect_confidence="skipped"))
            continue

        img = fetch_image(p["photo_url"])

        if img is None:
            rows.append(make_row(p, "MISSING"))

        else:
            detect_conf = None

            # Stage 1
            if not single_stage:
                is_cs, detect_conf = is_cube_separator(
                    detector, cs_idx, det_device, det_transform, img, args.detect_confidence
                )
                if not is_cs:
                    rows.append(make_row(p, "NOISE", detect_confidence=round(detect_conf, 4)))
                    if i % 50 == 0 or i == len(photos):
                        _print_progress(i, len(photos), rows, t0)
                    continue

            # Photo confirmed as cube separator by Stage 1 — track earliest per unit
            qr   = p["qrcode"]
            date = p["current_visit_date"]
            url  = p["photo_url"]
            if qr and (qr not in first_cs_photo or date < first_cs_photo[qr][0]):
                first_cs_photo[qr] = (date, url)

            # Stage 2
            if qual_version == "v2":
                result, qual_conf, qual_margin = quality_score_v2(
                    quality_net, idx_to_class, qual_device,
                    qual_transform, qual_tta, img, use_tta=args.tta
                )
            else:
                result, qual_conf, qual_margin = quality_score_v1(dino, dino_device, clf, threshold, img)

            # Flag low-confidence predictions for manual review
            if args.min_confidence and qual_conf < args.min_confidence:
                result = "UNCERTAIN"

            rows.append(make_row(p, result,
                                 quality_confidence=qual_conf,
                                 quality_margin=qual_margin if qual_margin is not None else "n/a",
                                 detect_confidence=round(detect_conf, 4) if detect_conf is not None else "n/a"))

        if i % 50 == 0 or i == len(photos):
            _print_progress(i, len(photos), rows, t0)

    # Backfill first_visit_cube_separator_photo using Stage 1-confirmed URLs
    for row in rows:
        qr = row["qrcode"]
        if qr in first_cs_photo:
            row["first_visit_cube_separator_photo"] = first_cs_photo[qr][1]

    # Per-unit conflict detection — units with BAD on one visit and GOOD on another
    n_conflicts = flag_conflicts(rows)

    # Sort: BAD → UNCERTAIN → CONFLICT → GOOD → CLEANING → NOISE → MISSING
    # Within each result group: region → site → unit → visit date
    rows.sort(key=lambda r: (
        RESULT_SORT.get(r["result"], 9),
        r["region"],
        r["site_num"].zfill(10),
        r["qrcode"],
        r["current_visit_date"],
    ))

    with open(Path(args.output), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {r: sum(1 for x in rows if x["result"] == r) for r in RESULT_SORT}
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Total URLs   : {len(photos)}")
    print(f"  BAD          : {counts['BAD']}")
    print(f"  UNCERTAIN    : {counts['UNCERTAIN']}  (low confidence or low margin — review manually)")
    print(f"  CONFLICT     : {counts['CONFLICT']}  (unit scored BAD on one visit, GOOD on another)")
    print(f"  GOOD         : {counts['GOOD']}")
    print(f"  CLEANING     : {counts['CLEANING']}  (disassembled for service — not a fault)")
    print(f"  NOISE        : {counts['NOISE']}  (not a cube separator)")
    print(f"  MISSING      : {counts['MISSING']}  (image could not be loaded)")
    print(f"  Time         : {elapsed:.0f}s")
    print(f"\n  Report saved : {Path(args.output).resolve()}")

    if counts["BAD"] > 0:
        print(f"\n  Flagged installs (BAD):")
        for r in rows:
            if r["result"] == "BAD":
                print(f"    conf={r['quality_confidence']}  margin={r['quality_margin']}  det={r['detect_confidence']}  {r['qrcode']}  {r['region']}")
                print(f"           {r['uar_link']}")

    if counts["CONFLICT"] > 0:
        print(f"\n  Conflicting units (BAD on one visit, GOOD on another):")
        seen = set()
        for r in rows:
            if r["result"] in ("BAD", "CONFLICT") and r["qrcode"] not in seen:
                seen.add(r["qrcode"])
                print(f"    {r['qrcode']}  {r['region']}  {r['uar_link']}")


def _print_progress(i, total, rows, t0):
    counts = {r: sum(1 for x in rows if x["result"] == r) for r in RESULT_SORT}
    print(f"  {i}/{total}  bad={counts['BAD']}  uncertain={counts['UNCERTAIN']}  "
          f"conflict={counts['CONFLICT']}  good={counts['GOOD']}  cleaning={counts['CLEANING']}  "
          f"noise={counts['NOISE']}  missing={counts['MISSING']}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

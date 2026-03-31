#!/mnt/c/xampp/htdocs/website_rebuild/chatbot/bin/python3
"""
Drop Zone Defect Report — 2-Stage Pipeline

Stage 1 — Detection (drop_zone_detector.pt, EfficientNet B0):
  "Is this photo the inside of an ice bin / drop zone?"
  → NO  → NOISE (skipped)
  → YES → Stage 2

Stage 2 — Quality (drop_zone_model.pkl, DINOv2 + Isolation Forest):
  "Is this drop zone GOOD or a manufacturing DEFECT?"
  → DEFECT / UNCERTAIN / GOOD

Scores the FIRST VISIT drop zone photo for each Hoshizaki ice machine unit.
Goal: identify units where the drop zone was already defective on arrival
(manufacturing defect or shipping damage).

Only the earliest visit photo per unit is scored — if the defect shows up
on visit 1 it points to manufacturer/shipping, not wear or tech error.

Models:  drop_zone-model/drop_zone_detector.pt   (Stage 1 — EfficientNet B0)
         drop_zone-model/drop_zone_model.pkl      (Stage 2 — DINOv2 + Isolation Forest)
Data:    queries units_visits DB directly

Result sort order: DEFECT → UNCERTAIN → GOOD → NOISE → MISSING

Outputs: drop_zone_report.csv

Usage:
    python run_drop_zone_report.py
    python run_drop_zone_report.py --output drop_zone_report_v2.csv
    python run_drop_zone_report.py --min-confidence 0.50
    python run_drop_zone_report.py --detect-confidence 0.85
    python run_drop_zone_report.py --all-visits   (score every visit, not just first)
"""

import argparse
import csv
import io
import json
import pickle
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

import requests
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image

CHATBOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHATBOT_DIR))
from database import get_connection  # noqa: E402

DEFAULT_DETECTOR = Path(__file__).parent / "drop_zone-model" / "drop_zone_detector.pt"
DEFAULT_MODEL    = Path(__file__).parent / "drop_zone-model" / "drop_zone_model.pkl"
DEFAULT_REPORT   = Path(__file__).parent / "drop_zone_report.csv"

UNIT_HISTORY_BASE = "https://filtrexone.filtrexservicegroup.com/unit/history/"

DROP_ZONE_DESCRIPTIONS = {"drop zone wiped down", "before cleaning drop zone"}

RESULT_SORT = {"DEFECT": 0, "UNCERTAIN": 1, "GOOD": 2, "NOISE": 3, "MISSING": 4}

REPORT_HEADERS = [
    "result", "anomaly_score", "threshold", "detect_confidence",
    "qr_code", "unit_history_link",
    "brand", "model", "serial", "unit_type",
    "region", "site_name", "site_number",
    "first_visit_date", "first_visit_photo_url",
    "photo_description",
]

# Standard ImageNet transform — used for both Stage 1 and DINOv2
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


# ── Model loading ───────────────────────────────────────────────────────────────

def _make_efficientnet(num_classes: int):
    net = efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    return net


def load_detector(path: Path):
    """Load Stage 1 EfficientNet detector. Returns None if not found (single-stage fallback)."""
    if not path.exists():
        print(f"Stage 1 detector not found: {path}")
        print("  Running in single-stage mode — all drop zone photos will be scored.")
        return None, None, None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    net = _make_efficientnet(num_classes=2)
    net.load_state_dict(bundle["model_state_dict"])
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    dz_idx = bundle["dz_class_idx"]
    print(f"Stage 1 loaded | v{bundle.get('version',1)} | val_acc={bundle.get('val_acc','?')} | dz_idx={dz_idx}")
    return net, dz_idx, device


def load_quality_model(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Stage 2 loaded | good={bundle['good_count']}  bad={bundle['bad_count']}"
          f"  threshold={bundle['threshold']:.4f}  dinov2={bundle['dinov2_model']}")
    return bundle["clf"], bundle["threshold"], bundle["dinov2_model"]


def load_dinov2(model_name: str):
    print(f"Loading {model_name} (first run ~330MB, cached after)...")
    dino = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    dino.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    return dino.to(device), device


# ── Inference ───────────────────────────────────────────────────────────────────

def is_drop_zone(detector, dz_idx, device, img: Image.Image, min_conf: float):
    """Stage 1 — returns (is_drop_zone, confidence)."""
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(detector(tensor), dim=1).squeeze()
    conf = probs[dz_idx].item()
    return conf >= min_conf, round(conf, 4)


def score_image(dino, device, clf, threshold, img: Image.Image, min_confidence: float):
    """Stage 2 — returns (result, anomaly_score)."""
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = dino(tensor)
    score = clf.decision_function([emb.squeeze().cpu().numpy()])[0]
    score = round(float(score), 6)

    if score < threshold:
        result = "DEFECT"
    elif score < threshold + abs(threshold) * min_confidence:
        result = "UNCERTAIN"
    else:
        result = "GOOD"

    return result, score


# ── Database ────────────────────────────────────────────────────────────────────

def load_first_visit_photos(all_visits: bool) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT uv.id as visit_id, uv.unit_id, uv.qr_code, "
            "       uv.photos, uv.problem_photos, uv.completion_time, "
            "       u.serial, u.model, "
            "       ut.name as unit_type, "
            "       b.name as brand, "
            "       s.site_name, s.site_number, "
            "       r.name as region "
            "FROM units_visits uv "
            "JOIN units u        ON uv.unit_id       = u.id "
            "JOIN unit_types ut  ON uv.unit_type_id   = ut.id "
            "JOIN unit_brands b  ON u.brand_id        = b.id "
            "LEFT JOIN sites s   ON uv.site_id        = s.id "
            "LEFT JOIN regions r ON uv.region_id      = r.id "
            "WHERE ut.name LIKE %s "
            "AND b.name LIKE %s "
            "AND (uv.photos LIKE %s OR uv.problem_photos LIKE %s) "
            "ORDER BY uv.qr_code, uv.completion_time ASC",
            ('%Ice%', '%Hoshizaki%', '%drop zone%', '%drop zone%')
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"DB returned {len(rows)} Hoshizaki drop zone visits")

    seen_units: set[str] = set()
    records = []

    for row in rows:
        qr = str(row["qr_code"] or "")
        if not all_visits:
            if qr in seen_units:
                continue
            seen_units.add(qr)

        completed = (
            row["completion_time"].strftime("%Y-%m-%d")
            if row["completion_time"] else ""
        )

        photo_url = ""
        photo_desc = ""
        for field in ("photos", "problem_photos"):
            raw = row.get(field)
            if not raw:
                continue
            try:
                photos = json.loads(raw)
            except Exception:
                continue
            for p in photos:
                desc = (p.get("description") or "").strip().lower()
                url  = (p.get("photo")       or "").strip().split("?")[0]
                if url and desc in DROP_ZONE_DESCRIPTIONS:
                    photo_url  = url
                    photo_desc = desc
                    break
            if photo_url:
                break

        if not photo_url:
            continue

        records.append({
            "qr_code":      qr,
            "unit_id":      str(row["unit_id"]),
            "unit_type":    str(row["unit_type"]   or ""),
            "brand":        str(row["brand"]        or ""),
            "model":        str(row["model"]        or ""),
            "serial":       str(row["serial"]       or ""),
            "region":       str(row["region"]       or ""),
            "site_name":    str(row["site_name"]    or ""),
            "site_number":  str(row["site_number"]  or ""),
            "first_visit_date":      completed,
            "first_visit_photo_url": photo_url,
            "photo_description":     photo_desc,
        })

    label = "visits" if all_visits else "units (first visit only)"
    print(f"Drop zone photos to score: {len(records)} {label}")
    return records


# ── Image fetch ─────────────────────────────────────────────────────────────────

def fetch_image(url: str):
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector",        default=str(DEFAULT_DETECTOR),
                        help="Stage 1 EfficientNet detector .pt")
    parser.add_argument("--model",           default=str(DEFAULT_MODEL),
                        help="Stage 2 DINOv2 + Isolation Forest .pkl")
    parser.add_argument("--output",          default=str(DEFAULT_REPORT))
    parser.add_argument("--detect-confidence", type=float, default=0.85,
                        help="Min Stage 1 confidence to pass as drop zone (default: 0.85)")
    parser.add_argument("--min-confidence",  type=float, default=0.50,
                        help="UNCERTAIN buffer above threshold (default: 0.50)")
    parser.add_argument("--all-visits",      action="store_true",
                        help="Score every visit instead of first visit only")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Stage 2 model not found: {model_path}")
        raise SystemExit(1)

    # Stage 1
    detector, dz_idx, det_device = load_detector(Path(args.detector))
    single_stage = detector is None

    # Stage 2
    clf, threshold, dinov2_name = load_quality_model(model_path)
    dino, dino_device = load_dinov2(dinov2_name)

    print(f"\nMode             : {'all visits' if args.all_visits else 'first visit per unit'}")
    if not single_stage:
        print(f"Stage 1 threshold: >= {args.detect_confidence}")
    print(f"Stage 2 threshold: {threshold:.4f}")
    print(f"Uncertain buffer : {args.min_confidence}")

    records = load_first_visit_photos(args.all_visits)
    print(f"Total to score   : {len(records)}\n")

    rows = []
    t0 = time.time()
    counts = {k: 0 for k in RESULT_SORT}

    for i, rec in enumerate(records, 1):
        img = fetch_image(rec["first_visit_photo_url"])

        if img is None:
            result       = "MISSING"
            score        = ""
            detect_conf  = ""
        else:
            detect_conf = ""

            # Stage 1 — is this actually a drop zone photo?
            if not single_stage:
                passed, detect_conf = is_drop_zone(detector, dz_idx, det_device, img, args.detect_confidence)
                if not passed:
                    result = "NOISE"
                    score  = ""
                    counts["NOISE"] = counts.get("NOISE", 0) + 1
                    rows.append({
                        "result":               result,
                        "anomaly_score":        score,
                        "threshold":            round(threshold, 6),
                        "detect_confidence":    detect_conf,
                        "qr_code":              rec["qr_code"],
                        "unit_history_link":    f"{UNIT_HISTORY_BASE}{rec['qr_code']}" if rec["qr_code"] else "",
                        "brand":                rec["brand"],
                        "model":                rec["model"],
                        "serial":               rec["serial"],
                        "unit_type":            rec["unit_type"],
                        "region":               rec["region"],
                        "site_name":            rec["site_name"],
                        "site_number":          rec["site_number"],
                        "first_visit_date":     rec["first_visit_date"],
                        "first_visit_photo_url": rec["first_visit_photo_url"],
                        "photo_description":    rec["photo_description"],
                    })
                    if i % 50 == 0 or i == len(records):
                        _print_progress(i, len(records), counts, t0)
                    continue

            # Stage 2 — quality score
            result, score = score_image(dino, dino_device, clf, threshold, img, args.min_confidence)

        counts[result] = counts.get(result, 0) + 1

        rows.append({
            "result":               result,
            "anomaly_score":        score,
            "threshold":            round(threshold, 6),
            "detect_confidence":    detect_conf,
            "qr_code":              rec["qr_code"],
            "unit_history_link":    f"{UNIT_HISTORY_BASE}{rec['qr_code']}" if rec["qr_code"] else "",
            "brand":                rec["brand"],
            "model":                rec["model"],
            "serial":               rec["serial"],
            "unit_type":            rec["unit_type"],
            "region":               rec["region"],
            "site_name":            rec["site_name"],
            "site_number":          rec["site_number"],
            "first_visit_date":     rec["first_visit_date"],
            "first_visit_photo_url": rec["first_visit_photo_url"],
            "photo_description":    rec["photo_description"],
        })

        if i % 50 == 0 or i == len(records):
            _print_progress(i, len(records), counts, t0)

    # Sort: DEFECT → UNCERTAIN → GOOD → NOISE → MISSING
    rows.sort(key=lambda r: (
        RESULT_SORT.get(r["result"], 9),
        r["region"],
        r["site_name"],
        r["first_visit_date"],
    ))

    with open(Path(args.output), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Units scored : {len(records)}")
    print(f"  DEFECT       : {counts.get('DEFECT', 0)}  (anomalous on first visit)")
    print(f"  UNCERTAIN    : {counts.get('UNCERTAIN', 0)}  (borderline — review manually)")
    print(f"  GOOD         : {counts.get('GOOD', 0)}")
    print(f"  NOISE        : {counts.get('NOISE', 0)}  (Stage 1 rejected — not a drop zone photo)")
    print(f"  MISSING      : {counts.get('MISSING', 0)}  (photo could not be loaded)")
    print(f"  Time         : {elapsed:.0f}s")
    print(f"\n  Report saved : {Path(args.output).resolve()}")

    if counts.get("DEFECT", 0) > 0:
        print(f"\n  Flagged units (DEFECT):")
        for r in rows:
            if r["result"] == "DEFECT":
                print(f"    score={r['anomaly_score']}  det={r['detect_confidence']}  {r['qr_code']}  {r['region']}  {r['site_name']}")
                print(f"           {r['unit_history_link']}")


def _print_progress(i, total, counts, t0):
    print(f"  {i}/{total}  defect={counts.get('DEFECT',0)}  "
          f"uncertain={counts.get('UNCERTAIN',0)}  good={counts.get('GOOD',0)}  "
          f"noise={counts.get('NOISE',0)}  missing={counts.get('MISSING',0)}  "
          f"elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

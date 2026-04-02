#!/usr/bin/env python3
"""
Model Version Comparison — CSV → Normalized 4-Table Schema

Reads 1–4 cube separator report CSVs (each from a different model version)
and outputs a normalized dataset for side-by-side model comparison.

Output tables:
    units.csv   — one row per physical machine (deduped across all input files)
    visits.csv  — one row per service visit
    photos.csv  — one row per scored photo
    scores.csv  — one row per (photo × model version) — this is where versions differ

Usage:
    python build_model_comparison.py report_v1.csv report_v2.csv
    python build_model_comparison.py report_v1.csv report_v2.csv report_v3.csv report_v4.csv
    python build_model_comparison.py report_v1.csv report_v2.csv --model-versions B0-v1 B0-v2
    python build_model_comparison.py report_v1.csv --output-dir comparison/
    python build_model_comparison.py report_v1.csv report_v2.csv --photo-type drop_zone

Model version is inferred from the filename if --model-versions is not supplied.
  e.g.  cube_separator_report_v2.csv  →  "v2"
        cube_separator_report_final.csv  →  "final"
        report_B2_20250401.csv  →  "B2_20250401"
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Column maps from report CSV → schema ──────────────────────────────────────

# result → pipeline_status
PIPELINE_STATUS_MAP = {
    "bad":       "scored",
    "good":      "scored",
    "cleaning":  "scored",
    "uncertain": "uncertain",
    "conflict":  "conflict",
    "noise":     "noise",
    "missing":   "missing",
}

# result → verdict (NULL if not a decisive quality result)
VERDICT_MAP = {
    "bad":      "bad",
    "good":     "good",
    "cleaning": "cleaning",
}

# ── Output headers ─────────────────────────────────────────────────────────────

UNITS_HEADERS = [
    "qrcode", "serial", "model", "unit_type",
    "region", "site_num", "site_id",
    "unit_history_link", "first_visit_date", "first_visit_photo_url",
]

VISITS_HEADERS = [
    "visit_id", "qrcode", "visit_num", "visit_date",
    "uar_link", "total_visits_at_time",
]

PHOTOS_HEADERS = [
    "photo_id", "visit_id", "photo_url", "photo_type",
    "tech_label", "detect_confidence", "scored_at", "model_version",
]

SCORES_HEADERS = [
    "score_id", "photo_id",
    "pipeline_status", "verdict", "defect_type",
    "quality_confidence", "quality_margin",
    "model_version", "scored_at",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_model_version(filepath: str) -> str:
    """Extract a version string from the CSV filename.

    cube_separator_report_v2.csv      → v2
    cube_separator_report_final.csv   → final
    report_B2_20250401.csv            → B2_20250401
    cube_separator_report.csv         → v1  (bare name = assume first version)
    """
    stem = Path(filepath).stem  # filename without extension
    # Strip common prefixes
    stem = re.sub(r'^cube_separator_report_?', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'^report_?',               '', stem, flags=re.IGNORECASE)
    version = stem.strip("_- ") or "v1"
    return version


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, headers: list[str], rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>6,} rows  →  {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normalize cube separator report CSVs into a 4-table comparison schema."
    )
    parser.add_argument("csvfiles", nargs="+", metavar="CSV",
                        help="1–4 report CSV files, one per model version")
    parser.add_argument("--model-versions", nargs="+", metavar="VER",
                        help="Model version labels matching order of CSV files "
                             "(default: inferred from filenames)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write output CSVs (default: current dir)")
    parser.add_argument("--photo-type", default="cube_separator",
                        choices=["cube_separator", "drop_zone", "ice_chute", "other"],
                        help="photo_type value written to photos table (default: cube_separator)")
    args = parser.parse_args()

    if len(args.csvfiles) > 4:
        print("Error: maximum 4 CSV files at once.")
        raise SystemExit(1)

    if args.model_versions and len(args.model_versions) != len(args.csvfiles):
        print("Error: --model-versions count must match number of CSV files.")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assign version labels
    versions = args.model_versions or [infer_model_version(f) for f in args.csvfiles]
    scored_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nInput files:")
    for csv_path, ver in zip(args.csvfiles, versions):
        rows = read_csv(csv_path)
        print(f"  [{ver}]  {csv_path}  ({len(rows):,} rows)")

    # ── Build tables ────────────────────────────────────────────────────────────

    units_map:  dict[str, dict] = {}          # qrcode → unit row
    visits_map: dict[tuple, dict] = {}        # (qrcode, visit_num) → visit row
    photos_map: dict[tuple, dict] = {}        # (visit_id, photo_url) → photo row
    scores_rows: list[dict] = []

    visit_id_seq  = 1
    photo_id_seq  = 1
    score_id_seq  = 1

    for csv_path, model_version in zip(args.csvfiles, versions):
        rows = read_csv(csv_path)

        for row in rows:
            qrcode   = (row.get("qrcode") or "").strip()
            visit_num = (row.get("visit_num") or "").strip().rstrip(".0")
            result    = (row.get("result") or "").strip().lower()

            if not qrcode:
                continue

            # ── units ──────────────────────────────────────────────────────────
            if qrcode not in units_map:
                units_map[qrcode] = {
                    "qrcode":              qrcode,
                    "serial":              (row.get("serial")           or "").strip(),
                    "model":               (row.get("model")            or "").strip(),
                    "unit_type":           (row.get("unit_type")        or "").strip(),
                    "region":              (row.get("region")           or "").strip(),
                    "site_num":            (row.get("site_num")         or "").strip(),
                    "site_id":             (row.get("site_id")          or "").strip(),
                    "unit_history_link":   (row.get("unit_history_link") or "").strip(),
                    "first_visit_date":    (row.get("first_visit_date") or "").strip(),
                    "first_visit_photo_url": (
                        row.get("first_visit_cube_separator_photo") or
                        row.get("first_visit_photo_url") or ""
                    ).strip(),
                }

            # ── visits ─────────────────────────────────────────────────────────
            visit_key = (qrcode, visit_num)
            if visit_key not in visits_map:
                visits_map[visit_key] = {
                    "visit_id":            visit_id_seq,
                    "qrcode":              qrcode,
                    "visit_num":           visit_num,
                    "visit_date":          (row.get("current_visit_date") or "").strip(),
                    "uar_link":            (row.get("uar_link")           or "").strip(),
                    "total_visits_at_time": (row.get("total_visits")      or "").strip(),
                }
                visit_id_seq += 1

            visit_id = visits_map[visit_key]["visit_id"]

            # ── photos ─────────────────────────────────────────────────────────
            photo_url = (
                row.get("current_visit_cube_separator_photo") or
                row.get("photo_url") or ""
            ).strip()

            if not photo_url:
                continue

            photo_key = (visit_id, photo_url)
            if photo_key not in photos_map:
                photos_map[photo_key] = {
                    "photo_id":         photo_id_seq,
                    "visit_id":         visit_id,
                    "photo_url":        photo_url,
                    "photo_type":       args.photo_type,
                    "tech_label":       (row.get("photo_description") or "").strip(),
                    "detect_confidence": (row.get("detect_confidence") or "").strip(),
                    "scored_at":        scored_at,
                    "model_version":    model_version,
                }
                photo_id_seq += 1

            photo_id = photos_map[photo_key]["photo_id"]

            # ── scores ─────────────────────────────────────────────────────────
            pipeline_status = PIPELINE_STATUS_MAP.get(result, "missing")
            verdict         = VERDICT_MAP.get(result, "")

            scores_rows.append({
                "score_id":           score_id_seq,
                "photo_id":           photo_id,
                "pipeline_status":    pipeline_status,
                "verdict":            verdict,
                "defect_type":        "",   # requires manual labeling or future model output
                "quality_confidence": (row.get("quality_confidence") or "").strip(),
                "quality_margin":     (row.get("quality_margin")     or "").strip(),
                "model_version":      model_version,
                "scored_at":          scored_at,
            })
            score_id_seq += 1

    # ── Write output ────────────────────────────────────────────────────────────

    print(f"\nBuilding output tables...")

    units_rows  = list(units_map.values())
    visits_rows = list(visits_map.values())
    photos_rows = list(photos_map.values())

    write_csv(output_dir / "units.csv",  UNITS_HEADERS,  units_rows)
    write_csv(output_dir / "visits.csv", VISITS_HEADERS, visits_rows)
    write_csv(output_dir / "photos.csv", PHOTOS_HEADERS, photos_rows)
    write_csv(output_dir / "scores.csv", SCORES_HEADERS, scores_rows)

    # ── Summary ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Units          : {len(units_rows):,}  (unique machines)")
    print(f"  Visits         : {len(visits_rows):,}  (unique service visits)")
    print(f"  Photos         : {len(photos_rows):,}  (unique photo URLs)")
    print(f"  Scores         : {len(scores_rows):,}  (verdicts across all model versions)")
    print(f"\n  Model versions : {', '.join(versions)}")

    # Per-version verdict breakdown
    print(f"\n  Verdict breakdown by model version:")
    for ver in versions:
        ver_scores = [s for s in scores_rows if s["model_version"] == ver]
        counts: dict[str, int] = {}
        for s in ver_scores:
            key = s["verdict"] or s["pipeline_status"]
            counts[key] = counts.get(key, 0) + 1
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"    [{ver}]  {breakdown}")

    # Verdict drift — photos where different versions disagree
    if len(versions) > 1:
        from collections import defaultdict
        photo_verdicts: dict[int, dict[str, str]] = defaultdict(dict)
        for s in scores_rows:
            photo_verdicts[s["photo_id"]][s["model_version"]] = s["verdict"] or s["pipeline_status"]

        drift = sum(
            1 for verdicts in photo_verdicts.values()
            if len(set(verdicts.values())) > 1
        )
        print(f"\n  Verdict drift  : {drift:,} photos scored differently across versions")
        print(f"  (Filter scores.csv by photo_id where verdict changes to review drift)")

    print(f"\n  Output dir     : {output_dir.resolve()}")


if __name__ == "__main__":
    main()

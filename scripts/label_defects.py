"""
Drop Zone Defect Sub-Labeler
────────────────────────────
Reviews every DEFECT and UNCERTAIN photo from the drop zone report
and lets you tag each one with a defect type.

Labels:
  b → bacterial     algae/film/discoloration, maintenance issue
  c → caulking      visible caulk bead, sealant blob, adhesive residue
  s → scratch       scoring, gouge, crack in surface
  m → misalignment  gap, crooked install, panel not flush
  o → other         doesn't fit above
  k → skip          leave blank for now (can re-label later)
  q → quit          save progress and exit

Usage:
    python label_defects.py
    python label_defects.py --report "path/to/drop_zone_report.csv"
    python label_defects.py --report "path/to/report.csv" --output "labeled.csv"
    python label_defects.py --summary          # show label counts without labeling

Output:
    labeled_defects.csv  — full report with added `defect_type` column
    defect_photos/       — subfolders (bacterial/ caulking/ scratch/ misalignment/ other/)
                           populated with downloaded copies for training
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import webbrowser
from collections import Counter
from pathlib import Path


def _open_url(url: str):
    """Open URL in browser — uses Windows browser when running inside WSL."""
    if 'microsoft' in os.uname().release.lower():
        subprocess.Popen(
            ['cmd.exe', '/c', 'start', '', url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        webbrowser.open(url)

try:
    import requests
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

LABEL_MAP = {
    'b': 'bacterial',
    'c': 'caulking',
    's': 'scratch',
    'm': 'misalignment',
    'o': 'other',
}

LABEL_COLORS = {
    'bacterial':    '\033[93m',   # yellow
    'caulking':     '\033[91m',   # red
    'scratch':      '\033[95m',   # magenta
    'misalignment': '\033[94m',   # blue
    'other':        '\033[96m',   # cyan
}
RESET = '\033[0m'
BOLD  = '\033[1m'


def load_report(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_output(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if 'defect_type' not in fieldnames:
        fieldnames.append('defect_type')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def download_photo(url: str, dest: Path) -> bool:
    if not HAS_PIL:
        return False
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True
    except Exception:
        return False


def prompt_label(row: dict, idx: int, total: int) -> str:
    """Show photo info, open browser, and wait for user input. Returns label key."""
    result = row.get('result', '')
    score  = row.get('quality_score', '')
    model  = row.get('model', '')
    site   = row.get('site_name', '') or row.get('region', '') or '—'
    url    = row.get('first_visit_photo_url', '')
    qr     = row.get('qr_code', '')

    print()
    print(f'{BOLD}── Photo {idx}/{total} ──────────────────────────────────────{RESET}')
    print(f'  Result  : {BOLD}{result}{RESET}   score={score}')
    print(f'  Unit    : {qr}   model={model}')
    print(f'  Site    : {site}')
    print(f'  URL     : {url[:90]}')
    print()
    print('  Opening photo in browser...')
    if url:
        _open_url(url)
    print()
    print(f'  {BOLD}Label:{RESET}')
    print(f'    [b] bacterial     [c] caulking    [s] scratch')
    print(f'    [m] misalignment  [o] other       [k] skip  [q] quit')
    print()

    while True:
        raw = input('  > ').strip().lower()
        if raw in LABEL_MAP:
            color = LABEL_COLORS.get(LABEL_MAP[raw], '')
            print(f'  Tagged: {color}{BOLD}{LABEL_MAP[raw]}{RESET}')
            return LABEL_MAP[raw]
        elif raw == 'k' or raw == '':
            print('  Skipped.')
            return ''
        elif raw == 'q':
            return '__quit__'
        else:
            print(f'  Unknown key "{raw}" — use b / c / s / m / o / k / q')


def print_summary(rows: list[dict]):
    to_label = [r for r in rows if r.get('result') in ('DEFECT', 'UNCERTAIN')]
    labeled  = [r for r in to_label if r.get('defect_type', '').strip()]
    unlabeled = len(to_label) - len(labeled)

    counts = Counter(r.get('defect_type', '') for r in labeled if r.get('defect_type'))
    counts_by_result = {res: Counter() for res in ('DEFECT', 'UNCERTAIN')}
    for r in to_label:
        dt = r.get('defect_type', '').strip()
        if dt:
            counts_by_result[r['result']][dt] += 1

    print(f'\n{BOLD}── Defect Label Summary ────────────────────────────{RESET}')
    print(f'  DEFECT + UNCERTAIN total : {len(to_label)}')
    print(f'  Labeled                  : {len(labeled)}')
    print(f'  Unlabeled / skipped      : {unlabeled}')
    print()
    print(f'  {BOLD}By label:{RESET}')
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        color = LABEL_COLORS.get(label, '')
        print(f'    {color}{label:14s}{RESET} : {count}')
    print()
    print(f'  {BOLD}By result × label:{RESET}')
    for res in ('DEFECT', 'UNCERTAIN'):
        sub = counts_by_result[res]
        if sub:
            print(f'    {res}:')
            for label, count in sorted(sub.items(), key=lambda x: -x[1]):
                color = LABEL_COLORS.get(label, '')
                print(f'      {color}{label:14s}{RESET} : {count}')


def main():
    parser = argparse.ArgumentParser(description='Sub-label drop zone DEFECT and UNCERTAIN photos.')
    parser.add_argument('--report',  default=None, help='Path to drop_zone_report CSV')
    parser.add_argument('--output',  default=None, help='Output CSV path (default: labeled_defects.csv next to report)')
    parser.add_argument('--photos',  default=None, help='Folder to download labeled photos into (optional)')
    parser.add_argument('--summary', action='store_true', help='Show label summary and exit')
    parser.add_argument('--reset',   action='store_true', help='Clear all existing labels and start fresh')
    args = parser.parse_args()

    # ── Locate report ─────────────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
    else:
        # Default: look next to this script
        candidates = list(Path(__file__).parent.glob('*drop_zone*report*.csv'))
        if not candidates:
            candidates = list(Path.cwd().glob('*drop_zone*report*.csv'))
        if not candidates:
            print('No drop zone report CSV found. Pass --report <path>')
            sys.exit(1)
        report_path = sorted(candidates)[-1]
        print(f'Using report: {report_path}')

    output_path = Path(args.output) if args.output else report_path.parent / 'labeled_defects.csv'
    photos_root = Path(args.photos) if args.photos else report_path.parent / 'defect_photos'

    rows = load_report(report_path)

    # ── Merge existing labeled output if present ───────────────────────────────
    if output_path.exists() and not args.reset:
        existing = load_report(output_path)
        existing_labels = {r.get('qr_code', '') + r.get('first_visit_photo_url', ''): r.get('defect_type', '')
                           for r in existing if r.get('defect_type', '').strip()}
        merged = 0
        for r in rows:
            key = r.get('qr_code', '') + r.get('first_visit_photo_url', '')
            if key in existing_labels and not r.get('defect_type', '').strip():
                r['defect_type'] = existing_labels[key]
                merged += 1
        if merged:
            print(f'Resumed — loaded {merged} existing labels from {output_path.name}')

    # Add defect_type column if missing
    for r in rows:
        r.setdefault('defect_type', '')

    if args.summary:
        print_summary(rows)
        return

    # ── Filter rows to label ───────────────────────────────────────────────────
    to_label = [r for r in rows
                if r.get('result') in ('DEFECT', 'UNCERTAIN')
                and not r.get('defect_type', '').strip()]

    already_done = sum(1 for r in rows
                       if r.get('result') in ('DEFECT', 'UNCERTAIN')
                       and r.get('defect_type', '').strip())

    total_flagged = sum(1 for r in rows if r.get('result') in ('DEFECT', 'UNCERTAIN'))

    print(f'\n{BOLD}Drop Zone Defect Sub-Labeler{RESET}')
    print(f'  Report         : {report_path.name}')
    print(f'  DEFECT         : {sum(1 for r in rows if r["result"]=="DEFECT")}')
    print(f'  UNCERTAIN      : {sum(1 for r in rows if r["result"]=="UNCERTAIN")}')
    print(f'  Already labeled: {already_done}')
    print(f'  Remaining      : {len(to_label)}')
    print(f'  Output         : {output_path}')
    if not HAS_PIL:
        print(f'\n  Note: pip install requests pillow  to enable photo download')
    print()

    if not to_label:
        print('All photos already labeled!')
        print_summary(rows)
        save_output(rows, output_path)
        return

    input('  Press Enter to start labeling  (Ctrl+C to abort)...')

    labeled_this_session = 0
    download_counts = Counter()

    for i, row in enumerate(to_label, 1):
        label = prompt_label(row, i + already_done, total_flagged)

        if label == '__quit__':
            print(f'\nSaved progress ({labeled_this_session} labeled this session).')
            break

        if label:
            row['defect_type'] = label
            labeled_this_session += 1

            # Download photo into subfolder
            url = row.get('first_visit_photo_url', '')
            if url and HAS_PIL:
                qr   = (row.get('qr_code') or 'unknown').replace('/', '_')
                ext  = Path(url.split('?')[0]).suffix or '.jpg'
                dest = photos_root / label / f"{qr}_{i:05d}{ext}"
                if download_photo(url, dest):
                    download_counts[label] += 1

        # Save after every photo (safe resume)
        save_output(rows, output_path)

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print(f'{BOLD}Session complete.{RESET}')
    print(f'  Labeled this session : {labeled_this_session}')
    save_output(rows, output_path)
    print(f'  Saved to             : {output_path}')

    if download_counts:
        print(f'\n  Downloaded photos (ready for training):')
        for label, count in sorted(download_counts.items(), key=lambda x: -x[1]):
            print(f'    {photos_root / label}  ({count} photos)')

    print_summary(rows)

    # Training readiness check
    labeled_counts = Counter(r.get('defect_type', '') for r in rows
                             if r.get('result') in ('DEFECT', 'UNCERTAIN')
                             and r.get('defect_type', '').strip())
    print(f'\n{BOLD}Training readiness (need 20+ per category):{RESET}')
    for label in ('bacterial', 'caulking', 'scratch', 'misalignment', 'other'):
        count  = labeled_counts.get(label, 0)
        status = '✓ ready' if count >= 20 else f'need {20 - count} more'
        color  = '\033[92m' if count >= 20 else '\033[90m'
        print(f'  {LABEL_COLORS.get(label,"")}{label:14s}{RESET} : {count:3d}  {color}{status}{RESET}')


if __name__ == '__main__':
    main()

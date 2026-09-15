"""
delete_critical_images.py

Reads audit_report.json, deletes every image referenced by a
CRITICAL-severity finding from the image folder.

Safe defaults:
  - DRY_RUN = True  -> prints what would be deleted, deletes nothing
  - Backs up file list to a manifest first
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# CONFIGURE
# ------------------------------------------------------------------
AUDIT_REPORT    = Path("reports/audit/audit_report.json")
ANNOTATION_PATH = Path("data/annotations/train_annotations.json")
IMAGE_DIR       = Path("data/raw/train")
BACKUP_DIR      = Path(f"reports/deleted_backup_{datetime.now():%Y%m%d_%H%M%S}")

DRY_RUN = False # <<< set to False to actually delete
BACKUP  = True          # <<< set to False to permanently delete

# Categories in the report that reference specific images we can identify.
# Some critical findings don't include filenames (e.g. Zero-Area Boxes),
# so we re-derive them from the COCO annotations instead.
OOB_TOLERANCE = 0       # pixels; overflow beyond this is "real" OOB


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
with open(AUDIT_REPORT) as f:
    findings = json.load(f)

with open(ANNOTATION_PATH) as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
images_by_name = {img["file_name"]: img for img in coco["images"]}


# ------------------------------------------------------------------
# COLLECT BAD FILENAMES
# ------------------------------------------------------------------
bad_files = set()
reason_by_file = {}

def mark(file_name, reason):
    if file_name not in images_by_name:
        return
    bad_files.add(file_name)
    reason_by_file.setdefault(file_name, set()).add(reason)


# --- 1. From the report: findings that store filenames in evidence ---
# These categories list examples as plain filename strings.
FILE_LIST_CATEGORIES = {
    "Missing Images",
    "Corrupt Images",
    "Unreadable by OpenCV",
}

for f in findings:
    if f.get("severity") != "CRITICAL":
        continue
    cat = f.get("category")
    if cat in FILE_LIST_CATEGORIES:
        for name in f.get("evidence", {}).get("examples", []):
            mark(name, cat)


# --- 2. From annotations directly: per-image validation ---
# (Report only stores first 10 examples per finding, so re-scan.)
for ann in coco["annotations"]:
    img = images_by_id.get(ann["image_id"])
    if img is None:
        continue

    x, y, w, h = ann["bbox"]
    W, H = img["width"], img["height"]
    name = img["file_name"]

    if ann["category_id"] not in {c["id"] for c in coco["categories"]}:
        mark(name, "Invalid Category ID")
    if w <= 0 or h <= 0:
        mark(name, "Zero-Area Box")
    if x < 0 or y < 0:
        mark(name, "Negative Coordinate")
    if (x + w) - W > OOB_TOLERANCE or (y + h) - H > OOB_TOLERANCE:
        mark(name, "Out-of-Bounds (severe)")


# ------------------------------------------------------------------
# REPORT / DELETE
# ------------------------------------------------------------------
print(f"Total images flagged for deletion: {len(bad_files)}")
print(f"Out of {len(images_by_name)} total images "
      f"({100 * len(bad_files) / len(images_by_name):.2f}%)")
print()

# Reason breakdown
from collections import Counter
reason_counts = Counter(r for reasons in reason_by_file.values() for r in reasons)
for reason, count in reason_counts.most_common():
    print(f"  {reason:<30} {count}")

print()
if DRY_RUN:
    print("DRY RUN — nothing will be deleted. Set DRY_RUN=False to proceed.")
    print("\nFirst 20 files that would be deleted:")
    for name in sorted(bad_files)[:20]:
        print(f"  {name}  ->  {', '.join(sorted(reason_by_file[name]))}")
    raise SystemExit(0)

# Create backup folder + manifest
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
with open(BACKUP_DIR / "deleted_files.json", "w") as f:
    json.dump(
        [{"file_name": n, "reasons": sorted(reason_by_file[n])} for n in sorted(bad_files)],
        f, indent=2,
    )

# Delete (or move)
deleted, missing = 0, 0
for name in bad_files:
    src = IMAGE_DIR / name
    if not src.exists():
        missing += 1
        continue
    if BACKUP:
        dst = BACKUP_DIR / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    else:
        src.unlink()
    deleted += 1

print(f"\nDone.")
print(f"  Moved/deleted : {deleted}")
print(f"  Not found     : {missing}")
print(f"  Backup dir    : {BACKUP_DIR if BACKUP else '(no backup)'}")
print(f"  Manifest      : {BACKUP_DIR / 'deleted_files.json'}")
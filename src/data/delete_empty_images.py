"""
delete_empty_images.py

Finds images with zero annotations in a COCO file, and either:
  - deletes them all, or
  - keeps a random KEEP_FRACTION as negatives and deletes the rest.

Also removes their (nonexistent) annotations from the COCO json for consistency
(no-op for empty images, but keeps the file clean).
"""

import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ------------------------------------------------------------------
# CONFIGURE
# ------------------------------------------------------------------
ANNOTATION_PATH = Path("data/annotations/train_annotations.json")
IMAGE_DIR       = Path("data/raw/train")
BACKUP_DIR      = Path(f"reports/deleted_empty_{datetime.now():%Y%m%d_%H%M%S}")

DRY_RUN         = False      # <<< set False to actually move files
BACKUP          = True      # <<< set False to permanently delete
KEEP_FRACTION   = 0.00      # <<< fraction of empties to KEEP as negatives
                            #     1.0 = delete none, 0.0 = delete all
RANDOM_SEED     = 42


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
with open(ANNOTATION_PATH) as f:
    coco = json.load(f)

# Which images have at least one annotation?
annotated_image_ids = {ann["image_id"] for ann in coco["annotations"]}

# Empty images = in coco["images"] but never referenced by an annotation
empty_images = [img for img in coco["images"] if img["id"] not in annotated_image_ids]

print(f"Total images       : {len(coco['images'])}")
print(f"Annotated images   : {len(annotated_image_ids)}")
print(f"Empty images       : {len(empty_images)}")
print(f"KEEP_FRACTION      : {KEEP_FRACTION} "
      f"(~{int(len(empty_images) * KEEP_FRACTION)} will be kept as negatives)")
print()


# ------------------------------------------------------------------
# CHOOSE WHICH TO KEEP / DELETE
# ------------------------------------------------------------------
rng = random.Random(RANDOM_SEED)
shuffled = empty_images[:]
rng.shuffle(shuffled)

n_keep = int(len(shuffled) * KEEP_FRACTION)
keep   = shuffled[:n_keep]
delete = shuffled[n_keep:]

print(f"Keeping : {len(keep)} empties as negatives")
print(f"Deleting: {len(delete)} empties")
print()


# ------------------------------------------------------------------
# OPTIONAL: group-by-prefix info (video frames?)
# ------------------------------------------------------------------
def prefix(name: str) -> str:
    # strip ".rf.<hash>.jpg" suffix Roboflow adds, keep base stem
    base = name.split(".rf.")[0] if ".rf." in name else Path(name).stem
    return base

from collections import Counter
dup_prefixes = Counter(prefix(i["file_name"]) for i in delete)
print("Top 10 filename prefixes among deletions (likely video frames):")
for p, c in dup_prefixes.most_common(10):
    print(f"  {c:>4}  {p}")
print()


# ------------------------------------------------------------------
# ACT
# ------------------------------------------------------------------
if DRY_RUN:
    print("DRY RUN — nothing deleted. Set DRY_RUN=False to proceed.")
    print("First 20 to delete:")
    for img in delete[:20]:
        print(f"  {img['file_name']}")
    raise SystemExit(0)

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
with open(BACKUP_DIR / "deleted_empty.json", "w") as f:
    json.dump(
        [{"image_id": i["id"], "file_name": i["file_name"]} for i in delete],
        f, indent=2,
    )
with open(BACKUP_DIR / "kept_empty.json", "w") as f:
    json.dump(
        [{"image_id": i["id"], "file_name": i["file_name"]} for i in keep],
        f, indent=2,
    )

moved, missing = 0, 0
for img in delete:
    src = IMAGE_DIR / img["file_name"]
    if not src.exists():
        missing += 1
        continue
    if BACKUP:
        dst = BACKUP_DIR / img["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    else:
        src.unlink()
    moved += 1

print(f"Done. Moved/deleted: {moved}  |  Not found: {missing}")
print(f"Backup dir: {BACKUP_DIR}")
print()
print("Don't forget to also update the annotation JSON:")
print("  remove deleted image entries from coco['images']")


# ------------------------------------------------------------------
# OPTIONAL: rewrite annotation file without deleted empties
# ------------------------------------------------------------------
# Uncomment to rewrite. Kept empties stay in coco['images'] (no annotations).
#
# deleted_ids = {img["id"] for img in delete}
# coco["images"] = [i for i in coco["images"] if i["id"] not in deleted_ids]
# with open(ANNOTATION_PATH.with_name("train_annotations_cleaned.json"), "w") as f:
#     json.dump(coco, f)
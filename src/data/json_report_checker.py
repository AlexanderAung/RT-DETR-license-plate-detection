"""
extract_oob_images.py

Reads audit_report.json, finds "Out-of-Bounds Boxes" findings,
and copies the referenced images into an output folder with
a drawn bbox preview so you can eyeball them quickly.
"""

import json
import shutil
from pathlib import Path

import cv2

# ------------------------------------------------------------------
# CONFIGURE
# ------------------------------------------------------------------
AUDIT_REPORT   = Path("reports/audit/audit_report.json")
ANNOTATION_PATH = Path("data/annotations/train_annotations.json")
IMAGE_DIR      = Path("data/raw/train")
OUT_DIR        = Path("reports/oob_review")
DRAW_BOXES     = True   # also save a preview with bbox drawn

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
with open(AUDIT_REPORT) as f:
    findings = json.load(f)

with open(ANNOTATION_PATH) as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
anns_by_id   = {ann["id"]: ann for ann in coco["annotations"]}


# ------------------------------------------------------------------
# COLLECT OOB ANNOTATIONS
# ------------------------------------------------------------------
oob_examples = []
for f in findings:
    if f.get("category") == "Out-of-Bounds Boxes":
        oob_examples.extend(f.get("evidence", {}).get("examples", []))

print(f"Found {len(oob_examples)} OOB examples in the report")

# Note: audit only stores the first 10 examples per finding.
# If you want ALL 533, re-scan the COCO file directly:
all_oob_anns = []
for ann in coco["annotations"]:
    img = images_by_id.get(ann["image_id"])
    if img is None:
        continue
    x, y, w, h = ann["bbox"]
    if x + w > img["width"] or y + h > img["height"] or x < 0 or y < 0:
        all_oob_anns.append(ann)

print(f"Full re-scan found {len(all_oob_anns)} OOB annotations")

# Use the full list (swap to oob_examples if you only want report entries)
target_anns = all_oob_anns

# Group by image so we copy each image once
anns_per_image = {}
for ann in target_anns:
    anns_per_image.setdefault(ann["image_id"], []).append(ann)

print(f"These span {len(anns_per_image)} unique images")


# ------------------------------------------------------------------
# COPY + ANNOTATE
# ------------------------------------------------------------------
manifest = []

for img_id, anns in anns_per_image.items():
    img_info = images_by_id[img_id]
    src = IMAGE_DIR / img_info["file_name"]

    if not src.exists():
        print(f"  [missing] {src}")
        continue

    # Copy original
    dst = OUT_DIR / img_info["file_name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    # Draw preview
    if DRAW_BOXES:
        img = cv2.imread(str(src))
        if img is not None:
            h_img, w_img = img.shape[:2]
            for ann in anns:
                x, y, w, h = map(int, ann["bbox"])
                # Draw actual bbox in red
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Clamp and draw clipped region in green
                x2, y2 = min(x + w, w_img - 1), min(y + h, h_img - 1)
                cv2.rectangle(img, (max(x, 0), max(y, 0)), (x2, y2), (0, 255, 0), 1)

            preview_name = f"{Path(img_info['file_name']).stem}__oob.jpg"
            cv2.imwrite(str(OUT_DIR / preview_name), img)

    manifest.append({
        "image_id": img_id,
        "file_name": img_info["file_name"],
        "img_size": [img_info["width"], img_info["height"]],
        "n_oob_anns": len(anns),
        "oob_anns": [
            {
                "ann_id": a["id"],
                "bbox": a["bbox"],
                "overflow_x": max(0, a["bbox"][0] + a["bbox"][2] - img_info["width"]),
                "overflow_y": max(0, a["bbox"][1] + a["bbox"][3] - img_info["height"]),
            }
            for a in anns
        ],
    })

# Save manifest
with open(OUT_DIR / "oob_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone. {len(manifest)} images copied to {OUT_DIR}")
print(f"Manifest: {OUT_DIR / 'oob_manifest.json'}")
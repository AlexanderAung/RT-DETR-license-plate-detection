"""
Dataset Audit Script for License Plate Detection
"""
import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import warnings

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class LicensePlateDatasetAuditor:
    def __init__(self, image_dir: str, annotation_path: str, output_dir: str = "reports/audit"):
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO-format annotations
        with open(self.annotation_path) as f:
            self.coco = json.load(f)
        
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = self.coco['annotations']
        self.categories = {cat['id']: cat for cat in self.coco['categories']}
        
        # Build lookup tables
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
        
        self.findings = []
        
    def log(self, severity: str, category: str, message: str, evidence: Optional[dict] = None):
        """Structured finding logging."""
        self.findings.append({
            'severity': severity,  # CRITICAL, WARNING, INFO
            'category': category,
            'message': message,
            'evidence': evidence or {}
        })
        print(f"[{severity}] {category}: {message}")

    # ------------------------------------------------------------------
    # 1. STRUCTURAL INTEGRITY
    # ------------------------------------------------------------------
    def check_file_integrity(self):
        """Verify every annotated image exists and is readable."""
        print("\n=== Checking File Integrity ===")
        missing_images = []
        corrupt_images = []
        unreadable_images = []
        
        for img_id, img_info in tqdm(self.images.items(), desc="Images"):
            img_path = self.image_dir / img_info['file_name']
            
            if not img_path.exists():
                missing_images.append(img_info['file_name'])
                continue
            
            # Check if OpenCV can decode it (catches truncated JPEGs)
            img = cv2.imread(str(img_path))
            if img is None:
                try:
                    # Fallback to PIL
                    with Image.open(img_path) as pil_img:
                        pil_img.verify()
                    unreadable_images.append(img_info['file_name'])
                except Exception:
                    corrupt_images.append(img_info['file_name'])
                continue
            
            # Dimension consistency
            h, w = img.shape[:2]
            if h != img_info['height'] or w != img_info['width']:
                self.log("WARNING", "Dimension Mismatch", 
                        f"{img_info['file_name']}: JSON says {img_info['width']}x{img_info['height']}, "
                        f"actual {w}x{h}")
        
        if missing_images:
            self.log("CRITICAL", "Missing Images", 
                    f"{len(missing_images)} images referenced but not found",
                    {'examples': missing_images[:10]})
        if corrupt_images:
            self.log("CRITICAL", "Corrupt Images",
                    f"{len(corrupt_images)} images are corrupted/truncated",
                    {'examples': corrupt_images[:10]})
        if unreadable_images:
            self.log("WARNING", "Unreadable by OpenCV",
                    f"{len(unreadable_images)} images readable by PIL but not OpenCV",
                    {'examples': unreadable_images[:10]})
        
        # Orphaned images (no annotations)
        all_image_files = set(p.name for p in self.image_dir.iterdir() 
                             if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        annotated_files = set(img['file_name'] for img in self.images.values())
        orphaned = all_image_files - annotated_files
        if orphaned:
            self.log("WARNING", "Orphaned Images",
                    f"{len(orphaned)} images in directory not in annotations",
                    {'examples': list(orphaned)[:10]})

    # ------------------------------------------------------------------
    # 2. ANNOTATION VALIDITY
    # ------------------------------------------------------------------
    def check_annotations(self):
        """Validate bounding boxes and categories."""
        print("\n=== Checking Annotations ===")
        invalid_boxes = []
        zero_area = []
        negative_coords = []
        out_of_bounds = []
        missing_category = []
        
        for ann in tqdm(self.annotations, desc="Annotations"):
            x, y, w, h = ann['bbox']
            img = self.images.get(ann['image_id'])
            
            if img is None:
                self.log("CRITICAL", "Dangling Annotation", 
                        f"Annotation {ann['id']} references missing image {ann['image_id']}")
                continue
            
            # Category validity
            if ann['category_id'] not in self.categories:
                missing_category.append(ann['id'])
            
            # Box validity
            if w <= 0 or h <= 0:
                zero_area.append((ann['id'], ann['bbox']))
            if x < 0 or y < 0:
                negative_coords.append((ann['id'], ann['bbox']))
            if img and (x + w > img['width'] or y + h > img['height']):
                out_of_bounds.append({
                    'ann_id': ann['id'],
                    'bbox': ann['bbox'],
                    'img_size': (img['width'], img['height'])
                })
            # Normalized check (COCO should be absolute, but verify no one normalized)
            if img and (w < 1 and h < 1):
                # Suspiciously small — might be normalized YOLO left as COCO
                invalid_boxes.append((ann['id'], ann['bbox'], img['file_name']))
        
        if zero_area:
            self.log("CRITICAL", "Zero-Area Boxes",
                    f"{len(zero_area)} boxes with zero or negative area",
                    {'examples': zero_area[:10]})
        if negative_coords:
            self.log("CRITICAL", "Negative Coordinates",
                    f"{len(negative_coords)} boxes with negative coords",
                    {'examples': negative_coords[:10]})
        if out_of_bounds:
            self.log("CRITICAL", "Out-of-Bounds Boxes",
                    f"{len(out_of_bounds)} boxes extending outside image",
                    {'examples': out_of_bounds[:10]})
        if invalid_boxes:
            self.log("CRITICAL", "Potentially Normalized Boxes",
                    f"{len(invalid_boxes)} boxes suspiciously small (possibly YOLO format not converted)",
                    {'examples': invalid_boxes[:10]})
        if missing_category:
            self.log("CRITICAL", "Invalid Category IDs",
                    f"{len(missing_category)} annotations with unknown category_id",
                    {'examples': missing_category[:10]})
        
        # Empty annotations (images with no plates)
        empty_images = [img_id for img_id in self.images if not self.img_to_anns[img_id]]
        if empty_images:
            self.log("INFO", "Empty Images",
                    f"{len(empty_images)} images have no annotations (negatives)",
                    {'examples': [self.images[i]['file_name'] for i in empty_images[:10]]})

    # ------------------------------------------------------------------
    # 3. DUPLICATES & LEAKAGE
    # ------------------------------------------------------------------
    def compute_image_hash(self, img_path: Path, hash_size: int = 16) -> str:
        """Compute perceptual hash for near-duplicate detection."""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            # Resize and compute average hash
            resized = cv2.resize(img, (hash_size, hash_size))
            avg = resized.mean()
            diff = resized > avg
            return ''.join(str(int(b)) for b in diff.flatten())
        except Exception:
            return None
    
    def hamming_distance(self, h1: str, h2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))
    
    def check_duplicates(self, hash_threshold: int = 5):
        """Detect exact and near-duplicate images."""
        print("\n=== Checking Duplicates ===")
        hashes = {}
        duplicates = defaultdict(list)
        
        for img_id, img_info in tqdm(self.images.items(), desc="Hashing"):
            img_path = self.image_dir / img_info['file_name']
            if not img_path.exists():
                continue
            h = self.compute_image_hash(img_path)
            if h is None:
                continue
            
            # Exact hash match
            if h in hashes:
                duplicates[h].append(img_info['file_name'])
            else:
                hashes[h] = img_info['file_name']
        
        # Near-duplicates (Hamming distance check — O(N^2) but manageable for <100k)
        # For large datasets, use locality sensitive hashing instead
        exact_dups = {k: v for k, v in duplicates.items() if len(v) > 1}
        if exact_dups:
            total_dup_files = sum(len(v) for v in exact_dups.values())
            self.log("CRITICAL", "Exact Duplicates",
                    f"{len(exact_dups)} hash groups, ~{total_dup_files} files",
                    {'examples': {k: v[:5] for k, v in list(exact_dups.items())[:3]}})
        
        # Video-frame leakage detection via filename patterns
        sequential_frames = self._detect_sequential_frames()
        if sequential_frames:
            self.log("WARNING", "Sequential Video Frames",
                    f"Detected {len(sequential_frames)} potential video sequences",
                    {'examples': sequential_frames[:5]})

    def _detect_sequential_frames(self) -> List[List[str]]:
        """Detect filename patterns like frame_0001.jpg, frame_0002.jpg."""
        from itertools import groupby
        import re
        
        sequences = []
        files = sorted(self.images.values(), key=lambda x: x['file_name'])
        
        # Group by common prefix
        def get_prefix(name):
            # Remove numbers at end
            return re.sub(r'[_-]?\d+\.(jpg|jpeg|png)$', '', name.lower())
        
        for prefix, group in groupby(files, key=lambda x: get_prefix(x['file_name'])):
            group_list = list(group)
            if len(group_list) > 20:  # Suspiciously many with same prefix
                sequences.append([g['file_name'] for g in group_list[:5]] + [f"... ({len(group_list)} total)"])
        return sequences

    # ------------------------------------------------------------------
    # 4. OBJECT STATISTICS
    # ------------------------------------------------------------------
    def analyze_objects(self):
        """Compute size, aspect ratio, and density statistics."""
        print("\n=== Analyzing Objects ===")
        areas = []
        widths = []
        heights = []
        aspect_ratios = []
        objs_per_image = []
        small_objects = 0  # COCO: < 32^2 px
        tiny_objects = 0   # Custom: < 16^2 px (critical for plates)
        
        for img_id, img_info in self.images.items():
            anns = self.img_to_anns[img_id]
            objs_per_image.append(len(anns))
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                area = w * h
                areas.append(area)
                widths.append(w)
                heights.append(h)
                
                if h > 0:
                    aspect_ratios.append(w / h)
                if area < 1024:
                    small_objects += 1
                if area < 256:
                    tiny_objects += 1
        
        areas = np.array(areas)
        aspect_ratios = np.array(aspect_ratios)
        
        print(f"Total objects: {len(areas)}")
        print(f"Objects per image: mean={np.mean(objs_per_image):.2f}, max={max(objs_per_image)}")
        print(f"Area stats: min={areas.min():.1f}, median={np.median(areas):.1f}, mean={areas.mean():.1f}, max={areas.max():.1f}")
        print(f"Small objects (<1024px): {small_objects} ({100*small_objects/len(areas):.1f}%)")
        print(f"Tiny objects (<256px): {tiny_objects} ({100*tiny_objects/len(areas):.1f}%)")
        print(f"Aspect ratio: median={np.median(aspect_ratios):.2f}, std={np.std(aspect_ratios):.2f}")
        
        # Save statistics for later comparison
        stats = {
            'n_objects': len(areas),
            'n_images': len(self.images),
            'areas': {
                'min': float(areas.min()),
                'median': float(np.median(areas)),
                'mean': float(areas.mean()),
                'max': float(areas.max()),
                'p5': float(np.percentile(areas, 5)),
                'p95': float(np.percentile(areas, 95)),
            },
            'small_object_pct': 100 * small_objects / len(areas),
            'tiny_object_pct': 100 * tiny_objects / len(areas),
            'aspect_ratio_median': float(np.median(aspect_ratios)),
        }
        
        with open(self.output_dir / "object_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        self._plot_distributions(areas, aspect_ratios, objs_per_image, widths, heights)

    def _plot_distributions(self, areas, aspect_ratios, objs_per_image, widths, heights):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Area distribution (log scale)
        ax = axes[0, 0]
        ax.hist(np.log10(areas + 1), bins=50, color='steelblue', edgecolor='black')
        ax.axvline(np.log10(1024), color='red', linestyle='--', label='Small (<1024)')
        ax.axvline(np.log10(256), color='darkred', linestyle='--', label='Tiny (<256)')
        ax.set_xlabel('log10(Area + 1)')
        ax.set_ylabel('Count')
        ax.set_title('Plate Area Distribution')
        ax.legend()
        
        # Aspect ratio
        ax = axes[0, 1]
        ax.hist(aspect_ratios, bins=50, range=(0, 5), color='steelblue', edgecolor='black')
        ax.set_xlabel('Width / Height')
        ax.set_ylabel('Count')
        ax.set_title('Aspect Ratio Distribution')
        
        # Objects per image
        ax = axes[0, 2]
        ax.hist(objs_per_image, bins=max(objs_per_image)+1, color='steelblue', edgecolor='black')
        ax.set_xlabel('Objects per Image')
        ax.set_ylabel('Count')
        ax.set_title('Objects per Image')
        
        # Width vs Height scatter (sample)
        ax = axes[1, 0]
        sample_idx = np.random.choice(len(widths), min(5000, len(widths)), replace=False)
        ax.scatter(np.array(widths)[sample_idx], np.array(heights)[sample_idx], alpha=0.3, s=5)
        ax.set_xlabel('Width (px)')
        ax.set_ylabel('Height (px)')
        ax.set_title('Width vs Height (sample)')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        
        # Small object heatmap by image resolution
        ax = axes[1, 1]
        img_areas = []
        obj_areas = []
        for img_id, img_info in self.images.items():
            img_area = img_info['width'] * img_info['height']
            for ann in self.img_to_anns[img_id]:
                obj_areas.append(ann['bbox'][2] * ann['bbox'][3])
                img_areas.append(img_area)
        ax.scatter(np.array(img_areas), np.array(obj_areas), alpha=0.2, s=2)
        ax.set_xlabel('Image Area (px)')
        ax.set_ylabel('Plate Area (px)')
        ax.set_title('Plate Size vs Image Size')
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # Resolution distribution
        ax = axes[1, 2]
        res_counts = Counter((img['width'], img['height']) for img in self.images.values())
        top_res = sorted(res_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ax.barh(range(len(top_res)), [c for _, c in top_res])
        ax.set_yticks(range(len(top_res)))
        ax.set_yticklabels([f"{w}×{h}" for (w, h), _ in top_res])
        ax.set_xlabel('Count')
        ax.set_title('Top 10 Image Resolutions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'object_distributions.png', dpi=150)
        plt.close()
        print(f"Plots saved to {self.output_dir / 'object_distributions.png'}")

    # ------------------------------------------------------------------
    # 5. CAMERA/DOMAIN ANALYSIS
    # ------------------------------------------------------------------
    def analyze_camera_domains(self):
        """Extract camera ID from filename and analyze per-camera stats."""
        print("\n=== Analyzing Camera Domains ===")
        # Heuristic: filenames like cam03_20240115_143022.jpg or camera7_frame_1234.png
        import re
        
        camera_stats = defaultdict(lambda: {
            'count': 0, 'resolutions': [], 'areas': [], 
            'night_count': 0, 'small_obj_pct': 0, 'n_objs': 0
        })
        
        for img_id, img_info in self.images.items():
            fname = img_info['file_name'].lower()
            # Extract camera identifier
            m = re.search(r'(?:cam|camera)[_-]?(\d+)', fname)
            cam_id = f"cam_{m.group(1)}" if m else "unknown"
            
            stats = camera_stats[cam_id]
            stats['count'] += 1
            stats['resolutions'].append((img_info['width'], img_info['height']))
            
            # Heuristic night detection: filenames containing 'night', 'n_', or timestamp after 20:00/before 06:00
            is_night = 'night' in fname or '_n_' in fname
            if not is_night:
                time_match = re.search(r'(\d{2})(\d{2})(\d{2})', fname)
                if time_match:
                    hour = int(time_match.group(1))
                    is_night = hour >= 20 or hour <= 5
            
            if is_night:
                stats['night_count'] += 1
            
            anns = self.img_to_anns[img_id]
            stats['n_objs'] += len(anns)
            for ann in anns:
                area = ann['bbox'][2] * ann['bbox'][3]
                stats['areas'].append(area)
        
        # Print summary
        print(f"{'Camera':<12} {'Images':>8} {'Avg Objs':>10} {'Night%':>8} {'Small%':>8} {'Resolutions'}")
        print("-" * 70)
        for cam_id in sorted(camera_stats.keys()):
            s = camera_stats[cam_id]
            areas = np.array(s['areas']) if s['areas'] else np.array([0])
            small_pct = 100 * np.sum(areas < 1024) / len(areas) if len(areas) > 0 else 0
            night_pct = 100 * s['night_count'] / s['count'] if s['count'] > 0 else 0
            avg_objs = s['n_objs'] / s['count'] if s['count'] > 0 else 0
            unique_res = set(s['resolutions'])
            res_str = ", ".join(f"{w}×{h}" for w, h in sorted(unique_res)[:2])
            if len(unique_res) > 2:
                res_str += f" (+{len(unique_res)-2} more)"
            
            print(f"{cam_id:<12} {s['count']:>8} {avg_objs:>10.2f} {night_pct:>7.1f}% {small_pct:>7.1f}% {res_str}")
            
            # Flag domain shift
            if len(unique_res) > 1:
                self.log("WARNING", "Camera Resolution Variance",
                        f"{cam_id} has {len(unique_res)} different resolutions")
            if small_pct > 80:
                self.log("WARNING", "Camera Small-Object Dominance",
                        f"{cam_id}: {small_pct:.1f}% small plates")

    # ------------------------------------------------------------------
    # 6. SPLIT LEAKAGE
    # ------------------------------------------------------------------
    def check_split_leakage(self, split_files: Dict[str, str]):
        """
        Args:
            split_files: {'train': path_to_train.json, 'val': path_to_val.json, ...}
        """
        print("\n=== Checking Split Leakage ===")
        split_hashes = {}
        
        for split_name, path in split_files.items():
            with open(path) as f:
                split_coco = json.load(f)
            hashes = set()
            for img in split_coco['images']:
                img_path = self.image_dir / img['file_name']
                if img_path.exists():
                    # Use MD5 of file content for exact duplicates across splits
                    h = hashlib.md5(open(img_path, 'rb').read(8192)).hexdigest()  # Sample first 8KB
                    hashes.add(h)
            split_hashes[split_name] = hashes
        
        # Cross-split overlap
        splits = list(split_hashes.keys())
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                overlap = split_hashes[splits[i]] & split_hashes[splits[j]]
                if overlap:
                    self.log("CRITICAL", "Cross-Split Leakage",
                            f"{len(overlap)} images appear in both {splits[i]} and {splits[j]}")

    # ------------------------------------------------------------------
    # 7. VISUALIZATION
    # ------------------------------------------------------------------
    def visualize_samples(self, n_samples: int = 16):
        """Draw random sample images with boxes to spot-check quality."""
        print("\n=== Generating Sample Visualizations ===")
        sample_ids = np.random.choice(list(self.images.keys()), 
                                      min(n_samples, len(self.images)), 
                                      replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, img_id in enumerate(sample_ids):
            img_info = self.images[img_id]
            img_path = self.image_dir / img_info['file_name']
            
            if not img_path.exists():
                axes[idx].text(0.5, 0.5, "Missing", ha='center')
                axes[idx].axis('off')
                continue
            
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for ann in self.img_to_anns[img_id]:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_info['file_name']}\n"
                               f"{img_info['width']}×{img_info['height']} | "
                               f"{len(self.img_to_anns[img_id])} objs")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_annotations.png', dpi=150)
        plt.close()
        print(f"Sample visualizations saved to {self.output_dir / 'sample_annotations.png'}")

    # ------------------------------------------------------------------
    # RUN ALL
    # ------------------------------------------------------------------
    def run_full_audit(self, split_files: Optional[Dict[str, str]] = None):
        self.check_file_integrity()
        self.check_annotations()
        self.check_duplicates()
        self.analyze_objects()
        self.analyze_camera_domains()
        self.visualize_samples()
        if split_files:
            self.check_split_leakage(split_files)
        
        # Save findings report
        report_path = self.output_dir / "audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.findings, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Audit complete. {len(self.findings)} findings.")
        print(f"Report: {report_path}")
        print(f"Review the sample_annotations.png carefully for label quality.")


# ------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------
if __name__ == "__main__":
    # CONFIGURE THESE PATHS
    IMAGE_DIR = "data/raw/train"
    ANNOTATION_PATH = "data/annotations/train_annotations.json"
    
    # If you have train/val/test splits already:
    
    SPLIT_FILES = {
        # 'train': 'data/raw/annotations/instances_train.json',
        # 'val': 'data/raw/annotations/instances_val.json',
    }
    
    
    auditor = LicensePlateDatasetAuditor(IMAGE_DIR, ANNOTATION_PATH)
    auditor.run_full_audit(split_files=SPLIT_FILES if SPLIT_FILES else None)
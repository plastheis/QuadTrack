#!/usr/bin/env python3
"""
Convert Det-Fly dataset from Pascal VOC XML format to YOLO format.

Output structure:
  training/datasets/Det-Fly-yolo/
    images/train/      (symlinks to original images)
    images/val/
    labels/train/      (YOLO .txt files)
    labels/val/
    data.yaml          (dataset descriptor)
"""

import os
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

# --- CONFIGURATION ---
SRC_DIR = Path("data/raw/Det-Fly/det_Fly-imgs")
DST_DIR = Path("data/detection/detfly")

CLASSES = ["UAV"]  # single class

# --- HELPER ---


def xml_to_yolo_bbox(xml_path, img_w, img_h):
    """
    Parse a Pascal VOC XML file and return list of YOLO-format lines.
    Each line: <class_id> <x_center> <y_center> <width> <height>  (normalized)
    Skips objects marked as 'difficult' = 1 (standard YOLO practice).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []
    for obj in root.findall("object"):
        # Optionally skip difficult objects
        # difficult_elem = obj.find("difficult")
        # if difficult_elem is not None and difficult_elem.text == "1":
        #     continue

        name = obj.find("name").text.strip()
        cls_id = CLASSES.index(name) if name in CLASSES else -1
        if cls_id == -1:
            print(f"  WARNING: unknown class '{name}' in {xml_path.name}")
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized center + width/height)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def convert_split(split_name):
    """Convert one split (train/val) from XML to YOLO labels."""
    xml_dir = SRC_DIR / "Annotations" / split_name
    img_src_dir = SRC_DIR / "Images" / split_name
    img_dst_dir = DST_DIR / "images" / split_name
    lbl_dst_dir = DST_DIR / "labels" / split_name

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    stats = {"total": len(xml_files), "with_objects": 0, "empty": 0, "errors": 0}

    for xml_path in xml_files:
        try:
            base_name = xml_path.stem  # e.g., "0200002"
            img_name = base_name + ".jpg"
            img_src = img_src_dir / img_name
            img_dst = img_dst_dir / img_name
            lbl_dst = lbl_dst_dir / (base_name + ".txt")

            if not img_src.exists():
                print(f"  WARNING: image not found for {xml_path.name}")
                stats["errors"] += 1
                continue

            # Symlink image (relative to avoid breakage on moves)
            target_rel = os.path.relpath(img_src.resolve(), img_dst_dir.resolve())
            if not img_dst.exists():
                os.symlink(target_rel, img_dst)

            # Parse XML -> YOLO lines
            lines = xml_to_yolo_bbox(xml_path, img_w=3840, img_h=2160)

            if lines:
                stats["with_objects"] += 1
                with open(lbl_dst, "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                stats["empty"] += 1
                # YOLO expects empty .txt for images with no objects
                with open(lbl_dst, "w") as f:
                    pass

        except Exception as e:
            print(f"  ERROR processing {xml_path.name}: {e}")
            stats["errors"] += 1

    return stats


def create_data_yaml():
    """Write the data.yaml descriptor file."""
    yaml_path = DST_DIR / "data.yaml"

    content = {
        "path": str(DST_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
    }

    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated {yaml_path}")


# --- MAIN ---

if __name__ == "__main__":
    print("=" * 60)
    print("  Det-Fly → YOLO Format Converter")
    print("=" * 60)

    for split in ["train", "val"]:
        print(f"\n--- Converting {split} split ---")
        stats = convert_split(split)
        print(
            f"  Total annotations: {stats['total']}"
        )
        print(
            f"  With objects:      {stats['with_objects']}"
        )
        print(
            f"  Empty (no objs):   {stats['empty']}"
        )
        if stats["errors"]:
            print(f"  Errors:            {stats['errors']}")

    create_data_yaml()

    # Final summary
    print("\n" + "=" * 60)
    print("  Output structure:")
    print(f"    {DST_DIR}/")
    print("      ├── images/")
    print("      │   ├── train/  (symlinks)")
    print("      │   └── val/    (symlinks)")
    print("      ├── labels/")
    print("      │   ├── train/  (.txt)")
    print("      │   └── val/    (.txt)")
    print("      └── data.yaml")
    print("=" * 60)

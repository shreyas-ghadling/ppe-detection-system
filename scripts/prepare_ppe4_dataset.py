"""Prepare a four-class PPE dataset from the raw SH-17 dataset."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "SH-17"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "PPE-4Class"

CLASS_MAPPING = {
    0: 0,   # person
    10: 1,  # helmet
    16: 2,  # safety-vest
    5: 3,   # face-mask-medical
}
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety-vest",
    3: "face-mask-medical",
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def read_split_file(path: Path) -> list[str]:
    """Read and validate image filenames from a split manifest."""
    filenames = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    filenames = [filename for filename in filenames if filename]

    unsupported = [
        filename
        for filename in filenames
        if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if unsupported:
        examples = ", ".join(unsupported[:5])
        raise ValueError(f"{path} contains unsupported image filenames: {examples}")

    return filenames


def filter_label_file(path: Path) -> tuple[list[str], Counter[int]]:
    """Return remapped YOLO rows while preserving coordinate tokens exactly."""
    filtered_rows: list[str] = []
    object_counts: Counter[int] = Counter()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Invalid YOLO row in {path} at line {line_number}: {line!r}"
            )

        try:
            original_class_id = int(fields[0])
        except ValueError as error:
            raise ValueError(
                f"Invalid class ID in {path} at line {line_number}: {fields[0]!r}"
            ) from error

        new_class_id = CLASS_MAPPING.get(original_class_id)
        if new_class_id is None:
            continue

        filtered_rows.append(f"{new_class_id} {' '.join(fields[1:])}")
        object_counts[new_class_id] += 1

    return filtered_rows, object_counts


def prepare_split(
    filenames: list[str],
    output_split_name: str,
    staging_root: Path,
) -> tuple[int, int, Counter[int]]:
    """Copy images and write labels that contain at least one target object."""
    output_images = staging_root / output_split_name / "images"
    output_labels = staging_root / output_split_name / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    object_counts: Counter[int] = Counter()

    for filename in filenames:
        image_path = SOURCE_ROOT / "images" / filename
        label_path = SOURCE_ROOT / "labels" / f"{Path(filename).stem}.txt"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing source label: {label_path}")

        filtered_rows, image_object_counts = filter_label_file(label_path)
        if not filtered_rows:
            skipped += 1
            continue

        shutil.copy2(image_path, output_images / filename)
        output_label_path = output_labels / f"{Path(filename).stem}.txt"
        output_label_path.write_text(
            "\n".join(filtered_rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        kept += 1
        object_counts.update(image_object_counts)

    return kept, skipped, object_counts


def write_data_yaml(staging_root: Path) -> None:
    """Write an Ultralytics-compatible dataset configuration."""
    yaml_text = """train: train/images
val: valid/images

nc: 4
names:
  0: person
  1: helmet
  2: safety-vest
  3: face-mask-medical
"""
    (staging_root / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    train_filenames = read_split_file(SOURCE_ROOT / "train_files.txt")
    val_filenames = read_split_file(SOURCE_ROOT / "val_files.txt")

    staging_root = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    try:
        kept_train, skipped_train, train_counts = prepare_split(
            train_filenames,
            "train",
            staging_root,
        )
        kept_val, skipped_val, val_counts = prepare_split(
            val_filenames,
            "valid",
            staging_root,
        )
        write_data_yaml(staging_root)

        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
        staging_root.replace(OUTPUT_ROOT)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    total_counts = train_counts + val_counts
    print("PPE-4Class dataset preparation complete")
    print(f"input train count: {len(train_filenames)}")
    print(f"input val count: {len(val_filenames)}")
    print(f"kept train images: {kept_train}")
    print(f"kept val images: {kept_val}")
    print(f"skipped train images: {skipped_train}")
    print(f"skipped val images: {skipped_val}")
    print("object counts per new class:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id} {class_name}: {total_counts[class_id]}")


if __name__ == "__main__":
    main()

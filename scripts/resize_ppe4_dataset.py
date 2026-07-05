"""Create a capped-resolution JPG copy of the PPE-4Class dataset.

Images are resized only when their longest side exceeds 1280 pixels. Labels
are copied byte-for-byte, and neither source dataset is modified.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "PPE-4Class"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "PPE-4Class-Resized"

SPLITS = ("train", "valid")
SUPPORTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
MAX_LONGEST_SIDE = 1280
JPEG_QUALITY = 90
PROGRESS_INTERVAL = 500


@dataclass
class ResizeStatistics:
    """Aggregate image counts and dimensions before and after resizing."""

    image_count: int = 0
    resized_count: int = 0
    before_width: int = 0
    before_height: int = 0
    after_width: int = 0
    after_height: int = 0

    def record(
        self,
        original_width: int,
        original_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Add one processed image to the aggregate statistics."""
        self.image_count += 1
        self.before_width += original_width
        self.before_height += original_height
        self.after_width += output_width
        self.after_height += output_height
        if (original_width, original_height) != (output_width, output_height):
            self.resized_count += 1

    @property
    def average_before(self) -> tuple[float, float]:
        """Return average source width and height."""
        if not self.image_count:
            return 0.0, 0.0
        return (
            self.before_width / self.image_count,
            self.before_height / self.image_count,
        )

    @property
    def average_after(self) -> tuple[float, float]:
        """Return average output width and height."""
        if not self.image_count:
            return 0.0, 0.0
        return (
            self.after_width / self.image_count,
            self.after_height / self.image_count,
        )


def list_images(directory: Path) -> list[Path]:
    """Return supported image files in stable order."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_source_structure() -> None:
    """Ensure all required source directories and files exist."""
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_ROOT}")
    if not (SOURCE_ROOT / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing source data.yaml: {SOURCE_ROOT / 'data.yaml'}")

    for split in SPLITS:
        image_directory = SOURCE_ROOT / split / "images"
        label_directory = SOURCE_ROOT / split / "labels"
        if not image_directory.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_directory}")
        if not label_directory.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_directory}")

        images = list_images(image_directory)
        image_stems = [path.stem.casefold() for path in images]
        if len(image_stems) != len(set(image_stems)):
            raise ValueError(
                f"{image_directory} contains duplicate stems that would collide as JPG"
            )

        label_stems = {
            path.stem.casefold()
            for path in label_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        missing_labels = [
            image.name
            for image in images
            if image.stem.casefold() not in label_stems
        ]
        if missing_labels:
            examples = ", ".join(missing_labels[:5])
            raise ValueError(
                f"{split} has {len(missing_labels)} image(s) without labels: {examples}"
            )


def capped_dimensions(width: int, height: int) -> tuple[int, int]:
    """Calculate aspect-ratio-preserving dimensions with no upscaling."""
    longest_side = max(width, height)
    if longest_side <= MAX_LONGEST_SIDE:
        return width, height

    scale = MAX_LONGEST_SIDE / longest_side
    return max(1, round(width * scale)), max(1, round(height * scale))


def process_split(
    split: str,
    staging_root: Path,
    statistics: ResizeStatistics,
) -> None:
    """Resize one image split and copy all of its labels unchanged."""
    source_images = SOURCE_ROOT / split / "images"
    source_labels = SOURCE_ROOT / split / "labels"
    output_images = staging_root / split / "images"
    output_labels = staging_root / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(source_images)
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")

        original_height, original_width = image.shape[:2]
        output_width, output_height = capped_dimensions(
            original_width,
            original_height,
        )

        if (output_width, output_height) != (original_width, original_height):
            image = cv2.resize(
                image,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )

        output_path = output_images / f"{image_path.stem}.jpg"
        written = cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not written:
            raise OSError(f"OpenCV could not write image: {output_path}")

        statistics.record(
            original_width,
            original_height,
            output_width,
            output_height,
        )
        if index % PROGRESS_INTERVAL == 0 or index == len(image_paths):
            print(f"{split}: processed {index}/{len(image_paths)} images")

    # copy2 preserves both file content and available filesystem metadata.
    for label_path in sorted(source_labels.glob("*.txt")):
        shutil.copy2(label_path, output_labels / label_path.name)


def write_data_yaml(staging_root: Path) -> None:
    """Write an Ultralytics configuration pointing at the resized dataset."""
    final_dataset_path = OUTPUT_ROOT.resolve().as_posix()
    yaml_text = f"""path: {final_dataset_path}
train: train/images
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


def directory_size(directory: Path) -> int:
    """Return the total size of all files below a directory in bytes."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def format_size(byte_count: int) -> str:
    """Format a byte count using binary units."""
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("Unreachable")


def print_summary(
    source_size: int,
    output_size: int,
    statistics: ResizeStatistics,
) -> None:
    """Print dataset size and image-dimension changes."""
    before_width, before_height = statistics.average_before
    after_width, after_height = statistics.average_after

    print()
    print("=" * 44)
    print("PPE-4Class Resize Summary")
    print("=" * 44)
    print(f"Images processed: {statistics.image_count}")
    print(f"Images resized: {statistics.resized_count}")
    print(f"Images kept at original dimensions: "
          f"{statistics.image_count - statistics.resized_count}")
    print(f"Before dataset size: {format_size(source_size)} ({source_size} bytes)")
    print(f"After dataset size: {format_size(output_size)} ({output_size} bytes)")
    print(f"Average dimensions before: {before_width:.2f} x {before_height:.2f}")
    print(f"Average dimensions after: {after_width:.2f} x {after_height:.2f}")
    print(f"Output dataset: {OUTPUT_ROOT}")


def main() -> None:
    """Build the resized dataset in staging and publish it on success."""
    validate_source_structure()
    source_size = directory_size(SOURCE_ROOT)
    statistics = ResizeStatistics()
    staging_root = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.tmp")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    try:
        for split in SPLITS:
            process_split(split, staging_root, statistics)
        write_data_yaml(staging_root)

        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
        staging_root.replace(OUTPUT_ROOT)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    output_size = directory_size(OUTPUT_ROOT)
    print_summary(source_size, output_size, statistics)


if __name__ == "__main__":
    main()

"""Audit the processed PPE-4Class dataset before YOLO training.

The script never modifies the dataset. Its only output files are ten annotated
sample images written to ``outputs/dataset_samples``.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "PPE-4Class"
SAMPLES_ROOT = PROJECT_ROOT / "outputs" / "dataset_samples"

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety-vest",
    3: "face-mask-medical",
}
DISPLAY_NAMES = {
    0: "Person",
    1: "Helmet",
    2: "Safety Vest",
    3: "Face Mask",
}
# OpenCV uses BGR color order.
CLASS_COLORS = {
    0: (60, 200, 60),
    1: (255, 150, 30),
    2: (0, 165, 255),
    3: (220, 60, 220),
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
SPLITS = {"Train": "train", "Validation": "valid"}
SAMPLE_COUNT = 10
RANDOM_SEED = 42


@dataclass(frozen=True)
class Annotation:
    """One validated normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitAudit:
    """Validation results and statistics for one dataset split."""

    display_name: str
    directory_name: str
    image_paths: list[Path] = field(default_factory=list)
    label_paths: list[Path] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    integrity_errors: list[str] = field(default_factory=list)
    invalid_annotations: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    object_counts: Counter[int] = field(default_factory=Counter)
    images_per_class: Counter[int] = field(default_factory=Counter)
    annotations_by_stem: dict[str, list[Annotation]] = field(default_factory=dict)
    images_by_stem: dict[str, Path] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        """Return the number of valid objects found in the split."""
        return sum(self.object_counts.values())

    @property
    def average_objects_per_image(self) -> float:
        """Return valid objects divided by all images in the split."""
        if not self.image_paths:
            return 0.0
        return self.object_count / len(self.image_paths)


def keyed_by_stem(paths: list[Path]) -> tuple[dict[str, Path], list[str]]:
    """Index paths by case-insensitive stem and report ambiguous duplicates."""
    indexed: dict[str, Path] = {}
    duplicate_errors: list[str] = []

    for path in paths:
        key = path.stem.casefold()
        if key in indexed:
            duplicate_errors.append(
                f"Duplicate basename: {indexed[key].name} and {path.name}"
            )
        else:
            indexed[key] = path

    return indexed, duplicate_errors


def validate_label_file(path: Path) -> tuple[list[Annotation], list[str], bool]:
    """Parse a YOLO label file and return valid annotations and error messages."""
    annotations: list[Annotation] = []
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    nonempty_lines = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]

    for line_number, line in nonempty_lines:
        fields = line.split()
        location = f"{path.parent.parent.name}/labels/{path.name}:{line_number}"

        if len(fields) != 5:
            errors.append(
                f"{location}: expected 5 fields, found {len(fields)} ({line!r})"
            )
            continue

        try:
            class_id = int(fields[0])
        except ValueError:
            errors.append(f"{location}: invalid class ID {fields[0]!r}")
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"{location}: class ID {class_id} is not one of {sorted(CLASS_NAMES)}"
            )
            continue

        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{location}: coordinates must be numeric ({line!r})")
            continue

        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{location}: coordinates must be finite ({line!r})")
            continue

        x_center, y_center, width, height = coordinates
        coordinate_names = ("x_center", "y_center", "width", "height")
        out_of_range = [
            f"{name}={value}"
            for name, value in zip(coordinate_names, coordinates)
            if not 0.0 <= value <= 1.0
        ]
        if out_of_range:
            errors.append(
                f"{location}: coordinates outside [0,1]: {', '.join(out_of_range)}"
            )
            continue

        if width <= 0.0 or height <= 0.0:
            errors.append(
                f"{location}: width and height must be greater than 0 "
                f"(width={width}, height={height})"
            )
            continue

        annotations.append(
            Annotation(class_id, x_center, y_center, width, height)
        )

    return annotations, errors, not nonempty_lines


def audit_split(display_name: str, directory_name: str) -> SplitAudit:
    """Check image/label pairing, validate labels, and calculate statistics."""
    report = SplitAudit(display_name=display_name, directory_name=directory_name)
    split_root = DATASET_ROOT / directory_name
    image_root = split_root / "images"
    label_root = split_root / "labels"

    if not image_root.is_dir():
        report.integrity_errors.append(f"Missing directory: {image_root}")
    if not label_root.is_dir():
        report.integrity_errors.append(f"Missing directory: {label_root}")
    if report.integrity_errors:
        return report

    report.image_paths = sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    report.label_paths = sorted(
        path
        for path in label_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )

    images_by_stem, image_duplicates = keyed_by_stem(report.image_paths)
    labels_by_stem, label_duplicates = keyed_by_stem(report.label_paths)
    report.images_by_stem = images_by_stem
    report.integrity_errors.extend(
        f"{directory_name}/images: {message}" for message in image_duplicates
    )
    report.integrity_errors.extend(
        f"{directory_name}/labels: {message}" for message in label_duplicates
    )

    for key in sorted(images_by_stem.keys() - labels_by_stem.keys()):
        image_path = images_by_stem[key]
        report.missing_labels.append(
            f"{directory_name}/images/{image_path.name} -> "
            f"expected {image_path.stem}.txt"
        )

    for key in sorted(labels_by_stem.keys() - images_by_stem.keys()):
        label_path = labels_by_stem[key]
        report.missing_images.append(
            f"{directory_name}/labels/{label_path.name} -> expected matching image"
        )

    # Validate every label, including orphan labels, so all problems are reported.
    for key, label_path in sorted(labels_by_stem.items()):
        annotations, errors, is_empty = validate_label_file(label_path)
        report.invalid_annotations.extend(errors)
        report.annotations_by_stem[key] = annotations

        if is_empty:
            report.empty_labels.append(f"{directory_name}/labels/{label_path.name}")

        classes_in_image: set[int] = set()
        for annotation in annotations:
            report.object_counts[annotation.class_id] += 1
            classes_in_image.add(annotation.class_id)
        for class_id in classes_in_image:
            report.images_per_class[class_id] += 1

    return report


def draw_annotations(image, annotations: list[Annotation]):
    """Draw colored class boxes and labels on an OpenCV image."""
    image_height, image_width = image.shape[:2]
    thickness = max(2, round(min(image_width, image_height) / 500))
    font_scale = max(0.5, min(image_width, image_height) / 1200)

    for annotation in annotations:
        half_width = annotation.width * image_width / 2
        half_height = annotation.height * image_height / 2
        x1 = round(annotation.x_center * image_width - half_width)
        y1 = round(annotation.y_center * image_height - half_height)
        x2 = round(annotation.x_center * image_width + half_width)
        y2 = round(annotation.y_center * image_height + half_height)

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))

        color = CLASS_COLORS[annotation.class_id]
        class_name = CLASS_NAMES[annotation.class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        (text_width, text_height), baseline = cv2.getTextSize(
            class_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_top = max(0, y1 - text_height - baseline - 6)
        text_bottom = min(image_height - 1, text_top + text_height + baseline + 6)
        text_right = min(image_width - 1, x1 + text_width + 8)
        cv2.rectangle(
            image,
            (x1, text_top),
            (text_right, text_bottom),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            image,
            class_name,
            (x1 + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image


def create_random_visualizations(
    reports: list[SplitAudit],
) -> tuple[list[Path], list[str]]:
    """Select ten paired images and save annotated copies without opening windows."""
    candidates: list[tuple[SplitAudit, str, Path]] = []
    for report in reports:
        for stem, image_path in report.images_by_stem.items():
            if stem in report.annotations_by_stem:
                candidates.append((report, stem, image_path))

    if len(candidates) < SAMPLE_COUNT:
        return [], [
            f"Only {len(candidates)} paired images are available; "
            f"{SAMPLE_COUNT} are required for visualization"
        ]

    randomizer = random.Random(RANDOM_SEED)
    selected = randomizer.sample(sorted(candidates, key=lambda item: str(item[2])), SAMPLE_COUNT)
    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    errors: list[str] = []
    for index, (report, stem, image_path) in enumerate(selected, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"OpenCV could not read {image_path}")
            continue

        annotated_image = draw_annotations(
            image,
            report.annotations_by_stem[stem],
        )
        output_path = SAMPLES_ROOT / (
            f"sample_{index:02d}_{report.directory_name}_{image_path.stem}.jpg"
        )
        if not cv2.imwrite(str(output_path), annotated_image):
            errors.append(f"OpenCV could not write {output_path}")
            continue
        saved_paths.append(output_path)

    return saved_paths, errors


def print_items(title: str, items: list[str]) -> None:
    """Print a count followed by details, or a clear None marker."""
    print(f"{title}: {len(items)}")
    if items:
        for item in items:
            print(f"  - {item}")
    else:
        print("  None")


def print_split_statistics(report: SplitAudit) -> None:
    """Print detailed statistics for a train or validation split."""
    print(f"{report.display_name} Statistics")
    print("-" * (len(report.display_name) + 11))
    print(f"Images: {len(report.image_paths)}")
    print(f"Label files: {len(report.label_paths)}")
    print(f"Objects: {report.object_count}")
    print("Images containing each class:")
    for class_id, display_name in DISPLAY_NAMES.items():
        print(f"  {display_name}: {report.images_per_class[class_id]}")
    print("Object count per class:")
    for class_id, display_name in DISPLAY_NAMES.items():
        print(f"  {display_name}: {report.object_counts[class_id]}")
    print(f"Average objects/image: {report.average_objects_per_image:.4f}")
    print()


def print_report(
    reports: list[SplitAudit],
    structure_errors: list[str],
    visualization_paths: list[Path],
    visualization_errors: list[str],
) -> bool:
    """Print the complete audit report and return whether the dataset passed."""
    missing_images = [item for report in reports for item in report.missing_images]
    missing_labels = [item for report in reports for item in report.missing_labels]
    integrity_errors = [
        *structure_errors,
        *(item for report in reports for item in report.integrity_errors),
    ]
    invalid_annotations = [
        item for report in reports for item in report.invalid_annotations
    ]
    empty_labels = [item for report in reports for item in report.empty_labels]

    failure_reasons: list[str] = []
    if integrity_errors:
        failure_reasons.append(f"{len(integrity_errors)} structure/integrity error(s)")
    if missing_images:
        failure_reasons.append(f"{len(missing_images)} label file(s) missing an image")
    if missing_labels:
        failure_reasons.append(f"{len(missing_labels)} image(s) missing a label")
    if invalid_annotations:
        failure_reasons.append(
            f"{len(invalid_annotations)} invalid annotation line(s)"
        )
    if empty_labels:
        failure_reasons.append(
            f"{len(empty_labels)} empty label file(s); PPE-4Class should only "
            "contain images with target objects"
        )
    if visualization_errors:
        failure_reasons.append(
            f"{len(visualization_errors)} visualization error(s)"
        )

    passed = not failure_reasons
    print("=" * 40)
    print("PPE-4Class Dataset Audit")
    print("=" * 40)
    print()
    print(f"Train Images: {len(reports[0].image_paths)}")
    print(f"Validation Images: {len(reports[1].image_paths)}")
    print()

    print_items("Missing Images", missing_images)
    print_items("Missing Labels", missing_labels)
    print_items("Integrity Errors", integrity_errors)
    print_items("Invalid Labels", invalid_annotations)
    print_items("Empty Labels", empty_labels)
    print()

    for report in reports:
        print_split_statistics(report)

    total_object_counts: Counter[int] = Counter()
    for report in reports:
        total_object_counts.update(report.object_counts)

    print("Class Distribution")
    print("------------------")
    for class_id, display_name in DISPLAY_NAMES.items():
        print(f"{display_name}: {total_object_counts[class_id]}")
    print()
    print("Average Objects/Image:")
    for report in reports:
        print(
            f"  {report.display_name}: "
            f"{report.average_objects_per_image:.4f}"
        )
    print()

    print(f"Random Visualizations: {len(visualization_paths)} saved")
    print(f"Visualization Directory: {SAMPLES_ROOT}")
    print_items("Visualization Errors", visualization_errors)
    print()

    print(f"Dataset Status: {'PASS' if passed else 'FAIL'}")
    if failure_reasons:
        print("Failure Reasons:")
        for reason in failure_reasons:
            print(f"  - {reason}")

    return passed


def main() -> int:
    """Run the complete dataset audit and return a process exit code."""
    structure_errors: list[str] = []
    if not DATASET_ROOT.is_dir():
        structure_errors.append(f"Missing dataset directory: {DATASET_ROOT}")
    if not (DATASET_ROOT / "data.yaml").is_file():
        structure_errors.append(f"Missing dataset configuration: {DATASET_ROOT / 'data.yaml'}")

    reports = [
        audit_split(display_name, directory_name)
        for display_name, directory_name in SPLITS.items()
    ]
    visualization_paths, visualization_errors = create_random_visualizations(reports)
    passed = print_report(
        reports,
        structure_errors,
        visualization_paths,
        visualization_errors,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

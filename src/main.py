"""Workflow entry point for extracting PDF images and checking reuse."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from detector import (
	configure_logger as configure_detection_logger,
	extract_features,
	find_similar_regions,
	list_image_files,
	prepare_matcher,
	save_summary,
)
from pdf2pic import configure_logger as configure_pdf_logger, extract_directory


def setup_workflow_logger(verbose: bool) -> logging.Logger:
	logger = logging.getLogger("picSensor.workflow")
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)
	logger.handlers.clear()

	handler = logging.StreamHandler()
	handler.setLevel(logging.DEBUG if verbose else logging.INFO)
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger


def run_detection(
	image_dir: Path,
	output_dir: Path,
	patterns: Sequence[str],
	max_features: int,
	min_matches: int,
	min_inliers: int,
	ratio_threshold: float,
	ransac_threshold: float,
	max_candidates: int,
	verbose: bool,
) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)
	logger = configure_detection_logger(output_dir / "picSensor.log", verbose=verbose)

	image_files = list_image_files(image_dir, patterns)
	if not image_files:
		logger.warning("No images available in %s", image_dir)
		return 0

	logger.info("Extracting features from %d images", len(image_files))
	features = extract_features(image_files, max_features, logger)
	matcher, dataset_map = prepare_matcher(features, logger)
	if matcher is None:
		logger.error("Matcher has no descriptors. Aborting detection stage.")
		return 0

	logger.info("Searching for similar regions across extracted images")
	results = find_similar_regions(
		features,
		matcher,
		dataset_map,
		ratio_threshold,
		min_matches,
		max_candidates,
		ransac_threshold,
		min_inliers,
		output_dir,
		logger,
	)

	if results:
		save_summary(results, output_dir)
		logger.info(
			"Detection complete. %d candidate reuses recorded in %s",
			len(results),
			output_dir / "match_summary.json",
		)
	else:
		logger.info("No suspicious similarities detected.")

	return len(results)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--pdf-dir",
		type=Path,
		default=Path("data/pdf"),
		help="Directory containing source PDF files.",
	)
	parser.add_argument(
		"--image-dir",
		type=Path,
		default=Path("data/input"),
		help="Directory to store extracted images.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/output"),
		help="Directory for detection outputs.",
	)
	parser.add_argument(
		"--clear-images",
		action="store_true",
		help="Remove existing files in the image directory before extraction.",
	)
	parser.add_argument(
		"--patterns",
		nargs="*",
		default=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"),
		help="Glob patterns to consider during detection.",
	)
	parser.add_argument("--max-features", type=int, default=800)
	parser.add_argument("--min-matches", type=int, default=24)
	parser.add_argument("--min-inliers", type=int, default=12)
	parser.add_argument("--ratio-threshold", type=float, default=0.75)
	parser.add_argument("--ransac-threshold", type=float, default=5.0)
	parser.add_argument("--max-candidates", type=int, default=5)
	parser.add_argument("--min-edge", type=int, default=32)
	parser.add_argument(
		"--pdf-workers",
		type=int,
		default=0,
		help="Worker threads for PDF extraction (0 = auto).",
	)
	parser.add_argument("--verbose", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	workflow_logger = setup_workflow_logger(args.verbose)
	pdf_logger = configure_pdf_logger(args.verbose)

	if not args.pdf_dir.exists():
		workflow_logger.error("PDF directory does not exist: %s", args.pdf_dir)
		return

	workflow_logger.info("Starting image extraction from PDFs")
	extracted_paths = extract_directory(
		pdf_dir=args.pdf_dir,
		output_dir=args.image_dir,
		clear_output=args.clear_images,
		min_edge=args.min_edge,
		workers=args.pdf_workers,
		logger=pdf_logger,
	)

	if extracted_paths:
		workflow_logger.info("Extracted %d images from PDFs", len(extracted_paths))
	else:
		workflow_logger.warning(
			"No images were extracted. Existing files in %s will be analysed instead.",
			args.image_dir,
		)

	workflow_logger.info("Running similarity detection")
	matches = run_detection(
		image_dir=args.image_dir,
		output_dir=args.output_dir,
		patterns=args.patterns,
		max_features=args.max_features,
		min_matches=args.min_matches,
		min_inliers=args.min_inliers,
		ratio_threshold=args.ratio_threshold,
		ransac_threshold=args.ransac_threshold,
		max_candidates=args.max_candidates,
		verbose=args.verbose,
	)

	if matches:
		workflow_logger.info(
			"Detected %d potentially reused images. Review %s for details.",
			matches,
			args.output_dir / "match_summary.json",
		)
	else:
		workflow_logger.info("Finished without detecting problematic overlaps.")


if __name__ == "__main__":  # pragma: no cover
	main()

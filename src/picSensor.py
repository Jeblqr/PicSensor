"""picSensor: detect similar local regions across images using feature matching.

This script scans all images inside ``data/input`` (configurable), extracts ORB
features, and searches for other images that share locally similar content even
under perspective transformations. For each confident match it creates visual
annotations inside ``data/output`` and logs the details for further inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
	import cv2  # type: ignore[import-not-found]
except ImportError as exc:
	raise ImportError("OpenCV (cv2) is required to run picSensor.") from exc

try:
	import numpy as np  # type: ignore[import-not-found]
except ImportError as exc:
	raise ImportError("NumPy is required to run picSensor.") from exc


@dataclass
class ImageFeatures:
	"""Container for the features computed per image."""

	index: int
	path: Path
	keypoints: Sequence[cv2.KeyPoint]
	descriptors: Optional[np.ndarray]
	shape: Tuple[int, int]

	@property
	def name(self) -> str:
		return self.path.name


def configure_logger(log_file: Path, verbose: bool) -> logging.Logger:
	"""Initialise the root logger with both console and file handlers."""

	log_file.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("picSensor")
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)

	# Clear any existing handlers set by previous runs in the same interpreter.
	logger.handlers.clear()

	formatter = logging.Formatter(
		fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
	)

	fh = logging.FileHandler(log_file, encoding="utf-8")
	fh.setFormatter(formatter)
	fh.setLevel(logging.DEBUG)
	logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	ch.setLevel(logging.DEBUG if verbose else logging.INFO)
	logger.addHandler(ch)

	logger.debug("Logger configured. Log file at %s", log_file)
	return logger


def list_image_files(input_dir: Path, patterns: Sequence[str]) -> List[Path]:
	"""Enumerate image files matching the provided glob patterns."""

	files: List[Path] = []
	for pattern in patterns:
		files.extend(sorted(input_dir.glob(pattern)))
	unique_files = sorted({f.resolve() for f in files if f.is_file()})
	return unique_files


def load_image(path: Path) -> Optional[np.ndarray]:
	"""Load an image from disk in BGR format."""

	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		return None
	return image


def extract_features(
	image_paths: Sequence[Path],
	max_features: int,
	logger: logging.Logger,
) -> List[ImageFeatures]:
	"""Compute ORB features for every image path."""

	orb = cv2.ORB_create(nfeatures=max_features)
	features: List[ImageFeatures] = []
	for path in image_paths:
		image = load_image(path)
		if image is None:
			logger.warning("Skipping unreadable image: %s", path)
			continue

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		keypoints, descriptors = orb.detectAndCompute(gray, None)
		if not keypoints:
			logger.info("No keypoints found for %s", path.name)

		features.append(
			ImageFeatures(
				index=len(features),
				path=path,
				keypoints=keypoints,
				descriptors=descriptors,
				shape=tuple(gray.shape),
			)
		)

	logger.info("Extracted features for %d images", len(features))
	return features


def create_flann_matcher() -> cv2.FlannBasedMatcher:
	"""Create a FLANN matcher configured for ORB descriptors (LSH)."""

	index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
	search_params = dict(checks=64)
	matcher = cv2.FlannBasedMatcher(index_params, search_params)
	return matcher


def prepare_matcher(
	features: Sequence[ImageFeatures],
	logger: logging.Logger,
) -> Tuple[Optional[cv2.FlannBasedMatcher], List[int]]:
	"""Add all descriptors to the matcher and keep dataset->image mapping."""

	matcher = create_flann_matcher()
	dataset_to_image: List[int] = []

	for feat in features:
		descriptors = feat.descriptors
		if descriptors is None or len(descriptors) == 0:
			continue

		if descriptors.dtype != np.uint8:
			descriptors = descriptors.astype(np.uint8)

		matcher.add([descriptors])
		dataset_to_image.append(feat.index)

	if not dataset_to_image:
		logger.warning("No descriptors were added to the matcher.")
		return None, []

	matcher.train()
	logger.debug("Matcher prepared with %d descriptor sets", len(dataset_to_image))
	return matcher, dataset_to_image


def compute_bounding_box(
	points: np.ndarray,
	image_shape: Tuple[int, int],
	padding: int = 10,
) -> Tuple[int, int, int, int]:
	"""Create an (x1, y1, x2, y2) bounding box around provided points."""

	if points.size == 0:
		return (0, 0, image_shape[1] - 1, image_shape[0] - 1)

	xs = points[:, 0]
	ys = points[:, 1]

	x1 = max(int(np.floor(xs.min())) - padding, 0)
	y1 = max(int(np.floor(ys.min())) - padding, 0)
	x2 = min(int(np.ceil(xs.max())) + padding, image_shape[1] - 1)
	y2 = min(int(np.ceil(ys.max())) + padding, image_shape[0] - 1)
	return x1, y1, x2, y2


def ensure_color(image: np.ndarray) -> np.ndarray:
	"""Guarantee that the image has three channels for drawing."""

	if image.ndim == 2:
		return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	if image.shape[2] == 3:
		return image
	return image[:, :, :3]


def draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
	"""Draw an in-place rectangle for the detected region."""

	x1, y1, x2, y2 = bbox
	cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def visualize_match(
	src_feat: ImageFeatures,
	dst_feat: ImageFeatures,
	matches: Sequence[cv2.DMatch],
	inlier_mask: np.ndarray,
	output_dir: Path,
	pair_id: str,
	logger: logging.Logger,
) -> Dict[str, object]:
	"""Create visual artefacts for the match."""

	src_raw = load_image(src_feat.path)
	dst_raw = load_image(dst_feat.path)

	if src_raw is None or dst_raw is None:
		logger.warning("Failed to reload images for visualisation: %s vs %s", src_feat.name, dst_feat.name)
		return {}

	src_image = ensure_color(src_raw)
	dst_image = ensure_color(dst_raw)

	inlier_indices = [idx for idx, flag in enumerate(inlier_mask.ravel()) if flag]
	if not inlier_indices:
		return {}

	src_points = np.float32([
		src_feat.keypoints[matches[idx].queryIdx].pt for idx in inlier_indices
	])
	dst_points = np.float32([
		dst_feat.keypoints[matches[idx].trainIdx].pt for idx in inlier_indices
	])

	bbox_src = compute_bounding_box(src_points, src_feat.shape)
	bbox_dst = compute_bounding_box(dst_points, dst_feat.shape)

	annotated_src = src_image.copy()
	annotated_dst = dst_image.copy()
	draw_bbox(annotated_src, bbox_src, (0, 255, 0))
	draw_bbox(annotated_dst, bbox_dst, (0, 255, 0))

	annotated_dir = output_dir / "regions"
	matches_dir = output_dir / "matches"
	annotated_dir.mkdir(parents=True, exist_ok=True)
	matches_dir.mkdir(parents=True, exist_ok=True)

	region_src_path = annotated_dir / f"{pair_id}_src_{src_feat.name}.png"
	region_dst_path = annotated_dir / f"{pair_id}_dst_{dst_feat.name}.png"

	cv2.imwrite(str(region_src_path), annotated_src)
	cv2.imwrite(str(region_dst_path), annotated_dst)

	# Draw matches with inliers highlighted.
	mask_list = inlier_mask.ravel().tolist()
	match_viz = cv2.drawMatches(
		annotated_src,
		list(src_feat.keypoints),
		annotated_dst,
		list(dst_feat.keypoints),
		list(matches),
		None,
		matchesMask=mask_list,
		flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)
	matches_path = matches_dir / f"{pair_id}_{src_feat.name}_vs_{dst_feat.name}.png"
	cv2.imwrite(str(matches_path), match_viz)

	return {
		"region_src": str(region_src_path.relative_to(output_dir)),
		"region_dst": str(region_dst_path.relative_to(output_dir)),
		"matches": str(matches_path.relative_to(output_dir)),
		"bbox_src": list(bbox_src),
		"bbox_dst": list(bbox_dst),
	}


def evaluate_matches(
	src_feat: ImageFeatures,
	dst_feat: ImageFeatures,
	matches: Sequence[cv2.DMatch],
	ransac_thresh: float,
	min_inliers: int,
	logger: logging.Logger,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
	"""Perform geometric verification on matches and return inlier mask."""

	if len(matches) < max(4, min_inliers):
		return None

	src_pts = np.float32([src_feat.keypoints[m.queryIdx].pt for m in matches])
	dst_pts = np.float32([dst_feat.keypoints[m.trainIdx].pt for m in matches])

	homography, mask = cv2.findHomography(
		src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh
	)

	if homography is None or mask is None:
		return None

	inliers = int(mask.ravel().sum())
	if inliers < min_inliers:
		logger.debug(
			"Rejected match %s vs %s: only %d inliers",
			src_feat.name,
			dst_feat.name,
			inliers,
		)
		return None

	return homography, mask


def find_similar_regions(
	features: Sequence[ImageFeatures],
	matcher: cv2.FlannBasedMatcher,
	dataset_to_image: Sequence[int],
	ratio_threshold: float,
	min_matches: int,
	max_candidates: int,
	ransac_thresh: float,
	min_inliers: int,
	output_dir: Path,
	logger: logging.Logger,
) -> List[Dict[str, object]]:
	"""Search for similar regions between images and produce artefacts."""

	results: List[Dict[str, object]] = []
	processed_pairs: set[Tuple[int, int]] = set()

	for src_feat in features:
		desc = src_feat.descriptors
		if desc is None or len(desc) == 0:
			continue

		matches = matcher.knnMatch(desc, k=2)
		candidate_matches: Dict[int, List[cv2.DMatch]] = defaultdict(list)

		for pair in matches:
			if len(pair) < 2:
				continue
			best, second = pair
			target_dataset_idx = best.imgIdx
			if target_dataset_idx >= len(dataset_to_image):
				continue
			target_image_idx = dataset_to_image[target_dataset_idx]
			if target_image_idx == src_feat.index:
				continue
			if best.distance >= ratio_threshold * second.distance:
				continue
			candidate_matches[target_image_idx].append(best)

		sorted_candidates = sorted(
			candidate_matches.items(), key=lambda item: len(item[1]), reverse=True
		)[:max_candidates]

		for target_idx, match_list in sorted_candidates:
			if len(match_list) < min_matches:
				continue
			pair_key = (
				min(src_feat.index, target_idx),
				max(src_feat.index, target_idx),
			)
			if pair_key in processed_pairs:
				continue

			dst_feat = features[target_idx]
			evaluation = evaluate_matches(
				src_feat, dst_feat, match_list, ransac_thresh, min_inliers, logger
			)
			if evaluation is None:
				continue
			homography, inlier_mask = evaluation
			artefacts = visualize_match(
				src_feat,
				dst_feat,
				match_list,
				inlier_mask,
				output_dir,
				pair_id=f"match_{src_feat.index:05d}_{target_idx:05d}",
				logger=logger,
			)

			result_entry = {
				"source": src_feat.name,
				"target": dst_feat.name,
				"source_index": src_feat.index,
				"target_index": target_idx,
				"num_matches": len(match_list),
				"num_inliers": int(inlier_mask.ravel().sum()),
				"homography": homography.tolist(),
				"artefacts": artefacts,
			}
			logger.info(
				"Match found: %s vs %s | matches=%d inliers=%d",
				src_feat.name,
				dst_feat.name,
				result_entry["num_matches"],
				result_entry["num_inliers"],
			)
			processed_pairs.add(pair_key)
			results.append(result_entry)

	return results


def save_summary(results: Sequence[Dict[str, object]], output_dir: Path) -> None:
	"""Persist a JSON summary of all matches."""

	summary_path = output_dir / "match_summary.json"
	with summary_path.open("w", encoding="utf-8") as fh:
		json.dump(results, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/input"),
		help="Directory containing input images.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/output"),
		help="Directory where outputs will be stored.",
	)
	parser.add_argument(
		"--patterns",
		nargs="*",
		default=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"),
		help="Glob patterns for image discovery.",
	)
	parser.add_argument(
		"--max-features",
		type=int,
		default=800,
		help="Maximum ORB features per image.",
	)
	parser.add_argument(
		"--min-matches",
		type=int,
		default=24,
		help="Minimum descriptor matches before geometric verification.",
	)
	parser.add_argument(
		"--min-inliers",
		type=int,
		default=12,
		help="Minimum inliers after RANSAC to accept a match.",
	)
	parser.add_argument(
		"--ratio-threshold",
		type=float,
		default=0.75,
		help="Lowe ratio test threshold.",
	)
	parser.add_argument(
		"--ransac-threshold",
		type=float,
		default=5.0,
		help="RANSAC reprojection threshold in pixels.",
	)
	parser.add_argument(
		"--max-candidates",
		type=int,
		default=5,
		help="Maximum candidate target images per source image.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose console logging.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output
	output_dir.mkdir(parents=True, exist_ok=True)

	log_file = output_dir / "picSensor.log"
	logger = configure_logger(log_file, verbose=args.verbose)

	input_dir = args.input
	if not input_dir.exists():
		logger.error("Input directory does not exist: %s", input_dir)
		return

	image_files = list_image_files(input_dir, args.patterns)
	if not image_files:
		logger.warning("No images found in %s", input_dir)
		return

	logger.info("Discovered %d images. Extracting features...", len(image_files))
	features = extract_features(image_files, args.max_features, logger)
	matcher, dataset_to_image = prepare_matcher(features, logger)
	if matcher is None:
		logger.error("Aborting: matcher has no descriptors to work with.")
		return

	logger.info("Searching for similar regions across images...")
	results = find_similar_regions(
		features,
		matcher,
		dataset_to_image,
		args.ratio_threshold,
		args.min_matches,
		args.max_candidates,
		args.ransac_threshold,
		args.min_inliers,
		output_dir,
		logger,
	)

	if results:
		save_summary(results, output_dir)
		logger.info("Completed with %d matched pairs. Summary saved to %s", len(results), output_dir / "match_summary.json")
	else:
		logger.info("No significant matches were found.")


if __name__ == "__main__":
	main()

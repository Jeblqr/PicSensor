"""Utilities to extract raster images from PDF documents."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence

import os
import threading

try:
	import fitz  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - imported at runtime
	raise ImportError(
		"PyMuPDF (imported as 'fitz') is required. Install it via micromamba or pip."
	) from exc


def configure_logger(verbose: bool = False) -> logging.Logger:
	logger = logging.getLogger("pdf2pic")
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)
	logger.handlers.clear()

	handler = logging.StreamHandler()
	handler.setLevel(logging.DEBUG if verbose else logging.INFO)
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger


def _clean_output_dir(output_dir: Path) -> None:
	for item in output_dir.iterdir():
		if item.is_file():
			item.unlink()


class HashRegistry:
	"""Thread-safe registry to avoid writing duplicate images."""

	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._seen: set[str] = set()

	def register(self, digest: str) -> bool:
		with self._lock:
			if digest in self._seen:
				return False
			self._seen.add(digest)
			return True


def extract_images_from_pdf(
	pdf_path: Path,
	output_dir: Path,
	hash_registry: Optional[HashRegistry] = None,
	min_edge: int = 32,
) -> List[Path]:
	output_paths: List[Path] = []
	doc = fitz.open(pdf_path)
	try:
		for page_index, page in enumerate(doc, start=1):
			image_list = page.get_images(full=True)
			for img_index, image_info in enumerate(image_list, start=1):
				xref = image_info[0]
				extracted = doc.extract_image(xref)
				image_bytes: bytes = extracted["image"]
				width: int = extracted.get("width", 0)
				height: int = extracted.get("height", 0)
				if min(width, height) < min_edge:
					continue

				digest = hashlib.md5(image_bytes).hexdigest()
				if hash_registry is not None and not hash_registry.register(digest):
					continue

				ext = extracted.get("ext", "png").lower()
				filename = (
					f"{pdf_path.stem}_p{page_index:03d}_img{img_index:02d}_{digest[:8]}.{ext}"
				)
				output_path = output_dir / filename
				with output_path.open("wb") as fh:
					fh.write(image_bytes)
				output_paths.append(output_path)
	finally:
		doc.close()
	return output_paths


def extract_directory(
	pdf_dir: Path,
	output_dir: Path,
	clear_output: bool = False,
	min_edge: int = 32,
	workers: int = 0,
	logger: Optional[logging.Logger] = None,
) -> List[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	if clear_output:
		_clean_output_dir(output_dir)

	if logger is None:
		logger = logging.getLogger("pdf2pic")

	pdf_files: List[Path] = sorted(f for f in pdf_dir.glob("*.pdf") if f.is_file())
	if not pdf_files:
		logger.warning("No PDF files found in %s", pdf_dir)
		return []

	hash_registry = HashRegistry()
	extracted_paths: List[Path] = []

	if workers <= 0:
		workers = max(1, (os.cpu_count() or 1) - 1)
		workers = max(workers, 1)

	if workers == 1:
		for pdf_file in pdf_files:
			logger.info("Processing %s", pdf_file.name)
			new_paths = extract_images_from_pdf(pdf_file, output_dir, hash_registry, min_edge)
			logger.info("Extracted %d images from %s", len(new_paths), pdf_file.name)
			extracted_paths.extend(new_paths)
	else:
		logger.info("Using %d worker threads for PDF extraction", workers)

		def task(pdf_file: Path) -> List[Path]:
			return extract_images_from_pdf(pdf_file, output_dir, hash_registry, min_edge)

		with ThreadPoolExecutor(max_workers=workers) as executor:
			future_map = {executor.submit(task, pdf_file): pdf_file for pdf_file in pdf_files}
			for future in as_completed(future_map):
				pdf_file = future_map[future]
				try:
					new_paths = future.result()
					logger.info("Extracted %d images from %s", len(new_paths), pdf_file.name)
					extracted_paths.extend(new_paths)
				except Exception as exc:  # pragma: no cover - best effort logging
					logger.exception("Failed to process %s: %s", pdf_file.name, exc)

	logger.info("Total extracted images: %d", len(extracted_paths))
	return extracted_paths


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--pdf-dir",
		type=Path,
		default=Path("data/pdf"),
		help="Directory containing PDF files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/input"),
		help="Directory where extracted images will be stored.",
	)
	parser.add_argument(
		"--clear-output",
		action="store_true",
		help="Remove existing files in the output directory before extraction.",
	)
	parser.add_argument(
		"--min-edge",
		type=int,
		default=32,
		help="Skip images whose shortest edge is smaller than this value (pixels).",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=0,
		help="Number of worker threads for PDF extraction (0 = auto).",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose logging output.",
	)
	return parser.parse_args(args=args)


def main() -> None:
	cli_args = parse_args()
	logger = configure_logger(verbose=cli_args.verbose)

	if not cli_args.pdf_dir.exists():
		logger.error("PDF directory does not exist: %s", cli_args.pdf_dir)
		return

	extract_directory(
		pdf_dir=cli_args.pdf_dir,
		output_dir=cli_args.output_dir,
		clear_output=cli_args.clear_output,
		min_edge=cli_args.min_edge,
		workers=cli_args.workers,
		logger=logger,
	)


if __name__ == "__main__":  # pragma: no cover
	main()

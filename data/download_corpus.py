"""
CampusIQ Corpus Downloader (Hardened)
Downloads ONLY verified public IIT Jodhpur documents explicitly listed in data/sources.csv.
Guarantees:
1. No guessed URLs.
2. Validates that HTTP response content-type contains 'pdf'.
3. Validates that binary payload begins with '%PDF' magic bytes.
4. If a 'sha256' column is present and populated in sources.csv, verifies the checksum
   and fails on mismatch instead of silently overwriting.
"""

import csv
import hashlib
import logging
import os
import sys
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.downloader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def verify_sha256(file_path: str, expected_hash: str) -> bool:
    h = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    actual_hash = h.hexdigest().lower()
    return actual_hash == expected_hash.lower().strip()


def download_corpus(sources_csv: str = SOURCES_CSV, raw_dir: str = RAW_DATA_DIR) -> int:
    os.makedirs(raw_dir, exist_ok=True)
    if not os.path.exists(sources_csv):
        raise FileNotFoundError(f"Sources file not found at: {sources_csv}")

    with open(sources_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    logger.info(f"Loaded {len(sources)} verified documents from {sources_csv}")
    success_count = 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    for idx, row in enumerate(sources, 1):
        doc_id = row["doc_id"].strip()
        file_name = row["file_name"].strip()
        url = row["source_url"].strip()
        expected_sha256 = row.get("sha256", "").strip()
        target_path = os.path.join(raw_dir, file_name)

        # Check existing file
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            # If SHA256 is configured, verify existing file
            if expected_sha256:
                if verify_sha256(target_path, expected_sha256):
                    logger.info(f"[{idx}/{len(sources)}] [{doc_id}] Exists and verified (SHA-256 match): {file_name}")
                    success_count += 1
                    continue
                else:
                    logger.error(
                        f"[{idx}/{len(sources)}] [{doc_id}] Local file '{file_name}' SHA-256 mismatch with sources.csv! Re-downloading."
                    )
            else:
                # Basic check: verify PDF header
                with open(target_path, "rb") as fp:
                    if fp.read(4) == b"%PDF":
                        logger.info(f"[{idx}/{len(sources)}] [{doc_id}] Already downloaded: {file_name} ({os.path.getsize(target_path)} bytes)")
                        success_count += 1
                        continue

        logger.info(f"[{idx}/{len(sources)}] [{doc_id}] Downloading: {url} -> {file_name}")
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=25, allow_redirects=True)
            if r.status_code != 200:
                raise ValueError(f"HTTP Error {r.status_code} while fetching {url}")

            content_type = r.headers.get("content-type", "").lower()
            if "pdf" not in content_type:
                raise ValueError(
                    f"Invalid content-type '{content_type}' for {url}. Expected 'application/pdf'."
                )

            payload = r.content
            if not payload.startswith(b"%PDF"):
                raise ValueError(
                    f"Invalid magic bytes for {url}. Payload does not start with '%PDF'."
                )

            # Check SHA-256 against sources.csv before saving
            if expected_sha256:
                downloaded_hash = hashlib.sha256(payload).hexdigest().lower()
                if downloaded_hash != expected_sha256.lower():
                    raise ValueError(
                        f"SHA-256 mismatch for {file_name}!\n"
                        f"  Expected: {expected_sha256}\n"
                        f"  Actual  : {downloaded_hash}"
                    )

            with open(target_path, "wb") as out_file:
                out_file.write(payload)

            size_kb = len(payload) / 1024
            logger.info(f"[{idx}/{len(sources)}] [{doc_id}] Successfully saved {file_name} ({size_kb:.1f} KB)")
            success_count += 1

        except Exception as e:
            logger.error(f"[{idx}/{len(sources)}] [{doc_id}] FAILED to download from {url}: {e}")

    logger.info(f"Download complete: {success_count}/{len(sources)} verified documents ready in {raw_dir}")
    return success_count


if __name__ == "__main__":
    download_corpus()

"""
CampusIQ Corpus Downloader
Reads data/sources.csv and downloads verified public IIT Jodhpur documents into data/raw/.
"""

import csv
import logging
import os
import sys
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.downloader")

SOURCES_CSV = os.path.join(os.path.dirname(__file__), "sources.csv")
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_corpus():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if not os.path.exists(SOURCES_CSV):
        logger.error(f"Sources file not found at {SOURCES_CSV}")
        sys.exit(1)

    with open(SOURCES_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    logger.info(f"Loaded {len(sources)} public documents to download.")
    success_count = 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    for row in sources:
        doc_id = row["doc_id"]
        file_name = row["file_name"]
        url = row["source_url"]
        target_path = os.path.join(RAW_DATA_DIR, file_name)

        if os.path.exists(target_path) and os.path.getsize(target_path) > 15000:
            logger.info(f"[{doc_id}] Already downloaded: {file_name} ({os.path.getsize(target_path)/1024:.1f} KB)")
            success_count += 1
            continue

        logger.info(f"[{doc_id}] Downloading: {url} -> {file_name}")
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=25)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(target_path, "wb") as out_file:
                    out_file.write(r.content)
                size_kb = len(r.content) / 1024
                logger.info(f"[{doc_id}] Successfully saved {file_name} ({size_kb:.1f} KB)")
                success_count += 1
            else:
                logger.warning(f"[{doc_id}] Download returned HTTP {r.status_code} or small size ({len(r.content)} bytes)")
        except Exception as e:
            logger.warning(f"[{doc_id}] Failed to download from {url}: {e}")

    logger.info(f"Download complete: {success_count}/{len(sources)} files ready in {RAW_DATA_DIR}")
    return success_count


if __name__ == "__main__":
    download_corpus()

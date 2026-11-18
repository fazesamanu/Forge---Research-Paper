"""Download official WFlib datasets from Zenodo (13731720 / 14195051) and inspect contents."""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm


ZENODO_RECORDS = ["13731720", "14195051"]

TABLE8_FILES = [
    "closed_2tab.npz.zip",
    "closed_3tab.npz.zip",
    "closed_4tab.npz.zip",
    "closed_5tab.npz.zip",
    "open_2tab.npz.zip",
    "open_3tab.npz.zip",
    "open_4tab.npz.zip",
    "open_5tab.npz.zip",
    "WTF-PAD.npz.zip",
    "Front.npz.zip",
    "Walkie-Talkie.npz.zip",
    "TrafficSliver.npz.zip",
]


def query_zenodo_record(record_id: str, timeout: int = 30) -> Dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def inspect_records(records: List[str] = None) -> Dict[str, List[Dict]]:
    if records is None:
        records = ZENODO_RECORDS
    catalog: Dict[str, List[Dict]] = {}
    for rec_id in records:
        try:
            data = query_zenodo_record(rec_id)
            title = data.get("metadata", {}).get("title", "Unknown")
            files = data.get("files", [])
            print(f"\n=== Zenodo Record {rec_id}: {title} ({len(files)} files) ===")
            entries = []
            for f in files:
                key = f.get("key", "")
                size_mb = f.get("size", 0) / (1024 * 1024)
                dl_url = f.get("links", {}).get("self", "")
                checksum = f.get("checksum", "")
                print(f"  - {key:40s}  {size_mb:9.2f} MB  {dl_url}")
                entries.append(
                    {
                        "key": key,
                        "size": f.get("size", 0),
                        "url": dl_url,
                        "checksum": checksum,
                        "record_id": rec_id,
                    }
                )
            catalog[rec_id] = entries
        except Exception as exc:
            print(f"[WARN] Failed to query Zenodo record {rec_id}: {exc}")
    return catalog


def download_file_with_resume(url: str, dest_path: Path, expected_size: int = 0) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if dest_path.exists() and (expected_size == 0 or dest_path.stat().st_size == expected_size):
        print(f"[SKIP] Already downloaded: {dest_path.name}")
        return

    existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {}
    mode = "wb"
    if existing_size > 0 and (expected_size == 0 or existing_size < expected_size):
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if resp.status_code == 416:
            if tmp_path.exists():
                tmp_path.replace(dest_path)
            return
        resp.raise_for_status()
        if resp.status_code == 200 and existing_size > 0:
            existing_size = 0
            mode = "wb"

        total = expected_size or (int(resp.headers.get("content-length", 0)) + existing_size)
        with open(tmp_path, mode) as f, tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
            mininterval=2.0,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp_path.replace(dest_path)


def extract_if_zip(archive_path: Path, out_dir: Path) -> None:
    if archive_path.suffix.lower() != ".zip":
        return
    print(f"[UNZIP] Extracting {archive_path.name} -> {out_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Inspect and download WFlib datasets from Zenodo.")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Target directory for datasets")
    parser.add_argument("--inspect", action="store_true", help="Only list available Zenodo files")
    parser.add_argument("--records", type=str, nargs="+", default=["13731720"], help="Zenodo record IDs")
    parser.add_argument(
        "--table8_only",
        action="store_true",
        help="Download only the 12 Table 8 datasets from record 13731720",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional substrings to filter which files to download",
    )
    parser.add_argument("--keep_zip", action="store_true", help="Keep .zip archives after extraction")
    args = parser.parse_args()

    catalog = inspect_records(args.records)
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "zenodo_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    if args.inspect:
        return

    all_entries = []
    for entries in catalog.values():
        all_entries.extend(entries)

    table8_lower = {name.lower() for name in TABLE8_FILES}

    for entry in all_entries:
        key = entry["key"]
        if args.table8_only and key.lower() not in table8_lower:
            continue
        if args.filter:
            if not any(flt.lower() in key.lower() for flt in args.filter):
                continue

        # Check if extracted .npz already exists
        expected_npz = out_dir / key[:-4] if key.lower().endswith(".zip") else out_dir / key
        if expected_npz.exists() and expected_npz.stat().st_size > 1_000_000:
            print(f"[SKIP] Already extracted: {expected_npz.name}")
            continue

        dest = out_dir / key
        download_file_with_resume(entry["url"], dest, expected_size=entry["size"])
        if dest.suffix.lower() == ".zip":
            extract_if_zip(dest, out_dir)
            if not args.keep_zip:
                dest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

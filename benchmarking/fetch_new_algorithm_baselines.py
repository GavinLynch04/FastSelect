"""Fetch the exact external reference tree used by benchmark_new_algorithms.

Reference-only GPL source is kept under ignored audit/new_algorithms/vendor;
it is not imported, copied or distributed by the FastSelect runtime package.
"""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "audit/new_algorithms/baseline_manifest.json"
manifest = json.loads(manifest_path.read_text())
destination = ROOT / "audit/new_algorithms/vendor/jundongl-scikit-feature-48cffad"
if destination.exists():
    print(f"Already present: {destination}")
else:
    url = f"https://api.github.com/repos/{manifest['repository']}/tarball/{manifest['commit']}"
    with urlopen(Request(url, headers={"User-Agent": "FastSelect-benchmark"}), timeout=60) as response:
        archive = response.read()
    actual = hashlib.sha256(archive).hexdigest()
    if actual != manifest["archive_sha256"]:
        raise RuntimeError(f"Archive hash mismatch: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        prefix = bundle.getmembers()[0].name.split("/")[0]
        bundle.extractall(destination.parent, filter="data")
    (destination.parent / prefix).rename(destination)
    print(f"Fetched pinned reference: {destination}")
manifest["source_directory"] = str(destination)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

"""Restore the archived pilot data, models and results; no training or test evaluation."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "pilot_artifacts.zip"
EXPECTED_SHA256 = "1e553b1de65c8763781142ee21523c9787f498f91ada606c196a133ec36a6130"

def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("The pilot archive checksum does not match this snapshot.")
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        for name, digest in manifest.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"artifacts", "data", "models", "report"}:
                raise ValueError(f"Unexpected archive path: {name}")
            target = ROOT / relative
            if not target.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Archive extraction would leave the project directory.")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"File checksum mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f"Restored {len(manifest)} verified pilot files. No training or test evaluation was performed.")

if __name__ == "__main__":
    main()

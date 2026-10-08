"""Restore split artifact files, checking all sizes and SHA-256 digests."""
from pathlib import Path
import hashlib
import json
import os
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Invalid path in artifact manifest")
    return path


def main():
    root = Path(__file__).resolve().parent
    entries = json.loads((root / "artifact_parts_manifest.json").read_text())
    for entry in entries:
        destination = safe_path(root, entry["path"])
        if destination.exists():
            if destination.stat().st_size == entry["size"] and digest(destination) == entry["sha256"]:
                print("Verified existing:", entry["path"])
                continue
            raise SystemExit("Refusing to overwrite a different file: " + entry["path"])
        parts = []
        for item in entry["parts"]:
            part = safe_path(root, item["path"])
            if (not part.is_file() or part.stat().st_size != item["size"]
                    or digest(part) != item["sha256"]):
                raise SystemExit("Missing or corrupted artifact part: " + item["path"])
            parts.append(part)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".restore-", dir=destination.parent)
        temporary = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as output:
                for part in parts:
                    with part.open("rb") as f:
                        for block in iter(lambda: f.read(1024 * 1024), b""):
                            output.write(block)
            if temporary.stat().st_size != entry["size"] or digest(temporary) != entry["sha256"]:
                raise SystemExit("Restored file checksum mismatch: " + entry["path"])
            # Exclusive creation preserves any file created concurrently.
            os.link(temporary, destination)
            print("Restored:", entry["path"])
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

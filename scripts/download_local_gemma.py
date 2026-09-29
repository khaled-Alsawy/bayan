from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORY = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / ".runtime/models/gemma-4-26b-a4b-it-4bit--0d77464e"


def main() -> None:
    print("Resuming the pinned Gemma snapshot; validated cached files will be reused.")
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=DESTINATION,
        max_workers=4,
    )
    print("Pinned Gemma snapshot download complete.")


if __name__ == "__main__":
    main()

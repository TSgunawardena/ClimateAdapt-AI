"""Prepare a shareable ClimateAdapt planning-model release directory.

This script copies the already-trained GGUF; it does not train, modify, or
requantise model weights. It writes a portable Ollama Modelfile and SHA-256
checksums. The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GGUF = (
    ROOT
    / "models"
    / "language"
    / "climateadapt_qwen3_4b"
    / "ClimateAdapt-Qwen3-4B-Q4_K_M.gguf"
)
SOURCE_MODELFILE = (
    ROOT / "models" / "language" / "climateadapt_qwen3_4b" / "Modelfile"
)
SOURCE_NOTEBOOK = ROOT / "notebooks" / "28_finetune_qwen3_4b_qlora_colab.ipynb"
SOURCE_MODEL_CARD = ROOT / "docs" / "model_cards" / "climateadapt_planner_4b.md"
SOURCE_REPRODUCTION = ROOT / "docs" / "LLM_REPRODUCIBILITY.md"
RELEASE_GGUF_NAME = "ClimateAdapt-Qwen3-4B-Q4_K_M.gguf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_modelfile(source: Path) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.lstrip().startswith("FROM ") and not replaced:
            output.append(f"FROM ./{RELEASE_GGUF_NAME}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        raise RuntimeError("The source Modelfile has no FROM instruction.")
    return "\n".join(output) + "\n"


def installation_readme() -> str:
    return """# ClimateAdapt Planner 4B — Model Release

This directory contains the fine-tuned ClimateAdapt planning-language model.
It is a research prototype and must not be used as an autonomous planning,
regulatory, engineering, emergency-warning, or public-safety system.

## Contents

- `ClimateAdapt-Qwen3-4B-Q4_K_M.gguf` — merged Q4_K_M model
- `Modelfile` — portable Ollama configuration
- `MODEL_CARD.md` — intended use, evaluation and limitations
- `LLM_REPRODUCIBILITY.md` — training and evaluation procedure
- `28_finetune_qwen3_4b_qlora_colab.ipynb` — authentic training code
- `SHA256SUMS.txt` — integrity hashes

## Verify

In PowerShell, calculate the GGUF checksum:

```powershell
Get-FileHash .\\ClimateAdapt-Qwen3-4B-Q4_K_M.gguf -Algorithm SHA256
```

Compare the result with `SHA256SUMS.txt`.

## Install with Ollama

From this directory:

```powershell
ollama create climateadapt-planner:4b -f .\\Modelfile
ollama list
```

The model produces constrained planning language from verified ClimateAdapt
evidence. Human review remains mandatory.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory, preferably inside the user's OneDrive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    required = [
        SOURCE_GGUF,
        SOURCE_MODELFILE,
        SOURCE_NOTEBOOK,
        SOURCE_MODEL_CARD,
        SOURCE_REPRODUCTION,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}\nChoose a new empty path."
        )

    output.mkdir(parents=True)
    destinations = {
        SOURCE_GGUF: output / RELEASE_GGUF_NAME,
        SOURCE_NOTEBOOK: output / SOURCE_NOTEBOOK.name,
        SOURCE_MODEL_CARD: output / "MODEL_CARD.md",
        SOURCE_REPRODUCTION: output / "LLM_REPRODUCIBILITY.md",
    }
    for source, destination in destinations.items():
        shutil.copy2(source, destination)

    (output / "Modelfile").write_text(
        portable_modelfile(SOURCE_MODELFILE), encoding="utf-8"
    )
    (output / "README_MODEL_INSTALLATION.md").write_text(
        installation_readme(), encoding="utf-8"
    )

    release_files = sorted(path for path in output.iterdir() if path.is_file())
    checksums = [f"{sha256(path)}  {path.name}" for path in release_files]
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    print("=" * 78)
    print("CLIMATEADAPT PLANNER 4B — MODEL RELEASE PREPARED")
    print("=" * 78)
    print(f"Output: {output}")
    print(f"Files:  {len(list(output.iterdir()))}")
    print("GGUF was copied without modification.")
    print("Verify SHA256SUMS.txt before sharing the folder.")


if __name__ == "__main__":
    main()

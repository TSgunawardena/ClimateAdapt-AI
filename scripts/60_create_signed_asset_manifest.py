"""Create an Ed25519-signed SHA-256 manifest for V3 serving assets."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "artifact_signing_policy.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON that may contain a Windows UTF-8 BOM."""

    with path.open("r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def project_path(relative_path: str) -> Path:
    """Resolve and constrain a path to the project directory."""

    candidate = (
        PROJECT_ROOT / Path(relative_path)
    ).resolve()

    project_root_resolved = PROJECT_ROOT.resolve()

    if not candidate.is_relative_to(project_root_resolved):
        raise ValueError(
            f"Path escapes the project root: {relative_path}"
        )

    return candidate


def request_signing_password() -> bytes:
    """Request and confirm an unseen signing-key password."""

    first_password = getpass(
        "Create a password for the Ed25519 signing key: "
    )
    second_password = getpass(
        "Confirm the Ed25519 signing-key password: "
    )

    if first_password != second_password:
        raise ValueError(
            "The signing-key passwords do not match."
        )

    if len(first_password) < 12:
        raise ValueError(
            "The signing-key password must contain at least "
            "12 characters."
        )

    return first_password.encode("utf-8")


def ensure_outputs_do_not_exist(paths: list[Path]) -> None:
    """Refuse to replace an existing signing identity or manifest."""

    existing = [path for path in paths if path.exists()]

    if existing:
        existing_text = "\n".join(
            f"  - {path.relative_to(PROJECT_ROOT)}"
            for path in existing
        )
        raise FileExistsError(
            "Refusing to overwrite existing signing material:\n"
            f"{existing_text}"
        )


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 digest."""

    initial_stat = path.stat()
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    final_stat = path.stat()

    if (
        initial_stat.st_size != final_stat.st_size
        or initial_stat.st_mtime_ns
        != final_stat.st_mtime_ns
    ):
        raise RuntimeError(
            f"Asset changed while being hashed: {path}"
        )

    return digest.hexdigest()


def canonical_json(data: dict[str, Any]) -> bytes:
    """Return the exact bytes covered by the Ed25519 signature."""

    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def main() -> None:
    policy = load_json(POLICY_PATH)

    if policy["digest_algorithm"] != "SHA-256":
        raise ValueError("The required digest is SHA-256.")

    if policy["signature_algorithm"] != "Ed25519":
        raise ValueError("The required signature is Ed25519.")

    private_key_path = project_path(
        policy["signing_key"]["private_key_path"]
    )
    public_key_path = project_path(
        policy["signing_key"]["public_key_path"]
    )
    manifest_path = project_path(
        policy["manifest"]["path"]
    )
    signature_path = project_path(
        policy["manifest"]["signature_path"]
    )
    evidence_path = project_path(
        policy["outputs"]["generation_evidence"]
    )

    ensure_outputs_do_not_exist(
        [
            private_key_path,
            public_key_path,
            manifest_path,
            signature_path,
            evidence_path,
        ]
    )

    asset_definitions = policy["protected_assets"]

    if not asset_definitions:
        raise ValueError(
            "The policy contains no protected assets."
        )

    print("=" * 76)
    print("CLIMATEADAPT – ED25519-SIGNED V3 ASSET MANIFEST")
    print("=" * 76)
    print(f"Protected assets: {len(asset_definitions)}")
    print("Digest:           SHA-256")
    print("Signature:        Ed25519")
    print("Private key:      encrypted PKCS#8")
    print("")

    password = request_signing_password()

    asset_records: list[dict[str, Any]] = []
    total_bytes = 0

    for index, asset_definition in enumerate(
        asset_definitions,
        start=1,
    ):
        relative_path = asset_definition["path"]
        asset_path = project_path(relative_path)

        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Protected asset is missing: {relative_path}"
            )

        size_bytes = asset_path.stat().st_size

        print(
            f"[{index:02d}/{len(asset_definitions):02d}] "
            f"Hashing {relative_path}"
        )

        digest = sha256_file(asset_path)

        asset_records.append(
            {
                "path": relative_path,
                "role": asset_definition["role"],
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )

        total_bytes += size_bytes

    signing_private_key = Ed25519PrivateKey.generate()
    signing_public_key = signing_private_key.public_key()

    public_key_der = signing_public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    public_key_fingerprint = hashlib.sha256(
        public_key_der
    ).hexdigest()

    manifest = {
        "manifest_version": 1,
        "manifest_name": (
            "ClimateAdapt V3 deployed serving assets"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "digest_algorithm": "SHA-256",
        "signature_algorithm": "Ed25519",
        "public_key_sha256": public_key_fingerprint,
        "asset_count": len(asset_records),
        "total_bytes": total_bytes,
        "assets": asset_records,
        "scope_note": policy["scope_note"],
        "verification_policy": (
            policy["verification_policy"]
        ),
    }

    canonical_manifest = canonical_json(manifest)
    signature = signing_private_key.sign(
        canonical_manifest
    )

    # Immediately verify the signature before writing evidence.
    signing_public_key.verify(
        signature,
        canonical_manifest,
    )

    private_key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    public_key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encrypted_private_key = (
        signing_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=(
                serialization.BestAvailableEncryption(
                    password
                )
            ),
        )
    )

    public_key_pem = signing_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.write_bytes(encrypted_private_key)
    public_key_path.write_bytes(public_key_pem)

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    signature_path.write_text(
        base64.b64encode(signature).decode("ascii")
        + "\n",
        encoding="ascii",
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 Ed25519 asset signing v1"
        ),
        "created_at_utc": manifest["created_at_utc"],
        "digest_algorithm": "SHA-256",
        "signature_algorithm": "Ed25519",
        "private_key_format": "Encrypted PKCS#8",
        "public_key_sha256": public_key_fingerprint,
        "asset_count": len(asset_records),
        "total_bytes": total_bytes,
        "manifest_sha256": hashlib.sha256(
            canonical_manifest
        ).hexdigest(),
        "signature_length_bytes": len(signature),
        "signature_encoding": "Base64",
        "signature_self_verified": True,
        "all_assets_present": True,
        "password_recorded": False,
        "private_key_recorded": False,
        "private_key_committed_to_git": False,
        "secret_values_in_evidence": 0,
        "outputs": {
            "public_key": str(
                public_key_path.relative_to(PROJECT_ROOT)
            ),
            "manifest": str(
                manifest_path.relative_to(PROJECT_ROOT)
            ),
            "signature": str(
                signature_path.relative_to(PROJECT_ROOT)
            ),
            "private_key": (
                "Stored locally under security/private/ and "
                "excluded from Git"
            ),
        },
    }

    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print("")
    print("-" * 76)
    print("SIGNED ASSET MANIFEST CREATED")
    print("-" * 76)
    print(f"Assets hashed:       {evidence['asset_count']}")
    print(
        "Total protected MB: "
        f"{total_bytes / (1024 * 1024):.2f}"
    )
    print("Digest algorithm:    SHA-256")
    print("Signature algorithm: Ed25519")
    print(
        "Signature verified:  "
        f"{evidence['signature_self_verified']}"
    )
    print("Private key encrypted: True")
    print(
        "Secrets recorded:    "
        f"{evidence['secret_values_in_evidence']}"
    )
    print(
        "Public-key SHA-256:   "
        f"{public_key_fingerprint}"
    )
    print("-" * 76)
    print(
        "Manifest:  "
        f"{manifest_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Signature: "
        f"{signature_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Public key:"
        f" {public_key_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Evidence:  "
        f"{evidence_path.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
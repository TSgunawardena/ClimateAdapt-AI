"""Rotate the ClimateAdapt Ed25519 asset-signing key from V1 to V2."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROTATION_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "artifact_signing_key_rotation_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def public_key_fingerprint(
    public_key: Ed25519PublicKey,
) -> str:
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(public_der).hexdigest()


def request_new_password() -> bytes:
    first_password = getpass(
        "Create a new password for the V2 signing key: "
    )
    second_password = getpass(
        "Confirm the V2 signing-key password: "
    )

    if first_password != second_password:
        raise ValueError(
            "The V2 signing-key passwords do not match."
        )

    if len(first_password) < 12:
        raise ValueError(
            "The V2 signing-key password must contain "
            "at least 12 characters."
        )

    return first_password.encode("utf-8")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )


def refuse_existing_outputs(paths: list[Path]) -> None:
    existing_paths = [
        path for path in paths if path.exists()
    ]

    if existing_paths:
        formatted = "\n".join(
            str(path.relative_to(PROJECT_ROOT))
            for path in existing_paths
        )
        raise FileExistsError(
            "Refusing to overwrite existing V2 rotation "
            f"material:\n{formatted}"
        )


def main() -> None:
    require_file(
        ROTATION_POLICY_PATH,
        "Signing-key rotation policy",
    )

    rotation_policy = load_json(
        ROTATION_POLICY_PATH
    )
    source_policy_path = project_path(
        rotation_policy["source_asset_policy"]
    )
    require_file(
        source_policy_path,
        "Source asset-signing policy",
    )

    source_policy = load_json(source_policy_path)
    previous_config = rotation_policy["previous_key"]
    replacement_config = rotation_policy[
        "replacement_key"
    ]

    previous_public_path = project_path(
        previous_config["public_key_path"]
    )
    previous_private_path = project_path(
        previous_config["private_key_path"]
    )
    replacement_private_path = project_path(
        replacement_config["private_key_path"]
    )
    replacement_public_path = project_path(
        replacement_config["public_key_path"]
    )
    replacement_manifest_path = project_path(
        replacement_config["manifest_path"]
    )
    replacement_signature_path = project_path(
        replacement_config["signature_path"]
    )
    rotation_evidence_path = project_path(
        rotation_policy["rotation_evidence"]
    )

    require_file(
        previous_public_path,
        "Previous V1 public signing key",
    )

    # We confirm the old private key exists, but deliberately
    # never read its content during rotation.
    require_file(
        previous_private_path,
        "Previous encrypted V1 private signing key",
    )

    refuse_existing_outputs(
        [
            replacement_private_path,
            replacement_public_path,
            replacement_manifest_path,
            replacement_signature_path,
            rotation_evidence_path,
        ]
    )

    previous_public_key = (
        serialization.load_pem_public_key(
            previous_public_path.read_bytes()
        )
    )

    if not isinstance(
        previous_public_key,
        Ed25519PublicKey,
    ):
        raise TypeError(
            "The previous public key is not Ed25519."
        )

    previous_fingerprint = public_key_fingerprint(
        previous_public_key
    )

    print("=" * 78)
    print("CLIMATEADAPT V3 – ED25519 SIGNING-KEY ROTATION V1 TO V2")
    print("=" * 78)
    print("Previous key:       V1 – retiring from active signing")
    print("Replacement key:    V2 – generated pending activation")
    print("Previous private key accessed: False")
    print("Protected assets:   recalculated using SHA-256")
    print("Passwords recorded: False")
    print("")

    password = request_new_password()

    asset_records: list[dict[str, Any]] = []

    for asset in source_policy["protected_assets"]:
        asset_path = project_path(asset["path"])
        require_file(
            asset_path,
            f"Protected asset {asset['path']}",
        )

        asset_records.append(
            {
                "path": asset["path"],
                "role": asset["role"],
                "size_bytes": asset_path.stat().st_size,
                "sha256": sha256_file(asset_path),
            }
        )

    replacement_private_key = (
        Ed25519PrivateKey.generate()
    )
    replacement_public_key = (
        replacement_private_key.public_key()
    )
    replacement_fingerprint = (
        public_key_fingerprint(
            replacement_public_key
        )
    )

    if replacement_fingerprint == previous_fingerprint:
        raise RuntimeError(
            "The V2 key unexpectedly matches the V1 key."
        )

    rotated_at_utc = datetime.now(
        timezone.utc
    ).isoformat()

    manifest = {
        "manifest_version": 2,
        "manifest_name": (
            "ClimateAdapt V3 serving-asset manifest V2"
        ),
        "created_at_utc": rotated_at_utc,
        "digest_algorithm": "SHA-256",
        "signature_algorithm": "Ed25519",
        "signing_key_version": "v2",
        "previous_signing_key_version": "v1",
        "public_key_sha256": replacement_fingerprint,
        "previous_public_key_sha256": (
            previous_fingerprint
        ),
        "rotation_reason": rotation_policy[
            "rotation_reason"
        ],
        "assets": asset_records,
    }

    canonical_manifest = canonical_json(manifest)
    signature = replacement_private_key.sign(
        canonical_manifest
    )

    replacement_public_key.verify(
        signature,
        canonical_manifest,
    )

    previous_key_rejected = False

    try:
        previous_public_key.verify(
            signature,
            canonical_manifest,
        )
    except InvalidSignature:
        previous_key_rejected = True

    if not previous_key_rejected:
        raise RuntimeError(
            "The V1 public key unexpectedly verified "
            "the V2 signature."
        )

    encrypted_private_key = (
        replacement_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=(
                serialization.BestAvailableEncryption(
                    password
                )
            ),
        )
    )

    public_key_pem = (
        replacement_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=(
                serialization.PublicFormat.SubjectPublicKeyInfo
            ),
        )
    )

    replacement_private_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    replacement_public_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    replacement_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rotation_evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacement_private_path.write_bytes(
        encrypted_private_key
    )
    replacement_public_path.write_bytes(
        public_key_pem
    )
    replacement_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    replacement_signature_path.write_text(
        base64.b64encode(signature).decode("ascii")
        + "\n",
        encoding="ascii",
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 Ed25519 "
            "signing-key rotation V1 to V2"
        ),
        "rotated_at_utc": rotated_at_utc,
        "status": "rotation_generated_pending_activation",
        "algorithm": "Ed25519",
        "digest_algorithm": "SHA-256",
        "previous_key_version": "v1",
        "previous_key_final_status": (
            "retired_from_active_signing"
        ),
        "replacement_key_version": "v2",
        "replacement_key_status": (
            "generated_pending_activation"
        ),
        "previous_public_key_sha256": (
            previous_fingerprint
        ),
        "replacement_public_key_sha256": (
            replacement_fingerprint
        ),
        "replacement_key_differs_from_previous": True,
        "previous_private_key_accessed": False,
        "previous_public_key_retained": True,
        "replacement_private_key_format": (
            "Encrypted PKCS#8"
        ),
        "replacement_private_key_committed_to_git": False,
        "password_recorded": False,
        "private_key_recorded": False,
        "protected_assets_recalculated": len(
            asset_records
        ),
        "replacement_signature_self_verified": True,
        "previous_key_rejected_replacement_signature": (
            previous_key_rejected
        ),
        "manifest": str(
            replacement_manifest_path.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "signature": str(
            replacement_signature_path.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "public_key": str(
            replacement_public_path.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "private_key_location": (
            "Local ignored security/private/signing directory"
        ),
        "secret_values_in_evidence": 0,
    }

    rotation_evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("SIGNING-KEY ROTATION MATERIAL GENERATED")
    print("=" * 78)
    print(f"Protected assets:             {len(asset_records)}")
    print("V2 signature verified:        True")
    print("V1 rejected V2 signature:     True")
    print("V1 private key accessed:      False")
    print("V2 private key encrypted:     True")
    print("Passwords recorded:           False")
    print("Secrets recorded:             0")
    print("-" * 78)
    print(f"V1 fingerprint: {previous_fingerprint}")
    print(f"V2 fingerprint: {replacement_fingerprint}")
    print("-" * 78)
    print(
        "Rotation evidence: "
        + str(
            rotation_evidence_path.relative_to(
                PROJECT_ROOT
            )
        )
    )
    print("=" * 78)
    print(
        "V2 is generated but is not active until "
        "the verifier configuration is updated."
    )


if __name__ == "__main__":
    main()
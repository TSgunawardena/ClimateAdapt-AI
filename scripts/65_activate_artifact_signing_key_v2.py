"""Validate and activate the rotated V2 asset-signing key."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROTATION_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "artifact_signing_key_rotation_v2.json"
)

ACTIVE_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "artifact_signing_policy.json"
)

VERIFIER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "61_verify_signed_asset_manifest.py"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )


def load_ed25519_public_key(path: Path) -> Ed25519PublicKey:
    public_key = serialization.load_pem_public_key(
        path.read_bytes()
    )

    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError(
            f"The key is not an Ed25519 public key: {path}"
        )

    return public_key


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_text_atomic(
    path: Path,
    value: str,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        value,
        encoding="utf-8",
    )
    temporary_path.replace(path)


def replace_once(
    content: str,
    old_value: str,
    new_value: str,
) -> str:
    occurrence_count = content.count(old_value)

    if occurrence_count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of "
            f"{old_value!r}, found {occurrence_count}."
        )

    return content.replace(
        old_value,
        new_value,
        1,
    )


def main() -> None:
    for path, description in [
        (
            ROTATION_POLICY_PATH,
            "Rotation policy",
        ),
        (
            ACTIVE_POLICY_PATH,
            "Active signing policy",
        ),
        (
            VERIFIER_PATH,
            "Independent verifier",
        ),
    ]:
        require_file(path, description)

    rotation_policy = load_json(
        ROTATION_POLICY_PATH
    )
    active_policy = load_json(
        ACTIVE_POLICY_PATH
    )

    previous = rotation_policy["previous_key"]
    replacement = rotation_policy["replacement_key"]

    v1_public_path = project_path(
        previous["public_key_path"]
    )
    v2_public_path = project_path(
        replacement["public_key_path"]
    )
    v2_manifest_path = project_path(
        replacement["manifest_path"]
    )
    v2_signature_path = project_path(
        replacement["signature_path"]
    )
    rotation_evidence_path = project_path(
        rotation_policy["rotation_evidence"]
    )

    for path, description in [
        (v1_public_path, "V1 public key"),
        (v2_public_path, "V2 public key"),
        (v2_manifest_path, "V2 manifest"),
        (v2_signature_path, "V2 signature"),
        (
            rotation_evidence_path,
            "Rotation-generation evidence",
        ),
    ]:
        require_file(path, description)

    v1_public_key = load_ed25519_public_key(
        v1_public_path
    )
    v2_public_key = load_ed25519_public_key(
        v2_public_path
    )

    manifest = load_json(v2_manifest_path)
    canonical_manifest = canonical_json(manifest)

    signature = base64.b64decode(
        v2_signature_path.read_text(
            encoding="ascii"
        ).strip(),
        validate=True,
    )

    # Confirm V2 authorises its own manifest.
    v2_public_key.verify(
        signature,
        canonical_manifest,
    )

    # Confirm the retired V1 key cannot authorise V2.
    v1_rejected_v2 = False

    try:
        v1_public_key.verify(
            signature,
            canonical_manifest,
        )
    except InvalidSignature:
        v1_rejected_v2 = True

    if not v1_rejected_v2:
        raise RuntimeError(
            "V1 unexpectedly verified the V2 signature."
        )

    missing_assets: list[str] = []
    mismatched_assets: list[str] = []
    verified_assets = 0

    for asset in manifest["assets"]:
        asset_path = project_path(asset["path"])

        if not asset_path.is_file():
            missing_assets.append(asset["path"])
            continue

        current_digest = sha256_file(asset_path)
        current_size = asset_path.stat().st_size

        if (
            current_digest != asset["sha256"]
            or current_size != asset["size_bytes"]
        ):
            mismatched_assets.append(asset["path"])
            continue

        verified_assets += 1

    if missing_assets or mismatched_assets:
        raise RuntimeError(
            "V2 activation refused because protected "
            "assets did not match the signed manifest."
        )

    if verified_assets != len(manifest["assets"]):
        raise RuntimeError(
            "Not every V2 protected asset was verified."
        )

    # Prepare the active policy update.
    active_policy["policy_version"] = 2

    active_policy["signing_key"][
        "private_key_path"
    ] = replacement["private_key_path"]

    active_policy["signing_key"][
        "public_key_path"
    ] = replacement["public_key_path"]

    active_policy["manifest"]["path"] = (
        replacement["manifest_path"]
    )
    active_policy["manifest"]["signature_path"] = (
        replacement["signature_path"]
    )

    active_policy["verification_policy"][
        "active_signing_key_version"
    ] = "v2"

    active_policy["key_lifecycle"] = {
        "current_key_version": "v2",
        "current_key_status": "active",
        "previous_key_version": "v1",
        "previous_key_status": (
            "retired_from_active_signing"
        ),
        "previous_public_key_retained": True,
        "previous_private_key_accessed_during_rotation": False,
        "rotation_policy": str(
            ROTATION_POLICY_PATH.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
    }

    active_policy["outputs"][
        "verification_evidence"
    ] = (
        "outputs/security_assessment_v1/"
        "artifact_manifest_verification_evidence_v1.json"
    )

    # Change the independent verifier from V1 paths to V2 paths.
    verifier_text = VERIFIER_PATH.read_text(
        encoding="utf-8"
    )

    verifier_text = replace_once(
        verifier_text,
        "climateadapt_v3_asset_manifest_v1.json",
        "climateadapt_v3_asset_manifest_v2.json",
    )
    verifier_text = replace_once(
        verifier_text,
        "climateadapt_v3_asset_manifest_v1.sig",
        "climateadapt_v3_asset_manifest_v2.sig",
    )
    verifier_text = replace_once(
        verifier_text,
        "climateadapt_artifact_signing_v1_public.pem",
        "climateadapt_artifact_signing_v2_public.pem",
    )

    activation_time = datetime.now(
        timezone.utc
    ).isoformat()

    rotation_evidence = load_json(
        rotation_evidence_path
    )
    rotation_evidence.update(
        {
            "status": "active",
            "activated_at_utc": activation_time,
            "previous_key_final_status": (
                "retired_from_active_signing"
            ),
            "replacement_key_status": "active",
            "active_manifest_version": 2,
            "active_verifier_updated": True,
            "v2_signature_verified_before_activation": True,
            "v1_rejected_v2_signature_before_activation": True,
            "protected_assets_verified_before_activation": (
                verified_assets
            ),
            "missing_assets_before_activation": 0,
            "mismatched_assets_before_activation": 0,
            "previous_private_key_accessed": False,
            "secret_values_in_evidence": 0,
        }
    )

    # All validation passed. Apply the prepared updates.
    write_json_atomic(
        ACTIVE_POLICY_PATH,
        active_policy,
    )
    write_text_atomic(
        VERIFIER_PATH,
        verifier_text,
    )
    write_json_atomic(
        rotation_evidence_path,
        rotation_evidence,
    )

    print("=" * 78)
    print("CLIMATEADAPT V3 – SIGNING KEY V2 ACTIVATED")
    print("=" * 78)
    print("Active signing key:          V2")
    print("Previous signing key:        V1 – retired")
    print(f"Protected assets verified:   {verified_assets}")
    print("V2 signature verified:       True")
    print("V1 rejected V2 signature:    True")
    print("V1 public key retained:      True")
    print("V1 private key accessed:     False")
    print("Verifier updated to V2:      True")
    print("Missing assets:              0")
    print("Mismatched assets:           0")
    print("Secrets recorded:            0")
    print("-" * 78)
    print(
        "Next: run Script 61 for independent "
        "post-activation verification."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
"""Safely demonstrate detection of a modified protected asset."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "security/manifests/climateadapt_v3_asset_manifest_v1.json"
)

SIGNATURE_PATH = (
    PROJECT_ROOT
    / "security/manifests/climateadapt_v3_asset_manifest_v1.sig"
)

PUBLIC_KEY_PATH = (
    PROJECT_ROOT
    / "security/public/signing/"
    / "climateadapt_artifact_signing_v1_public.pem"
)

TARGET_RELATIVE_PATH = (
    "outputs/test_v3/v3_locked_test_summary.json"
)

TEMP_DIRECTORY = (
    PROJECT_ROOT
    / "security/runtime/artifact_tamper_test"
)

TEMP_ASSET_PATH = (
    TEMP_DIRECTORY
    / "v3_locked_test_summary_tampered.json"
)

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs/security_assessment_v1/"
    / "artifact_tamper_detection_evidence_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def main() -> None:
    print("=" * 78)
    print("CLIMATEADAPT V3 – CONTROLLED ASSET-TAMPER DETECTION TEST")
    print("=" * 78)
    print("Real serving assets will not be modified.")
    print("Test method: modify a temporary copy and compare its SHA-256 digest.")
    print("")

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    signature = base64.b64decode(
        SIGNATURE_PATH.read_text(
            encoding="utf-8"
        ).strip(),
        validate=True,
    )

    public_key = serialization.load_pem_public_key(
        PUBLIC_KEY_PATH.read_bytes()
    )

    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Expected an Ed25519 public key.")

    public_key.verify(
        signature,
        canonical_json_bytes(manifest),
    )

    asset_entry = next(
        (
            asset
            for asset in manifest["assets"]
            if asset["path"] == TARGET_RELATIVE_PATH
        ),
        None,
    )

    if asset_entry is None:
        raise RuntimeError(
            "The selected tamper-test asset is not in the signed manifest."
        )

    original_path = (
        PROJECT_ROOT / TARGET_RELATIVE_PATH
    ).resolve()

    if not original_path.is_file():
        raise FileNotFoundError(original_path)

    expected_hash = str(
        asset_entry["sha256"]
    ).lower()

    original_hash_before = sha256_file(
        original_path
    )

    if original_hash_before != expected_hash:
        raise RuntimeError(
            "Original protected asset was already inconsistent."
        )

    TEMP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_copy_removed = False

    try:
        shutil.copy2(
            original_path,
            TEMP_ASSET_PATH,
        )

        with TEMP_ASSET_PATH.open("ab") as file_handle:
            file_handle.write(
                b"\nCONTROLLED-TAMPER-TEST\n"
            )

        tampered_hash = sha256_file(
            TEMP_ASSET_PATH
        )

        tamper_detected = (
            tampered_hash != expected_hash
        )
    finally:
        if TEMP_ASSET_PATH.exists():
            TEMP_ASSET_PATH.unlink()

        temporary_copy_removed = (
            not TEMP_ASSET_PATH.exists()
        )

    original_hash_after = sha256_file(
        original_path
    )

    original_unchanged = (
        original_hash_before
        == original_hash_after
        == expected_hash
    )

    passed = (
        tamper_detected
        and original_unchanged
        and temporary_copy_removed
    )

    evidence = {
        "experiment": (
            "ClimateAdapt V3 controlled "
            "asset-tamper detection v1"
        ),
        "tested_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "passed"
            if passed
            else "failed"
        ),
        "signature_verified_before_test": True,
        "tested_asset": TARGET_RELATIVE_PATH,
        "test_used_temporary_copy": True,
        "expected_sha256": expected_hash,
        "tampered_copy_sha256": tampered_hash,
        "tamper_detected": tamper_detected,
        "original_asset_unchanged": original_unchanged,
        "temporary_copy_removed": temporary_copy_removed,
        "private_key_accessed": False,
        "secret_values_in_evidence": 0,
    }

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVIDENCE_PATH.write_text(
        json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Signed manifest verified:       True")
    print(f"Protected test asset:           {TARGET_RELATIVE_PATH}")
    print(f"Tampering detected:             {tamper_detected}")
    print(f"Original asset unchanged:       {original_unchanged}")
    print(f"Temporary test copy removed:    {temporary_copy_removed}")
    print("Private key accessed:           False")
    print("Secrets recorded:               0")
    print("-" * 78)

    if not passed:
        raise RuntimeError(
            "Controlled tamper-detection test failed."
        )

    print("RESULT: PASS – modified asset content was detected safely.")
    print(
        "Evidence:",
        EVIDENCE_PATH.relative_to(PROJECT_ROOT),
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
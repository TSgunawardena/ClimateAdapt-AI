"""Independently verify the signed ClimateAdapt V3 asset manifest."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "security"
    / "manifests"
    / "climateadapt_v3_asset_manifest_v1.json"
)

SIGNATURE_PATH = (
    PROJECT_ROOT
    / "security"
    / "manifests"
    / "climateadapt_v3_asset_manifest_v1.sig"
)

PUBLIC_KEY_PATH = (
    PROJECT_ROOT
    / "security"
    / "public"
    / "signing"
    / "climateadapt_artifact_signing_v1_public.pem"
)

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "artifact_manifest_verification_evidence_v1.json"
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


def safe_asset_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()

    if not candidate.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            f"Asset path escapes the project directory: {relative_path}"
        )

    return candidate


def write_evidence(evidence: dict) -> None:
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 78)
    print("CLIMATEADAPT V3 – INDEPENDENT SIGNED-ASSET VERIFICATION")
    print("=" * 78)
    print("Private signing key required: False")
    print("Verification key: public Ed25519 key")
    print("Asset digest algorithm: SHA-256")
    print("")

    required_files = [
        MANIFEST_PATH,
        SIGNATURE_PATH,
        PUBLIC_KEY_PATH,
    ]

    missing_control_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_files
        if not path.is_file()
    ]

    if missing_control_files:
        raise FileNotFoundError(
            "Missing verification file(s): "
            + ", ".join(missing_control_files)
        )

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
        raise TypeError(
            "The verification key is not an Ed25519 public key."
        )

    signature_verified = False

    try:
        public_key.verify(
            signature,
            canonical_json_bytes(manifest),
        )
        signature_verified = True
    except InvalidSignature:
        signature_verified = False

    asset_results: list[dict] = []

    if signature_verified:
        for asset in manifest.get("assets", []):
            relative_path = str(asset["path"])
            asset_path = safe_asset_path(relative_path)
            exists = asset_path.is_file()

            actual_hash = (
                sha256_file(asset_path)
                if exists
                else None
            )

            expected_hash = str(
                asset["sha256"]
            ).lower()

            hash_matches = (
                exists
                and actual_hash == expected_hash
            )

            expected_size = asset.get("size_bytes")
            actual_size = (
                asset_path.stat().st_size
                if exists
                else None
            )

            size_matches = (
                expected_size is None
                or actual_size == int(expected_size)
            )

            asset_results.append(
                {
                    "path": relative_path,
                    "exists": exists,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "hash_matches": hash_matches,
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                    "size_matches": size_matches,
                    "verified": (
                        hash_matches and size_matches
                    ),
                }
            )

    assets_declared = len(
        manifest.get("assets", [])
    )

    assets_verified = sum(
        1
        for result in asset_results
        if result["verified"]
    )

    missing_assets = [
        result["path"]
        for result in asset_results
        if not result["exists"]
    ]

    mismatched_assets = [
        result["path"]
        for result in asset_results
        if (
            result["exists"]
            and not result["verified"]
        )
    ]

    overall_valid = (
        signature_verified
        and assets_declared > 0
        and assets_verified == assets_declared
        and not missing_assets
        and not mismatched_assets
    )

    evidence = {
        "experiment": (
            "ClimateAdapt V3 independent "
            "signed-asset verification v1"
        ),
        "verified_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "passed"
            if overall_valid
            else "failed_closed"
        ),
        "manifest": str(
            MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "signature": str(
            SIGNATURE_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "public_key": str(
            PUBLIC_KEY_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "digest_algorithm": "SHA-256",
        "signature_algorithm": "Ed25519",
        "signature_verified": signature_verified,
        "assets_declared": assets_declared,
        "assets_verified": assets_verified,
        "missing_assets": missing_assets,
        "mismatched_assets": mismatched_assets,
        "private_key_accessed": False,
        "secret_values_in_evidence": 0,
        "asset_results": asset_results,
    }

    write_evidence(evidence)

    print(f"Manifest signature valid: {signature_verified}")
    print(f"Assets declared:         {assets_declared}")
    print(f"Assets verified:         {assets_verified}")
    print(f"Missing assets:          {len(missing_assets)}")
    print(f"Mismatched assets:       {len(mismatched_assets)}")
    print("Private key accessed:    False")
    print("Secrets recorded:        0")
    print("-" * 78)

    if overall_valid:
        print("RESULT: PASS – signature and every protected asset verified.")
    else:
        print("RESULT: FAIL CLOSED – protected assets must not be served.")

    print(
        "Evidence:",
        EVIDENCE_PATH.relative_to(PROJECT_ROOT),
    )
    print("=" * 78)

    if not overall_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
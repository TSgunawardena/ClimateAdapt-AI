"""Prove that modified signed assets prevent secure API startup."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_ASSET = (
    PROJECT_ROOT
    / "outputs"
    / "test_v3"
    / "v3_locked_test_summary.json"
)

SECURE_SERVER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "57_run_prediction_api_mtls.py"
)

VERIFIER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "61_verify_signed_asset_manifest.py"
)

VERIFICATION_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "artifact_manifest_verification_evidence_v1.json"
)

TEST_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "integrated_fail_closed_startup_test_v1.json"
)

HOST = "127.0.0.1"
PORT = 9443


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def port_is_open() -> bool:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as client:
        client.settimeout(0.5)
        return client.connect_ex((HOST, PORT)) == 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(script: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    return subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env=environment,
    )


def main() -> None:
    print("=" * 78)
    print("CLIMATEADAPT V3 – INTEGRATED FAIL-CLOSED STARTUP TEST")
    print("=" * 78)
    print(f"Target asset: {TARGET_ASSET.relative_to(PROJECT_ROOT)}")
    print("Modification: controlled temporary append")
    print("Private signing key required: False")
    print("Original asset restoration: mandatory")
    print("")

    if port_is_open():
        raise RuntimeError(
            f"Port {PORT} is already active. Stop the API before testing."
        )

    if not TARGET_ASSET.is_file():
        raise FileNotFoundError(
            f"Protected test asset was not found: {TARGET_ASSET}"
        )

    original_content = TARGET_ASSET.read_bytes()
    original_digest = sha256_bytes(original_content)

    controlled_marker = (
        b"\nCONTROLLED-INTEGRITY-TEST-ONLY\n"
    )
    modified_digest = ""
    server_result: subprocess.CompletedProcess[str] | None = None
    failed_verification: dict[str, Any] = {}
    restored = False

    try:
        TARGET_ASSET.write_bytes(
            original_content + controlled_marker
        )
        modified_digest = sha256_bytes(
            TARGET_ASSET.read_bytes()
        )

        if modified_digest == original_digest:
            raise RuntimeError(
                "The controlled modification did not change the digest."
            )

        server_result = run_command(
            SECURE_SERVER_SCRIPT
        )

        if VERIFICATION_EVIDENCE.is_file():
            failed_verification = load_json(
                VERIFICATION_EVIDENCE
            )

    finally:
        TARGET_ASSET.write_bytes(original_content)
        restored = (
            sha256_bytes(TARGET_ASSET.read_bytes())
            == original_digest
        )

    # Restore the normal passing verification evidence after restoring
    # the protected asset.
    restored_verification_result = run_command(
        VERIFIER_SCRIPT
    )

    restored_verification = {}
    if VERIFICATION_EVIDENCE.is_file():
        restored_verification = load_json(
            VERIFICATION_EVIDENCE
        )

    combined_output = ""
    if server_result is not None:
        combined_output = (
            server_result.stdout
            + "\n"
            + server_result.stderr
        )

    nonzero_exit = (
        server_result is not None
        and server_result.returncode != 0
    )
    password_not_requested = (
        "Enter the API server private-key password"
        not in combined_output
    )
    network_bind_not_reached = (
        "Uvicorn running on"
        not in combined_output
        and not port_is_open()
    )
    mismatch_detected = (
        failed_verification.get("status") == "failed"
        or bool(
            failed_verification.get(
                "mismatched_assets"
            )
        )
    )
    normal_verification_restored = (
        restored_verification_result.returncode == 0
        and restored_verification.get("status")
        == "passed"
        and restored_verification.get(
            "signature_verified"
        )
        is True
        and restored_verification.get(
            "assets_verified"
        )
        == restored_verification.get(
            "assets_declared"
        )
    )

    passed = all(
        [
            original_digest != modified_digest,
            nonzero_exit,
            password_not_requested,
            network_bind_not_reached,
            mismatch_detected,
            restored,
            normal_verification_restored,
        ]
    )

    evidence = {
        "experiment": (
            "ClimateAdapt V3 integrated fail-closed "
            "asset-startup test v1"
        ),
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "passed" if passed else "failed",
        "target_asset": str(
            TARGET_ASSET.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "controlled_test": True,
        "production_asset_permanently_changed": False,
        "asset_digest_changed_during_test": (
            original_digest != modified_digest
        ),
        "signed_manifest_mismatch_detected": (
            mismatch_detected
        ),
        "secure_server_exit_was_nonzero": nonzero_exit,
        "password_prompt_reached": (
            not password_not_requested
        ),
        "network_bind_reached": (
            not network_bind_not_reached
        ),
        "fail_closed_before_secret_request": (
            nonzero_exit
            and password_not_requested
            and network_bind_not_reached
        ),
        "original_asset_restored": restored,
        "normal_verification_restored": (
            normal_verification_restored
        ),
        "private_signing_key_accessed": False,
        "secret_values_in_evidence": 0,
    }

    TEST_EVIDENCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    TEST_EVIDENCE.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Asset digest changed:       {original_digest != modified_digest}")
    print(f"Manifest mismatch detected: {mismatch_detected}")
    print(f"Server exited non-zero:     {nonzero_exit}")
    print(f"Password prompt reached:    {not password_not_requested}")
    print(f"Network bind reached:       {not network_bind_not_reached}")
    print(f"Original asset restored:    {restored}")
    print(
        "Normal verification restored: "
        f"{normal_verification_restored}"
    )
    print("Private signing key accessed: False")
    print("Secrets recorded:             0")
    print("-" * 78)

    if passed:
        print(
            "RESULT: PASS – modified assets prevented API startup "
            "before secrets or network binding."
        )
    else:
        print("RESULT: FAIL – inspect the generated evidence.")

    print(
        "Evidence: "
        + str(TEST_EVIDENCE.relative_to(PROJECT_ROOT))
    )
    print("=" * 78)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
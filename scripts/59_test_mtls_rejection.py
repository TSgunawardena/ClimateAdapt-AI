"""Verify that the ClimateAdapt mTLS API rejects unauthorized access."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOST = "127.0.0.1"
PORT = 9443

ROOT_CA = (
    PROJECT_ROOT
    / "security"
    / "public"
    / "certificates"
    / "climateadapt_root_ca_v1.crt"
)

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "mtls_rejection_test_v1.json"
)


def safe_error(error: Exception) -> dict[str, str]:
    """Record only the exception type and message."""

    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def main() -> None:
    if not ROOT_CA.is_file():
        raise FileNotFoundError(
            f"Root CA certificate not found: {ROOT_CA}"
        )

    print("=" * 76)
    print("CLIMATEADAPT mTLS – REJECTION CONTROL TEST")
    print("=" * 76)
    print(f"Secure endpoint: https://{HOST}:{PORT}")
    print("Client certificate supplied: No")
    print("Secrets displayed: 0")
    print("-" * 76)

    tcp_reachable = False
    plain_http_rejected = False
    missing_certificate_rejected = False

    plain_http_observation: dict[str, Any]
    missing_certificate_observation: dict[str, Any]

    try:
        with socket.create_connection(
            (HOST, PORT),
            timeout=5,
        ):
            tcp_reachable = True

        print("PASS  Secure API TCP port is reachable")

    except Exception as error:
        print(
            "FAIL  Secure API TCP port is unavailable: "
            f"{type(error).__name__}: {error}"
        )

    plain_connection: (
        http.client.HTTPConnection | None
    ) = None

    try:
        plain_connection = http.client.HTTPConnection(
            HOST,
            PORT,
            timeout=5,
        )
        plain_connection.request("GET", "/health")
        response = plain_connection.getresponse()

        plain_http_observation = {
            "unexpected_http_status": response.status,
        }

        print(
            "FAIL  Plain HTTP unexpectedly returned "
            f"status {response.status}"
        )

    except Exception as error:
        plain_http_rejected = True
        plain_http_observation = safe_error(error)

        print("PASS  Plain HTTP was rejected")
        print(
            "      "
            f"{type(error).__name__}: {error}"
        )

    finally:
        if plain_connection is not None:
            plain_connection.close()

    tls_context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=str(ROOT_CA),
    )
    tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
    tls_context.maximum_version = ssl.TLSVersion.TLSv1_3
    tls_context.check_hostname = True
    tls_context.verify_mode = ssl.CERT_REQUIRED

    try:
        request = urllib.request.Request(
            f"https://{HOST}:{PORT}/health",
            method="GET",
        )

        with urllib.request.urlopen(
            request,
            context=tls_context,
            timeout=5,
        ) as response:
            missing_certificate_observation = {
                "unexpected_http_status": response.status,
            }

            print(
                "FAIL  A caller without a client certificate "
                f"received status {response.status}"
            )

    except Exception as error:
        missing_certificate_rejected = True
        missing_certificate_observation = safe_error(error)

        print(
            "PASS  Missing client certificate was rejected"
        )
        print(
            "      "
            f"{type(error).__name__}: {error}"
        )

    overall_passed = (
        tcp_reachable
        and plain_http_rejected
        and missing_certificate_rejected
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 mutual-TLS rejection test v1"
        ),
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "endpoint": f"https://{HOST}:{PORT}",
        "tcp_reachable": tcp_reachable,
        "plain_http": {
            "rejected": plain_http_rejected,
            "observation": plain_http_observation,
        },
        "https_without_client_certificate": {
            "client_certificate_supplied": False,
            "server_ca_verification_enabled": True,
            "rejected": missing_certificate_rejected,
            "observation": (
                missing_certificate_observation
            ),
        },
        "security_interpretation": (
            "The secure API permits neither plaintext HTTP nor "
            "HTTPS clients without a trusted client certificate."
        ),
        "overall_passed": overall_passed,
        "passwords_required": False,
        "secret_values_in_evidence": 0,
        "gitea_writes": 0,
    }

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 76)
    print(f"TCP reachable:                  {tcp_reachable}")
    print(f"Plain HTTP rejected:            {plain_http_rejected}")
    print(
        "Missing client cert rejected: "
        f"{missing_certificate_rejected}"
    )
    print(f"Secrets recorded:               0")
    print(
        "Evidence: "
        f"{EVIDENCE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "RESULT: "
        + (
            "UNAUTHORIZED ACCESS SAFELY BLOCKED"
            if overall_passed
            else "TEST FAILED"
        )
    )
    print("=" * 76)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
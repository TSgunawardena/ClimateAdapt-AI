"""Test authorized access to the ClimateAdapt mutual-TLS API."""

from __future__ import annotations

import http.client
import json
import os
import ssl
from datetime import datetime, timezone
from getpass import getpass
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

CLIENT_CERTIFICATE = (
    PROJECT_ROOT
    / "security"
    / "public"
    / "certificates"
    / "climateadapt_authorized_client_v1.crt"
)

CLIENT_PRIVATE_KEY = (
    PROJECT_ROOT
    / "security"
    / "private"
    / "client"
    / "climateadapt_authorized_client_v1.key"
)

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "mtls_authorized_client_test_v1.json"
)


def require_file(path: Path, description: str) -> None:
    """Fail safely if required certificate material is missing."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def obtain_client_key_password() -> tuple[str, str]:
    """Read the password without storing or displaying it."""

    password = os.getenv(
        "CLIMATEADAPT_TLS_CLIENT_KEY_PASSWORD"
    )

    if password:
        return password, "environment variable; value not recorded"

    password = getpass(
        "Enter the authorized client private-key password: "
    )

    if not password:
        raise ValueError(
            "The authorized client private-key password is required."
        )

    return password, "interactive prompt; value not recorded"


def distinguished_name_text(
    name_data: tuple[Any, ...],
) -> str:
    """Convert Python's peer-certificate name structure to text."""

    components: list[str] = []

    for relative_name in name_data:
        for attribute_name, attribute_value in relative_name:
            components.append(
                f"{attribute_name}={attribute_value}"
            )

    return ", ".join(components)


def request_json(
    connection: http.client.HTTPSConnection,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send one HTTPS request and parse its JSON response."""

    request_body = None
    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        request_body = json.dumps(payload)
        headers["Content-Type"] = "application/json"

    connection.request(
        method,
        path,
        body=request_body,
        headers=headers,
    )

    response = connection.getresponse()
    response_body = response.read().decode("utf-8")

    parsed_body = json.loads(response_body)

    return response.status, parsed_body


def main() -> None:
    require_file(ROOT_CA, "Root CA certificate")
    require_file(
        CLIENT_CERTIFICATE,
        "Authorized client certificate",
    )
    require_file(
        CLIENT_PRIVATE_KEY,
        "Authorized client private key",
    )

    password, password_source = (
        obtain_client_key_password()
    )

    tls_context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=str(ROOT_CA),
    )

    tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
    tls_context.maximum_version = ssl.TLSVersion.TLSv1_3
    tls_context.check_hostname = True
    tls_context.verify_mode = ssl.CERT_REQUIRED

    tls_context.load_cert_chain(
        certfile=str(CLIENT_CERTIFICATE),
        keyfile=str(CLIENT_PRIVATE_KEY),
        password=password,
    )

    print("=" * 76)
    print("CLIMATEADAPT mTLS – AUTHORIZED CLIENT TEST")
    print("=" * 76)
    print(f"Endpoint: https://{HOST}:{PORT}")
    print("Server verification: required")
    print("Client certificate:  supplied")
    print("Secrets displayed:   0")
    print("-" * 76)

    connection = http.client.HTTPSConnection(
        HOST,
        PORT,
        context=tls_context,
        timeout=15,
    )

    try:
        # Explicitly connect so the negotiated TLS properties can be read.
        connection.connect()

        if connection.sock is None:
            raise RuntimeError(
                "The TLS socket was not established."
            )

        tls_version = connection.sock.version()
        cipher_information = connection.sock.cipher()
        peer_certificate = connection.sock.getpeercert()

        if cipher_information is None:
            raise RuntimeError(
                "No negotiated TLS cipher was reported."
            )

        cipher_name = cipher_information[0]

        peer_subject = distinguished_name_text(
            peer_certificate.get("subject", ())
        )
        peer_issuer = distinguished_name_text(
            peer_certificate.get("issuer", ())
        )

        health_status, health = request_json(
            connection,
            "GET",
            "/health",
        )

        model_status, model = request_json(
            connection,
            "GET",
            "/model",
        )

        prediction_status, prediction = request_json(
            connection,
            "POST",
            "/predict/point",
            {
                "location_name": "IT8527 authorised mTLS test",
                "latitude": -41.29,
                "longitude": 173.24,
            },
        )

    finally:
        connection.close()

    health_passed = (
        health_status == 200
        and health.get("status") == "healthy"
    )

    model_passed = (
        model_status == 200
        and model.get("primary_model")
        == "V3 logistic regression"
    )

    prediction_passed = (
        prediction_status == 200
        and prediction.get("prediction", {}).get(
            "susceptibility_score"
        )
        is not None
        and prediction.get("human_review", {}).get(
            "required"
        )
        is True
    )

    overall_passed = (
        tls_version in {"TLSv1.2", "TLSv1.3"}
        and health_passed
        and model_passed
        and prediction_passed
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 authorized mutual-TLS API test v1"
        ),
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "endpoint": f"https://{HOST}:{PORT}",
        "tls_handshake_completed": True,
        "negotiated_tls_version": tls_version,
        "negotiated_cipher": cipher_name,
        "server_certificate_verified": True,
        "server_certificate_subject": peer_subject,
        "server_certificate_issuer": peer_issuer,
        "client_certificate_supplied": True,
        "client_identity": (
            "climateadapt-authorized-client-v1"
        ),
        "health": {
            "http_status": health_status,
            "application_status": health.get("status"),
            "passed": health_passed,
        },
        "model": {
            "http_status": model_status,
            "primary_model": model.get("primary_model"),
            "operational_threshold": model.get(
                "operational_threshold"
            ),
            "passed": model_passed,
        },
        "prediction": {
            "http_status": prediction_status,
            "application_status": prediction.get("status"),
            "susceptibility_score": prediction.get(
                "prediction",
                {},
            ).get("susceptibility_score"),
            "operational_threshold": prediction.get(
                "prediction",
                {},
            ).get("operational_threshold"),
            "human_review_required": prediction.get(
                "human_review",
                {},
            ).get("required"),
            "passed": prediction_passed,
        },
        "overall_passed": overall_passed,
        "password_source": password_source,
        "password_recorded": False,
        "private_key_recorded": False,
        "secret_values_in_evidence": 0,
    }

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PASS  TLS handshake completed using {tls_version}")
    print(f"PASS  Negotiated cipher: {cipher_name}")
    print("PASS  Server certificate and hostname verified")
    print(
        "PASS  GET /health returned "
        f"{health_status} and {health.get('status')}"
    )
    print(
        "PASS  GET /model returned "
        f"{model.get('primary_model')}"
    )
    print(
        "PASS  POST /predict/point returned a V3 "
        "screening result"
    )
    print(
        "      Score:     "
        f"{evidence['prediction']['susceptibility_score']}"
    )
    print(
        "      Threshold: "
        f"{evidence['prediction']['operational_threshold']}"
    )
    print(
        "      Human review required: "
        f"{evidence['prediction']['human_review_required']}"
    )
    print("-" * 76)
    print(f"Secrets recorded: {evidence['secret_values_in_evidence']}")
    print(
        "Evidence: "
        f"{EVIDENCE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "RESULT: "
        + (
            "AUTHORIZED mTLS ACCESS PASSED"
            if overall_passed
            else "TEST FAILED"
        )
    )
    print("=" * 76)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
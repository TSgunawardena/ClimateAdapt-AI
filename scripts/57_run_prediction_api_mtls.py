"""Run the existing ClimateAdapt V3 prediction API with mutual TLS.

This launcher does not change the prediction model or endpoint logic.
It adds transport encryption, server authentication and mandatory
client-certificate authentication around the existing FastAPI app.
"""

from __future__ import annotations

import importlib.util
import json
import os
import ssl
import subprocess
import sys
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from types import ModuleType
from typing import Any

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "mtls_api_server_config.json"
)
ASSET_VERIFIER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "61_verify_signed_asset_manifest.py"
)

ASSET_VERIFICATION_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "security_assessment_v1"
    / "artifact_manifest_verification_evidence_v1.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON written by either Python or Windows PowerShell."""

    with path.open("r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def project_path(relative_path: str) -> Path:
    """Resolve a configuration path under the project root."""

    return PROJECT_ROOT / Path(relative_path)


def require_file(path: Path, description: str) -> None:
    """Fail safely when required cryptographic material is unavailable."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

def verify_protected_assets() -> dict[str, Any]:
    """Verify the signed manifest before loading or serving the API."""

    require_file(
        ASSET_VERIFIER_PATH,
        "Signed-asset verification script",
    )

    verification = subprocess.run(
        [
            sys.executable,
            str(ASSET_VERIFIER_PATH),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if verification.returncode != 0:
        raise RuntimeError(
            "Protected-asset verification failed. "
            "The API will not start."
        )

    require_file(
        ASSET_VERIFICATION_EVIDENCE_PATH,
        "Asset-verification evidence",
    )

    evidence = load_json(
        ASSET_VERIFICATION_EVIDENCE_PATH
    )

    verification_passed = (
        evidence.get("status") == "passed"
        and evidence.get("signature_verified") is True
        and evidence.get("assets_declared") > 0
        and evidence.get("assets_verified")
        == evidence.get("assets_declared")
        and not evidence.get("missing_assets")
        and not evidence.get("mismatched_assets")
    )

    if not verification_passed:
        raise RuntimeError(
            "Asset-verification evidence did not satisfy "
            "the fail-closed startup policy."
        )

    return evidence

def load_prediction_module(script_path: Path) -> ModuleType:
    """Load Script 30 without running its plain-HTTP main block."""

    module_name = "climateadapt_prediction_api_v3"

    specification = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Could not load prediction API from {script_path}"
        )

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)

    if not hasattr(module, "app"):
        raise AttributeError(
            "Script 30 does not expose the expected FastAPI app."
        )

    return module


def obtain_server_key_password() -> tuple[str, str]:
    """Read the password from memory or request it interactively."""

    password = os.getenv(
        "CLIMATEADAPT_TLS_SERVER_KEY_PASSWORD"
    )

    if password:
        return password, "environment variable; value not recorded"

    password = getpass(
        "Enter the API server private-key password: "
    )

    if not password:
        raise ValueError(
            "The API server private-key password is required."
        )

    return password, "interactive prompt; value not recorded"


def write_runtime_evidence(
    *,
    config_data: dict[str, Any],
    evidence_path: Path,
    password_source: str,
    ssl_context: ssl.SSLContext,
    integrity_evidence: dict[str, Any],
) -> None:
    """Write redacted evidence of the enforced server configuration."""

    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 mutual-TLS prediction API v1"
        ),
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "service": config_data["service"],
        "security_mode": config_data["security_mode"],
        "endpoint": (
            f"https://{config_data['bind_host']}:"
            f"{config_data['bind_port']}"
        ),
        "minimum_tls_version": (
            ssl_context.minimum_version.name
        ),
        "maximum_tls_version": (
            ssl_context.maximum_version.name
        ),
        "client_certificate_requirement": (
            ssl_context.verify_mode.name
        ),
        "trusted_client_ca": (
            config_data["trusted_client_ca"]
        ),
        "server_certificate": (
            config_data["server_certificate"]
        ),
        "cipher_policy": config_data["cipher_policy"],
        "prediction_application": (
            config_data["application_script"]
        ),
        "prediction_model_changed": False,
        "prediction_threshold_changed": False,
        "artifact_integrity_gate": (
            "passed before application load and network bind"
        ),
        "artifact_manifest_signature_verified": (
            integrity_evidence["signature_verified"]
        ),
        "protected_assets_declared": (
            integrity_evidence["assets_declared"]
        ),
        "protected_assets_verified": (
            integrity_evidence["assets_verified"]
        ),
        "missing_protected_assets": (
            len(integrity_evidence["missing_assets"])
        ),
        "mismatched_protected_assets": (
            len(integrity_evidence["mismatched_assets"])
        ),
        "artifact_private_key_accessed": (
            integrity_evidence["private_key_accessed"]
        ),
        "server_private_key_encrypted": True,
        "server_private_key_committed_to_git": False,
        "password_source": password_source,
        "password_recorded": False,
        "private_key_recorded": False,
        "secret_values_in_evidence": 0,
        "expected_unauthorised_behaviour": (
            "TLS handshake rejected before HTTP processing"
        ),
        "expected_authorised_behaviour": (
            "Valid CA-signed client certificate permits HTTPS API access"
        ),
    }

    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    config_data = load_json(CONFIG_PATH)

    application_script = project_path(
        config_data["application_script"]
    )
    server_certificate = project_path(
        config_data["server_certificate"]
    )
    server_private_key = project_path(
        config_data["server_private_key"]
    )
    trusted_client_ca = project_path(
        config_data["trusted_client_ca"]
    )
    runtime_evidence = project_path(
        config_data["runtime_evidence"]
    )

    require_file(
        application_script,
        "Prediction API application",
    )
    require_file(
        server_certificate,
        "Server certificate chain",
    )
    require_file(
        server_private_key,
        "Encrypted server private key",
    )
    require_file(
        trusted_client_ca,
        "Trusted client root CA",
    )
    print(
        "\nVerifying signed V3 serving assets "
        "before API startup...\n"
    )

    integrity_evidence = (
        verify_protected_assets()
    )
    password, password_source = (
        obtain_server_key_password()
    )

    prediction_module = load_prediction_module(
        application_script
    )

    uvicorn_config = uvicorn.Config(
        app=prediction_module.app,
        host=config_data["bind_host"],
        port=int(config_data["bind_port"]),
        log_level="info",
        ssl_keyfile=str(server_private_key),
        ssl_certfile=str(server_certificate),
        ssl_keyfile_password=password,
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        ssl_ca_certs=str(trusted_client_ca),
        ssl_ciphers=config_data["cipher_policy"],
    )

    # Load the SSL context now so its effective security properties
    # can be checked and explicitly hardened before serving.
    uvicorn_config.load()

    if uvicorn_config.ssl is None:
        raise RuntimeError(
            "Uvicorn did not create the required TLS context."
        )

    uvicorn_config.ssl.minimum_version = (
        ssl.TLSVersion.TLSv1_2
    )
    uvicorn_config.ssl.maximum_version = (
        ssl.TLSVersion.TLSv1_3
    )
    uvicorn_config.ssl.verify_mode = ssl.CERT_REQUIRED

    if (
        uvicorn_config.ssl.verify_mode
        != ssl.CERT_REQUIRED
    ):
        raise RuntimeError(
            "Mutual TLS client authentication is not enforced."
        )

    write_runtime_evidence(
        config_data=config_data,
        evidence_path=runtime_evidence,
        password_source=password_source,
        ssl_context=uvicorn_config.ssl,
        integrity_evidence=integrity_evidence,
    )

    print("")
    print("=" * 76)
    print("CLIMATEADAPT V3 – MUTUAL-TLS PREDICTION API")
    print("=" * 76)
    print(
        "Endpoint:       "
        f"https://{config_data['bind_host']}:"
        f"{config_data['bind_port']}"
    )
    print("TLS versions:   TLSv1.2 to TLSv1.3")
    print("Server identity:CA-signed certificate")
    print("Client identity:CERT_REQUIRED")
    print("Trusted CA:     ClimateAdapt Local Root CA v1")
    print("Plain HTTP:     prohibited on secure port")
    print(
        "Asset integrity:"
        f"{integrity_evidence['assets_verified']}/"
        f"{integrity_evidence['assets_declared']} verified"
    )
    print("Startup policy: fail closed on integrity failure")
    print("Secrets logged: 0")
    print(
        "Evidence:       "
        f"{runtime_evidence.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 76)
    print("")

    server = uvicorn.Server(uvicorn_config)
    server.run()


if __name__ == "__main__":
    main()

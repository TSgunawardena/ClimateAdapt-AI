"""Issue an authorized ClimateAdapt mutual-TLS client certificate.

The root CA and client private keys remain encrypted. Passwords are
requested interactively and are never written to evidence or source code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "api_client_certificate_policy.json"
)


def project_path(relative_path: str) -> Path:
    """Resolve a policy path relative to the project root."""

    return PROJECT_ROOT / Path(relative_path)


def load_policy() -> dict[str, Any]:
    """Load and validate the client-certificate policy."""

    with POLICY_PATH.open("r", encoding="utf-8-sig") as policy_file:
        policy = json.load(policy_file)

    if policy["key"]["algorithm"] != "RSA":
        raise ValueError("Only the declared RSA policy is supported.")

    if int(policy["key"]["size_bits"]) < 3072:
        raise ValueError("The client RSA key must be at least 3072 bits.")

    if policy["certificate"]["is_ca"]:
        raise ValueError("The client certificate must not be a CA.")

    usages = policy["certificate"]["extended_key_usage"]

    if "client_authentication" not in usages:
        raise ValueError("Client authentication usage is required.")

    if "server_authentication" in usages:
        raise ValueError(
            "Server authentication usage is prohibited for this identity."
        )

    return policy


def build_subject(subject_policy: dict[str, str]) -> x509.Name:
    """Build the X.509 subject from the documented policy."""

    return x509.Name(
        [
            x509.NameAttribute(
                NameOID.COUNTRY_NAME,
                subject_policy["country_name"],
            ),
            x509.NameAttribute(
                NameOID.ORGANIZATION_NAME,
                subject_policy["organisation_name"],
            ),
            x509.NameAttribute(
                NameOID.ORGANIZATIONAL_UNIT_NAME,
                subject_policy["organisational_unit_name"],
            ),
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                subject_policy["common_name"],
            ),
        ]
    )


def request_client_password() -> bytes:
    """Request and confirm an unseen client-key password."""

    first_password = getpass(
        "Create a password for the authorized client private key: "
    )
    second_password = getpass(
        "Confirm the authorized client private-key password: "
    )

    if first_password != second_password:
        raise ValueError("The client private-key passwords do not match.")

    if len(first_password) < 12:
        raise ValueError(
            "The client private-key password must contain at least "
            "12 characters."
        )

    return first_password.encode("utf-8")


def ensure_outputs_do_not_exist(paths: list[Path]) -> None:
    """Prevent accidental replacement of cryptographic identities."""

    existing_paths = [path for path in paths if path.exists()]

    if existing_paths:
        formatted_paths = "\n".join(
            f"  - {path.relative_to(PROJECT_ROOT)}"
            for path in existing_paths
        )
        raise FileExistsError(
            "Refusing to overwrite existing client material:\n"
            f"{formatted_paths}"
        )


def fingerprint_text(certificate: x509.Certificate) -> str:
    """Return a colon-separated SHA-256 certificate fingerprint."""

    hexadecimal = certificate.fingerprint(hashes.SHA256()).hex().upper()

    return ":".join(
        hexadecimal[index : index + 2]
        for index in range(0, len(hexadecimal), 2)
    )


def sha256_file(path: Path) -> str:
    """Calculate a file digest without exposing its contents."""

    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    policy = load_policy()

    ca_certificate_path = project_path(
        policy["issuer"]["certificate_path"]
    )
    ca_private_key_path = project_path(
        policy["issuer"]["private_key_path"]
    )

    client_private_key_path = project_path(
        policy["outputs"]["private_key_path"]
    )
    client_certificate_path = project_path(
        policy["outputs"]["certificate_path"]
    )
    client_chain_path = project_path(
        policy["outputs"]["certificate_chain_path"]
    )
    evidence_path = project_path(
        policy["outputs"]["evidence_path"]
    )

    ensure_outputs_do_not_exist(
        [
            client_private_key_path,
            client_certificate_path,
            client_chain_path,
            evidence_path,
        ]
    )

    if not ca_certificate_path.is_file():
        raise FileNotFoundError(
            f"Root CA certificate not found: {ca_certificate_path}"
        )

    if not ca_private_key_path.is_file():
        raise FileNotFoundError(
            f"Root CA private key not found: {ca_private_key_path}"
        )

    print("=" * 76)
    print("CLIMATEADAPT – AUTHORIZED mTLS CLIENT CERTIFICATE ISSUANCE")
    print("=" * 76)
    print("Passwords are entered securely and will not be recorded.")

    ca_password = getpass("Enter the root CA private-key password: ")
    client_password = request_client_password()

    ca_certificate = x509.load_pem_x509_certificate(
        ca_certificate_path.read_bytes()
    )

    ca_private_key = serialization.load_pem_private_key(
        ca_private_key_path.read_bytes(),
        password=ca_password.encode("utf-8"),
    )

    if not isinstance(ca_private_key, rsa.RSAPrivateKey):
        raise TypeError("The root CA private key is not RSA.")

    ca_constraints = ca_certificate.extensions.get_extension_for_class(
        x509.BasicConstraints
    ).value

    if not ca_constraints.ca:
        raise ValueError("The issuer certificate is not a CA.")

    ca_usage = ca_certificate.extensions.get_extension_for_class(
        x509.KeyUsage
    ).value

    if not ca_usage.key_cert_sign:
        raise ValueError("The CA certificate cannot sign certificates.")

    client_private_key = rsa.generate_private_key(
        public_exponent=int(policy["key"]["public_exponent"]),
        key_size=int(policy["key"]["size_bits"]),
    )

    now = datetime.now(timezone.utc)
    validity_days = int(policy["certificate"]["validity_days"])

    client_certificate = (
        x509.CertificateBuilder()
        .subject_name(build_subject(policy["subject"]))
        .issuer_name(ca_certificate.subject)
        .public_key(client_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=validity_days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(
                client_private_key.public_key()
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_private_key.public_key()
            ),
            critical=False,
        )
        .sign(
            private_key=ca_private_key,
            algorithm=hashes.SHA256(),
        )
    )

    # Independently verify that the root CA signed the certificate.
    ca_private_key.public_key().verify(
        client_certificate.signature,
        client_certificate.tbs_certificate_bytes,
        padding.PKCS1v15(),
        client_certificate.signature_hash_algorithm,
    )

    extended_usage = (
        client_certificate.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        ).value
    )

    client_auth_present = (
        ExtendedKeyUsageOID.CLIENT_AUTH in extended_usage
    )
    server_auth_absent = (
        ExtendedKeyUsageOID.SERVER_AUTH not in extended_usage
    )

    if not client_auth_present or not server_auth_absent:
        raise ValueError(
            "The generated certificate has incorrect extended usage."
        )

    client_private_key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    client_certificate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encrypted_private_key = client_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(
            client_password
        ),
    )

    client_certificate_pem = client_certificate.public_bytes(
        serialization.Encoding.PEM
    )
    ca_certificate_pem = ca_certificate.public_bytes(
        serialization.Encoding.PEM
    )

    client_private_key_path.write_bytes(encrypted_private_key)
    client_certificate_path.write_bytes(client_certificate_pem)
    client_chain_path.write_bytes(
        client_certificate_pem + ca_certificate_pem
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 authorized mTLS client issuance v1"
        ),
        "certificate_name": policy["certificate_name"],
        "certificate_role": "TLS client authentication",
        "authorised_identity": (
            policy["authorisation"]["identity"]
        ),
        "permitted_components": (
            policy["authorisation"]["permitted_components"]
        ),
        "permitted_service": (
            policy["authorisation"]["permitted_service"]
        ),
        "subject": client_certificate.subject.rfc4514_string(),
        "issuer": client_certificate.issuer.rfc4514_string(),
        "serial_number_hex": format(
            client_certificate.serial_number,
            "X",
        ),
        "key_algorithm": "RSA",
        "key_size_bits": client_private_key.key_size,
        "private_key_format": "Encrypted PKCS#8",
        "signature_hash": (
            client_certificate.signature_hash_algorithm.name
        ),
        "not_valid_before_utc": (
            client_certificate.not_valid_before_utc.isoformat()
        ),
        "not_valid_after_utc": (
            client_certificate.not_valid_after_utc.isoformat()
        ),
        "is_ca": False,
        "client_authentication_present": client_auth_present,
        "server_authentication_absent": server_auth_absent,
        "ca_signature_verified": True,
        "password_recorded": False,
        "secret_values_in_evidence": 0,
        "certificate_sha256_fingerprint": (
            fingerprint_text(client_certificate)
        ),
        "root_ca_sha256_fingerprint": (
            fingerprint_text(ca_certificate)
        ),
        "public_certificate_sha256": sha256_file(
            client_certificate_path
        ),
        "certificate_chain_sha256": sha256_file(
            client_chain_path
        ),
        "private_key_committed_to_git": False,
        "production_limitation": (
            policy["authorisation"]["production_limitation"]
        ),
        "outputs": {
            "public_certificate": str(
                client_certificate_path.relative_to(PROJECT_ROOT)
            ),
            "certificate_chain": str(
                client_chain_path.relative_to(PROJECT_ROOT)
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

    print("-" * 76)
    print("AUTHORIZED mTLS CLIENT CERTIFICATE ISSUED")
    print("-" * 76)
    print(f"Subject:        {evidence['subject']}")
    print(f"Issuer:         {evidence['issuer']}")
    print(f"Identity:       {evidence['authorised_identity']}")
    print(f"Algorithm:      RSA-{evidence['key_size_bits']}")
    print(
        "Client auth:    "
        f"{evidence['client_authentication_present']}"
    )
    print(
        "Server auth absent: "
        f"{evidence['server_authentication_absent']}"
    )
    print(f"CA verified:    {evidence['ca_signature_verified']}")
    print("Encrypted key:  True")
    print(f"Secrets shown:  {evidence['secret_values_in_evidence']}")
    print(
        "Fingerprint:    "
        f"{evidence['certificate_sha256_fingerprint']}"
    )
    print("-" * 76)
    print(
        "Public cert:    "
        f"{client_certificate_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Chain:          "
        f"{client_chain_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Evidence:       "
        f"{evidence_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Private key:    stored locally, encrypted and excluded "
        "from Git"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
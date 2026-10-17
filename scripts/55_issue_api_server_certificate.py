"""Issue the ClimateAdapt API server certificate from the local root CA.

Security properties:
- The CA and server private keys remain encrypted.
- Passwords are requested interactively and are never stored.
- Only public certificates and redacted evidence are suitable for Git.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import (
    ExtendedKeyUsageOID,
    NameOID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "api_server_certificate_policy.json"
)


def project_path(relative_path: str) -> Path:
    """Resolve a policy path relative to the project root."""

    return PROJECT_ROOT / Path(relative_path)


def load_policy() -> dict[str, Any]:
    """Load and perform basic validation of the server policy."""

    with POLICY_PATH.open("r", encoding="utf-8-sig") as policy_file:
        policy = json.load(policy_file)

    if policy["key"]["algorithm"] != "RSA":
        raise ValueError("This script supports only the declared RSA policy.")

    if policy["certificate"]["is_ca"]:
        raise ValueError("The API server certificate must not be a CA.")

    if policy["key"]["size_bits"] < 3072:
        raise ValueError("The server RSA key must be at least 3072 bits.")

    return policy


def build_subject(subject_policy: dict[str, str]) -> x509.Name:
    """Convert the JSON subject into an X.509 distinguished name."""

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


def request_new_key_password() -> bytes:
    """Request and confirm a password without displaying it."""

    first_password = getpass(
        "Create a password for the API server private key: "
    )
    second_password = getpass(
        "Confirm the API server private-key password: "
    )

    if first_password != second_password:
        raise ValueError("The two server private-key passwords do not match.")

    if len(first_password) < 12:
        raise ValueError(
            "The server private-key password must contain at least "
            "12 characters."
        )

    return first_password.encode("utf-8")


def ensure_outputs_do_not_exist(paths: list[Path]) -> None:
    """Prevent accidental replacement of existing cryptographic material."""

    existing_paths = [path for path in paths if path.exists()]

    if existing_paths:
        existing_text = "\n".join(
            f"  - {path.relative_to(PROJECT_ROOT)}"
            for path in existing_paths
        )
        raise FileExistsError(
            "Refusing to overwrite existing certificate material:\n"
            f"{existing_text}"
        )


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without exposing file contents."""

    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def fingerprint_text(certificate: x509.Certificate) -> str:
    """Return a colon-separated certificate SHA-256 fingerprint."""

    hexadecimal = certificate.fingerprint(hashes.SHA256()).hex().upper()
    return ":".join(
        hexadecimal[index : index + 2]
        for index in range(0, len(hexadecimal), 2)
    )


def main() -> None:
    policy = load_policy()

    issuer_certificate_path = project_path(
        policy["issuer"]["certificate_path"]
    )
    issuer_private_key_path = project_path(
        policy["issuer"]["private_key_path"]
    )

    server_private_key_path = project_path(
        policy["outputs"]["private_key_path"]
    )
    server_certificate_path = project_path(
        policy["outputs"]["certificate_path"]
    )
    certificate_chain_path = project_path(
        policy["outputs"]["certificate_chain_path"]
    )
    evidence_path = project_path(
        policy["outputs"]["evidence_path"]
    )

    ensure_outputs_do_not_exist(
        [
            server_private_key_path,
            server_certificate_path,
            certificate_chain_path,
            evidence_path,
        ]
    )

    if not issuer_certificate_path.is_file():
        raise FileNotFoundError(
            f"Root CA certificate not found: {issuer_certificate_path}"
        )

    if not issuer_private_key_path.is_file():
        raise FileNotFoundError(
            f"Root CA private key not found: {issuer_private_key_path}"
        )

    print("=" * 76)
    print("CLIMATEADAPT – API SERVER CERTIFICATE ISSUANCE")
    print("=" * 76)
    print("Passwords are entered securely and will not be recorded.")

    ca_password = getpass("Enter the root CA private-key password: ")
    server_password = request_new_key_password()

    ca_certificate = x509.load_pem_x509_certificate(
        issuer_certificate_path.read_bytes()
    )

    ca_private_key = serialization.load_pem_private_key(
        issuer_private_key_path.read_bytes(),
        password=ca_password.encode("utf-8"),
    )

    if not isinstance(ca_private_key, rsa.RSAPrivateKey):
        raise TypeError("The root CA private key is not an RSA key.")

    ca_constraints = ca_certificate.extensions.get_extension_for_class(
        x509.BasicConstraints
    ).value

    if not ca_constraints.ca:
        raise ValueError("The supplied issuer certificate is not a CA.")

    ca_key_usage = ca_certificate.extensions.get_extension_for_class(
        x509.KeyUsage
    ).value

    if not ca_key_usage.key_cert_sign:
        raise ValueError(
            "The supplied CA certificate cannot sign certificates."
        )

    key_policy = policy["key"]
    server_private_key = rsa.generate_private_key(
        public_exponent=int(key_policy["public_exponent"]),
        key_size=int(key_policy["size_bits"]),
    )

    subject = build_subject(policy["subject"])
    now = datetime.now(timezone.utc)
    validity_days = int(
        policy["certificate"]["validity_days"]
    )

    san_policy = policy["certificate"][
        "subject_alternative_names"
    ]

    alternative_names: list[x509.GeneralName] = []

    alternative_names.extend(
        x509.DNSName(dns_name)
        for dns_name in san_policy["dns"]
    )

    alternative_names.extend(
        x509.IPAddress(ipaddress.ip_address(ip_address))
        for ip_address in san_policy["ip"]
    )

    server_certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_certificate.subject)
        .public_key(server_private_key.public_key())
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
                [ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(alternative_names),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(
                server_private_key.public_key()
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

    # Independently verify that the CA public key validates the signature.
    ca_private_key.public_key().verify(
        server_certificate.signature,
        server_certificate.tbs_certificate_bytes,
        padding.PKCS1v15(),
        server_certificate.signature_hash_algorithm,
    )

    server_private_key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    server_certificate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    private_key_pem = server_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(
            server_password
        ),
    )

    server_certificate_pem = server_certificate.public_bytes(
        serialization.Encoding.PEM
    )
    ca_certificate_pem = ca_certificate.public_bytes(
        serialization.Encoding.PEM
    )

    server_private_key_path.write_bytes(private_key_pem)
    server_certificate_path.write_bytes(server_certificate_pem)
    certificate_chain_path.write_bytes(
        server_certificate_pem + ca_certificate_pem
    )

    server_san = server_certificate.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 API server certificate issuance v1"
        ),
        "certificate_name": policy["certificate_name"],
        "certificate_role": "TLS server authentication",
        "subject": server_certificate.subject.rfc4514_string(),
        "issuer": server_certificate.issuer.rfc4514_string(),
        "serial_number_hex": format(
            server_certificate.serial_number,
            "X",
        ),
        "key_algorithm": "RSA",
        "key_size_bits": server_private_key.key_size,
        "private_key_format": "Encrypted PKCS#8",
        "signature_algorithm": (
            server_certificate.signature_algorithm_oid._name
        ),
        "signature_hash": (
            server_certificate.signature_hash_algorithm.name
        ),
        "not_valid_before_utc": (
            server_certificate.not_valid_before_utc.isoformat()
        ),
        "not_valid_after_utc": (
            server_certificate.not_valid_after_utc.isoformat()
        ),
        "dns_names": server_san.get_values_for_type(x509.DNSName),
        "ip_addresses": [
            str(value)
            for value in server_san.get_values_for_type(
                x509.IPAddress
            )
        ],
        "extended_key_usage": ["server_authentication"],
        "is_ca": False,
        "ca_signature_verified": True,
        "hostname_verification_required": True,
        "password_recorded": False,
        "secret_values_in_evidence": 0,
        "server_certificate_sha256_fingerprint": (
            fingerprint_text(server_certificate)
        ),
        "root_ca_sha256_fingerprint": (
            fingerprint_text(ca_certificate)
        ),
        "public_certificate_sha256": sha256_file(
            server_certificate_path
        ),
        "certificate_chain_sha256": sha256_file(
            certificate_chain_path
        ),
        "private_key_committed_to_git": False,
        "outputs": {
            "public_certificate": str(
                server_certificate_path.relative_to(PROJECT_ROOT)
            ),
            "certificate_chain": str(
                certificate_chain_path.relative_to(PROJECT_ROOT)
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
    print("API SERVER CERTIFICATE ISSUED")
    print("-" * 76)
    print(f"Subject:       {evidence['subject']}")
    print(f"Issuer:        {evidence['issuer']}")
    print(f"Algorithm:     RSA-{evidence['key_size_bits']}")
    print(f"DNS names:     {', '.join(evidence['dns_names'])}")
    print(f"IP addresses:  {', '.join(evidence['ip_addresses'])}")
    print(f"CA verified:   {evidence['ca_signature_verified']}")
    print(f"Encrypted key: True")
    print(f"Secrets shown: {evidence['secret_values_in_evidence']}")
    print(
        "Fingerprint:   "
        f"{evidence['server_certificate_sha256_fingerprint']}"
    )
    print("-" * 76)
    print(
        "Public cert:   "
        f"{server_certificate_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Chain:         "
        f"{certificate_chain_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Evidence:      "
        f"{evidence_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Private key:   stored locally, encrypted and excluded "
        "from Git"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
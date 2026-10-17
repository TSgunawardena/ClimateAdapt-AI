"""Create the self-managed ClimateAdapt root certificate authority."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID


# Locate the project regardless of the current PowerShell directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "security"
    / "root_ca_policy.json"
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def certificate_fingerprint(
    certificate: x509.Certificate,
) -> str:
    """Return a readable SHA-256 certificate fingerprint."""

    fingerprint = certificate.fingerprint(
        hashes.SHA256()
    ).hex().upper()

    return ":".join(
        fingerprint[index:index + 2]
        for index in range(0, len(fingerprint), 2)
    )


def request_ca_password() -> bytes:
    """Request a password without displaying it."""

    first_password = getpass.getpass(
        "Create CA private-key password: "
    )

    second_password = getpass.getpass(
        "Confirm CA private-key password: "
    )

    if first_password != second_password:
        raise ValueError(
            "The two CA passwords did not match."
        )

    if len(first_password) < 12:
        raise ValueError(
            "The CA password must contain at least 12 characters."
        )

    return first_password.encode("utf-8")


def main() -> None:
    """Generate and verify the root CA."""

    # Read the human-readable CA policy.
    with POLICY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        policy = json.load(file_handle)

    private_key_path = (
        PROJECT_ROOT
        / policy["storage"]["private_key"]
    )

    certificate_path = (
        PROJECT_ROOT
        / policy["storage"]["public_certificate"]
    )

    evidence_path = (
        PROJECT_ROOT
        / policy["storage"]["evidence"]
    )

    # Never overwrite cryptographic material silently.
    existing_files = [
        path
        for path in (
            private_key_path,
            certificate_path,
            evidence_path,
        )
        if path.exists()
    ]

    if existing_files:
        existing_text = "\n".join(
            f"  - {path}"
            for path in existing_files
        )

        raise FileExistsError(
            "Refusing to overwrite existing CA files:\n"
            f"{existing_text}"
        )

    password = request_ca_password()

    key_settings = policy["key"]
    subject_settings = policy["subject"]
    certificate_settings = policy["certificate"]

    # Generate the CA asymmetric private key.
    private_key = rsa.generate_private_key(
        public_exponent=int(
            key_settings["public_exponent"]
        ),
        key_size=int(
            key_settings["size_bits"]
        ),
    )

    subject = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COUNTRY_NAME,
                subject_settings["country"],
            ),
            x509.NameAttribute(
                NameOID.ORGANIZATION_NAME,
                subject_settings["organisation"],
            ),
            x509.NameAttribute(
                NameOID.ORGANIZATIONAL_UNIT_NAME,
                subject_settings["organisational_unit"],
            ),
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                subject_settings["common_name"],
            ),
        ]
    )

    current_time = datetime.now(timezone.utc)

    # A root CA is self-signed: subject and issuer are identical.
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(
            current_time - timedelta(minutes=5)
        )
        .not_valid_after(
            current_time
            + timedelta(
                days=int(
                    certificate_settings["validity_days"]
                )
            )
        )
        .add_extension(
            x509.BasicConstraints(
                ca=True,
                path_length=int(
                    certificate_settings[
                        "basic_constraints"
                    ]["path_length"]
                ),
            ),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(
                private_key.public_key()
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                private_key.public_key()
            ),
            critical=False,
        )
        .sign(
            private_key=private_key,
            algorithm=hashes.SHA256(),
        )
    )

    private_key_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    certificate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the CA private key as encrypted PKCS#8.
    encrypted_private_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=(
            serialization.BestAvailableEncryption(
                password
            )
        ),
    )

    private_key_path.write_bytes(
        encrypted_private_key
    )

    # Attempt owner-only permissions where supported.
    try:
        os.chmod(private_key_path, 0o600)
    except OSError:
        pass

    # The certificate is public and safe to distribute.
    certificate_path.write_bytes(
        certificate.public_bytes(
            serialization.Encoding.PEM
        )
    )

    # Independently verify the self-signature.
    public_key = certificate.public_key()

    public_key.verify(
        certificate.signature,
        certificate.tbs_certificate_bytes,
        padding.PKCS1v15(),
        certificate.signature_hash_algorithm,
    )

    evidence = {
        "experiment": (
            "ClimateAdapt IT8527 self-managed root CA v1"
        ),
        "generated_at_utc": current_time.isoformat(),
        "policy_file": str(
            POLICY_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "policy_sha256": sha256_file(POLICY_PATH),
        "key_algorithm": "RSA",
        "key_size_bits": private_key.key_size,
        "public_exponent": (
            private_key.public_key()
            .public_numbers()
            .e
        ),
        "private_key_format": "Encrypted PKCS#8",
        "private_key_location": str(
            private_key_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "private_key_committed_to_git": False,
        "password_recorded": False,
        "certificate_location": str(
            certificate_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "certificate_sha256": sha256_file(
            certificate_path
        ),
        "subject": certificate.subject.rfc4514_string(),
        "issuer": certificate.issuer.rfc4514_string(),
        "serial_number_hex": format(
            certificate.serial_number,
            "X",
        ),
        "signature_hash": (
            certificate.signature_hash_algorithm.name
        ),
        "sha256_fingerprint": (
            certificate_fingerprint(certificate)
        ),
        "not_valid_before_utc": (
            certificate.not_valid_before_utc.isoformat()
        ),
        "not_valid_after_utc": (
            certificate.not_valid_after_utc.isoformat()
        ),
        "basic_constraints": {
            "ca": True,
            "path_length": 0,
        },
        "self_signature_verified": True,
        "secret_values_in_evidence": 0,
    }

    evidence_path.write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "CLIMATEADAPT IT8527 - SELF-MANAGED ROOT CA CREATED"
    )
    print("=" * 72)
    print(
        f"Subject:      "
        f"{certificate.subject.rfc4514_string()}"
    )
    print(
        f"Algorithm:    RSA {private_key.key_size}"
    )
    print(
        "Signature:    "
        f"{certificate.signature_hash_algorithm.name}"
    )
    print(
        "Fingerprint:  "
        f"{certificate_fingerprint(certificate)}"
    )
    print(
        "Private key:  "
        f"{private_key_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Certificate:  "
        f"{certificate_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Evidence:     "
        f"{evidence_path.relative_to(PROJECT_ROOT)}"
    )
    print("Encrypted key: Yes")
    print("Self-check:    PASS")
    print("Secrets shown: 0")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ROOT CA CREATION FAILED: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
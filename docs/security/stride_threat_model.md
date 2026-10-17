# ClimateAdapt AI STRIDE Threat Model

## Purpose

This threat model identifies security threats affecting the ClimateAdapt AI
prediction service and its supporting artefacts. It is based on the assets and
trust boundaries documented in `target_asset_inventory.md`.

The primary protection boundary is the connection between an authorised
ClimateAdapt client and the FastAPI prediction service. The secondary protection
boundary is between the prediction service and the deployed model artefacts it
loads.

## Adversaries and capabilities

### External or unauthorised client

An unauthorised program can discover the prediction API address and attempt to
send requests. Before this assessment, the API accepted HTTP requests without
requiring a cryptographic client identity.

### Network-positioned attacker

An attacker able to observe or modify traffic between separately deployed
components may attempt to read prediction requests, alter responses or
impersonate the API.

The current prototype runs locally, reducing the likelihood of this attack.
However, the boundary would become exposed if the API were moved to a virtual
machine, container or remote host.

### Local unprivileged process

A process running under another account or with limited host access may attempt
to read private keys or modify deployed model files.

### Malicious or compromised administrator

An administrator may intentionally replace an artefact or misuse valid
credentials. Cryptography cannot completely eliminate this risk, but
verification, restricted key storage, rotation and audit evidence can improve
detection and accountability.

## Risk scale

Likelihood and impact are rated as Low, Medium or High. Overall risk is derived
from the combination of likelihood and impact.

## STRIDE threat table

| ID | STRIDE category | Asset and boundary | Threat scenario | Likelihood | Impact | Initial risk | Planned control | Verification evidence | Residual risk |
|---|---|---|---|---|---|---|---|---|---|
| T01 | Spoofing | A04, TB02 | An unauthorised application calls the prediction API while pretending to be the approved Streamlit or agent client | High | High | High | Self-managed CA and mutual TLS requiring an approved client certificate | Request with valid client certificate succeeds; request without certificate fails | Theft of a valid client private key could permit impersonation |
| T02 | Spoofing | A05, TB02 | A malicious service impersonates the ClimateAdapt API and receives client requests | Medium | High | High | Server certificate issued by the project CA, hostname verification and CA trust validation | Authorised client succeeds with genuine server; connection using incorrect trust fails | Compromise of the CA private key could enable trusted server impersonation |
| T03 | Tampering | A03, TB02 | Prediction requests or responses are modified while travelling between components | Medium | High | High | TLS 1.2 or later with authenticated encryption through mutual TLS | Valid TLS session succeeds; altered or invalid TLS connection cannot be established | TLS does not protect data after it reaches a compromised endpoint |
| T04 | Tampering | A01 and A02, TB03 | The logistic-regression model or susceptibility raster is replaced or modified, changing prediction results | Medium | High | High | SHA-256 artefact manifest signed using Ed25519 and verified before serving | Original artefacts verify; a modified test copy fails verification | A malicious artefact signed by a stolen authorised signing key would still verify |
| T05 | Repudiation | A08, TB01 and TB05 | A user or workflow denies making an approval decision or publishing an artefact | Medium | Medium | Medium | Existing timestamped workflow audit records, run identifiers and independently verified Gitea issue evidence | Audit record shows reviewer, decision, time and run identifier | Local administrators may still modify unsigned historical log files |
| T06 | Information disclosure | A03, TB02 | A network observer reads requested locations, model scores or API responses | Medium | Medium | Medium | TLS encryption with server and client authentication | Packet or client test shows HTTPS is required and plain HTTP is unavailable | Endpoint compromise can expose decrypted request and response data |
| T07 | Information disclosure | A06, A07 and A10 | Private CA, signing or Gitea credentials are committed to Git or exposed in evidence | Medium | High | High | Git-ignore rules, separate private directories, filesystem access restrictions and secret scanning | Git reports private paths as ignored; evidence scan finds no private keys or tokens | A user could manually copy a secret into an unprotected location |
| T08 | Denial of service | A03, TB02 | Repeated requests exhaust API resources or prevent legitimate prediction requests | Medium | Medium | Medium | Safe request validation, timeouts and local service monitoring | Invalid and malformed requests are rejected safely | Mutual TLS does not stop excessive requests from a valid compromised client; rate limiting remains future work |
| T09 | Elevation of privilege | A04 and A06, TB02 | An attacker obtains certificate-authority capability or reuses a retired client identity | Low | High | Medium | Restricted CA-key storage, limited CA use, certificate separation and documented key rotation | New key succeeds after rotation; retired identity or signature fails under the active trust policy | Host-administrator compromise can bypass filesystem restrictions |
| T10 | Tampering | A09, TB04 | The planning model or its deployment package is replaced, altering generated recommendations | Low | High | Medium | Include the GGUF deployment artefact or release hash in the signed manifest where available | Recorded GGUF hash verifies against the signed manifest | The manifest proves integrity and approved provenance, not factual correctness or safety of model output |

## Control mapping

### C01 — Self-managed certificate authority

The project will operate its own local CA and issue separate server and client
certificates. This control primarily addresses T01, T02 and T09.

### C02 — Mutually authenticated TLS

The prediction API will require a certificate from both sides of the
communication. This control addresses T01, T02, T03 and T06.

### C03 — Signed artefact manifest

SHA-256 hashes of the deployed model artefacts will be recorded in a manifest
signed with an Ed25519 private key. This addresses T04 and supports T10.

### C04 — Private-key storage controls

Private keys will be stored outside tracked repository content in ignored
directories with restricted operating-system permissions. This addresses T07
and reduces the likelihood associated with T09.

### C05 — Key rotation

The signing key will be rotated from version 1 to version 2. The new signature
will be accepted while the old signature will fail under the updated active
trust policy. This addresses key-lifecycle requirements and T09.

### C06 — Existing human-review and audit controls

The existing agent workflow records a reviewer, decision, timestamp and run
identifier before an approved Gitea action. This supports accountability for
T05 but does not replace the cryptographic controls.

## Controls intentionally not claimed

The implementation does not claim that cryptography:

- proves that a prediction is scientifically correct;
- prevents a valid authorised user from acting maliciously;
- protects data after complete host compromise;
- prevents all denial-of-service attacks;
- makes the local prototype equivalent to enterprise PKI;
- proves that a signed model is unbiased or safe.

These limitations will be evaluated as residual risks.
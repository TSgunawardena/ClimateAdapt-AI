\# ClimateAdapt AI Security Target and Asset Inventory



\## Target environment



The target environment is the locally deployed ClimateAdapt AI integrated

project running on a Windows workstation. The system includes a Streamlit user

interface, LangGraph agents, a FastAPI prediction service, a deployed V3

logistic-regression model, geospatial raster products, an Ollama-hosted

planning language model and an optional local Gitea integration.



Although the prototype components run on one physical workstation, they are

treated as separate logical trust zones. The Streamlit interface and agent

workflow act as clients of the prediction API. The API accesses the deployed

model and regional raster artefacts. The planning agent communicates separately

with Ollama, and approved workflow actions may communicate with Gitea.



The cryptography assessment focuses on protecting communication with the

prediction API and verifying the integrity of deployed model artefacts.



\## Asset inventory



| ID | Asset | Why it has value | Security requirement | Consequence of compromise |

|---|---|---|---|---|

| A01 | V3 logistic-regression model | Produces the deployed susceptibility-screening result | Integrity and provenance | A modified model could produce misleading screening results |

| A02 | Full-region V3 susceptibility rasters | Supply the scores returned by the API | Integrity and availability | Modified rasters could change place-analysis results |

| A03 | Prediction API requests and responses | Contain locations, model scores and screening interpretations | Confidentiality, integrity and authenticated origin | Interception or modification could mislead users |

| A04 | Authorised client identity | Determines which component may call the protected API | Authentication and private-key protection | A stolen identity could permit unauthorised API access |

| A05 | API server identity | Allows clients to confirm that they reached the genuine API | Authentication and private-key protection | Server impersonation could capture or alter requests |

| A06 | Certificate-authority private key | Establishes trust for server and client certificates | Strong confidentiality and restricted use | Theft would allow an attacker to create trusted identities |

| A07 | Artefact-signing private key | Proves the approved origin of the deployment manifest | Strong confidentiality and controlled rotation | Theft would allow malicious artefacts to be signed |

| A08 | Human-review decision and audit record | Records accountability for agent-generated recommendations | Integrity and traceability | Modification could falsely represent human approval |

| A09 | Fine-tuned planning model and GGUF package | Generates the planning-support language output | Integrity and provenance | Replacement could change safety behaviour or instructions |

| A10 | Gitea access token | Authorises issue publication after human approval | Confidentiality and least privilege | Disclosure could allow unauthorised repository actions |



\## Trust boundaries



\### TB01 — User to Streamlit interface



A user supplies a location and requests a screening result. Input validation is

required because this boundary accepts user-controlled data.



\### TB02 — Streamlit or agent client to prediction API



The client sends a request to the prediction service. In the original baseline,

this used plain HTTP and did not require a client certificate. This assessment

will protect this boundary using mutually authenticated TLS.



\### TB03 — Prediction API to local model artefacts



The API reads the deployed model and raster products. This assessment will use

a signed SHA-256 manifest to verify these artefacts before they are trusted.



\### TB04 — Planning agent to Ollama



The planning agent submits structured prediction evidence to the locally served

language model. This boundary is in scope for threat analysis but is not the

primary cryptographic deployment boundary.



\### TB05 — Approved workflow to Gitea



A workflow may create a Gitea issue only after recorded human approval. The

Gitea token remains a separate secret and is not stored in assessment evidence.



\## Security scope



The practical cryptographic implementation will provide:



1\. A self-managed local certificate authority.

2\. Separate prediction-API server and authorised-client certificates.

3\. Mutual TLS for authenticated and encrypted API communication.

4\. SHA-256 artefact hashing.

5\. Ed25519 signing and verification of the deployment manifest.

6\. Signing-key rotation and rejection of retired-key signatures.

7\. Safe failure when authentication or integrity verification fails.



\## Scope limitations



The implementation is a local assessment prototype rather than a production

public-key infrastructure. It does not protect against complete compromise of

the Windows host, theft of an already-authorised private key, malicious actions

by an authorised administrator or denial-of-service attacks. Production

deployment should separate components across independently administered

systems, such as virtual machines or containers.


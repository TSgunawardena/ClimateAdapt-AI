# ClimateAdapt Planner 4B — Model Card

## Model summary

`climateadapt-planner:4b` is a locally deployed planning-language model for the ClimateAdapt AI prototype. It converts verified flood-susceptibility screening evidence into a constrained planning brief for authorised human review. It does not produce flood predictions itself and cannot approve, publish, fund, regulate, or initiate a public-safety action.

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3-4B |
| Model family | Qwen3 dense causal language model |
| Parameters | 4.0 billion |
| Base-model licence | Apache License 2.0 |
| Adaptation | QLoRA through Unsloth |
| LoRA rank / alpha | 16 / 16 |
| Training epochs | 3 |
| Learning rate | 0.0001 |
| Effective batch size | 8 |
| Maximum training sequence length | 2,048 tokens |
| Training objective | Completion-only supervised fine-tuning |
| Export | Merged GGUF, Q4_K_M |
| Local runtime | Ollama |
| Runtime name | `climateadapt-planner:4b` |
| Evaluation mode | Thinking disabled, temperature 0 |

## Intended use

The model is intended only for research and prototype decision-support workflows. It receives structured, already-validated evidence from the ClimateAdapt prediction and analysis components and produces a JSON planning brief containing a status, summary, limitations, recommendations and a human-approval requirement.

Every planning interpretation requires authorised human review. The model is not a hydrodynamic simulator, emergency-warning system, regulatory decision maker, engineering design tool, or replacement for GIS, engineering, community or cultural review.

## Training data

The project constructed a 243-example instruction dataset:

- 192 training examples;
- 24 validation examples; and
- 27 held-out test examples.

The examples teach constrained reporting, numeric fidelity, explicit limitations, escalation behaviour and the mandatory human-review boundary. Synthetic examples were used for training and validation. Three real ClimateAdapt outputs were held only in the final test split. The test split was not opened during fine-tuning or validation-driven model selection.

Dataset construction and audit are implemented in:

- `scripts/25_create_instruction_dataset.py`
- `scripts/26_validate_instruction_dataset.py`
- `data/instruction/`
- `outputs/instruction_dataset/`

## Fine-tuning

The authentic training notebook is:

`notebooks/28_finetune_qwen3_4b_qlora_colab.ipynb`

QLoRA was selected to adapt a four-billion-parameter model within the memory available from a Google Colab Tesla T4. The base weights remained quantised and frozen while low-rank adapter parameters were trained. Rank 16 and alpha 16 provided a compact adapter appropriate for the small, specialised instruction dataset. Three epochs were predeclared, and validation loss was recorded at each epoch.

Recorded training evidence:

| Measurement | Result |
|---|---:|
| Training examples | 192 |
| Validation examples | 24 |
| Training time | 1,012.84 seconds |
| Final reported training loss | 0.5118 |
| Best validation loss | 0.004355 |
| GPU | Tesla T4 |

The very low final validation loss must not be treated as proof of generalisation. The dataset is deliberately narrow and template-constrained, so memorisation of its reporting structure is possible. Final behaviour was therefore assessed on a locked test split and no further tuning was performed from test results.

## Held-out evaluation

The same prompt structure, context length, output budget, thinking setting, temperature and seed were used for the base and fine-tuned models. The model backend was the intended variable.

| Metric | Base Qwen3 4B | Fine-tuned model |
|---|---:|---:|
| Parse success | 1.0000 | 1.0000 |
| Schema completeness | 0.9959 | 1.0000 |
| Status accuracy | 0.2593 | 0.7778 |
| Human-approval accuracy | 0.9630 | 1.0000 |
| Task-guardrail accuracy | 0.2593 | 0.7778 |
| Numeric fidelity | 0.9590 | 0.9785 |
| Reference-token F1 | 0.3911 | 0.8031 |
| Behaviour score | 0.7689 | 0.9323 |

Fine-tuning materially improved the targeted reporting behaviour but did not produce perfect status or task-guardrail accuracy. Deterministic validation, workflow rules and human approval remain necessary.

## Limitations and risks

- The training dataset is small and specialised.
- The model can produce fluent but incorrect language.
- It does not independently verify geospatial evidence.
- It must not reinterpret the V3 susceptibility score as a calibrated flood probability.
- Its safety behaviour was evaluated on a limited test set.
- Results apply to the recorded prompt, chat template and runtime configuration.
- Quantisation can change behaviour relative to full-precision inference.
- The model does not establish causality, engineering safety or regulatory suitability.

## Deployment controls

- Thinking is disabled for controlled JSON generation.
- Structured output is parsed and schema-checked.
- Numeric fidelity and required safety statements are checked.
- Unsafe or invalid output is rejected or replaced by a deterministic fallback.
- A human-review checkpoint is mandatory.
- Gitea publication is prohibited unless explicit approval is recorded.
- The composed workflow records provenance and an audit log.

## Artefact availability

The approximately 2.5 GB GGUF is excluded from Git. The separately supplied model release contains the GGUF, a portable Ollama `Modelfile`, this model card, installation instructions, the training notebook and SHA-256 checksums.

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems, 36*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Qwen Team. (2025). *Qwen3 technical report* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2505.09388

Unsloth AI. (n.d.). *Unsloth*. GitHub. Retrieved October 13, 2026, from https://github.com/unslothai/unsloth

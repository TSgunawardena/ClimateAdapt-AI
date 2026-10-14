# ClimateAdapt Planning LLM — Reproducibility Guide

## Inspecting the implementation

The complete fine-tuning workflow is recorded in:

`notebooks/28_finetune_qwen3_4b_qlora_colab.ipynb`

Supporting repository files include:

- `scripts/25_create_instruction_dataset.py`
- `scripts/26_validate_instruction_dataset.py`
- `scripts/27_evaluate_base_llm.py`
- `scripts/29_compare_base_and_finetuned_llm.py`
- `scripts/37_document_qlora_training.py`
- `data/instruction/`
- `outputs/language_finetuning_v1/`
- `outputs/language_model_comparison_v1/`

The notebook contains the executable Unsloth model loading, QLoRA adapter configuration, completion-only supervised training, validation, adapter saving, merged-model export and Q4_K_M GGUF conversion code.

## Recorded configuration

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| Seed | 8532 |
| Max sequence length | 2,048 |
| Quantised training | 4-bit QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Effective batch size | 8 |
| Loss | Completion-only |
| Training examples | 192 |
| Validation examples | 24 |
| Test examples | 27, held out |

The authoritative machine-readable record is:

`outputs/language_finetuning_v1/fine_tuning_summary.json`

## Reproduction sequence

1. Create the instruction dataset with Script 25.
2. Audit it with Script 26.
3. Upload the train and validation JSONL files to the Colab runtime.
4. Run the notebook from top to bottom on a compatible CUDA GPU.
5. Preserve the training history, summary, adapter and GGUF export.
6. Register the Q4_K_M GGUF in Ollama with its supplied `Modelfile`.
7. Evaluate the base model on validation data with Script 27.
8. Run the final locked comparison with Script 29 once only.
9. Do not tune prompts, weights or hyperparameters from final test results.

Exact byte-for-byte reproduction is not guaranteed across GPU, CUDA, PyTorch, Transformers, TRL, Unsloth and llama.cpp versions. The notebook and recorded version metadata make the procedure inspectable, while the supplied checksum identifies the evaluated GGUF artefact exactly.

## Local installation of the supplied GGUF

From inside the extracted model-release directory:

```powershell
ollama create climateadapt-planner:4b -f .\Modelfile
ollama list
```

Then test structured generation through the Ollama chat API. The assessment evaluation used thinking disabled, temperature 0, a 4,096-token context and a fixed seed.

## Licence and attribution

Qwen3-4B is distributed under Apache License 2.0. The repository cites Qwen3, LoRA, QLoRA and Unsloth in the training notebook, model card and technical report. Project-specific instruction data and workflow code remain subject to the repository's stated licence and data-source conditions.

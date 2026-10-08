# Model artifacts

Large model files are intentionally excluded from Git.

Expected local artifacts include:

- `climateadapt_unet_v2_best.pt`
- `logistic_regression_baseline_v2.joblib`
- the locally imported Ollama model `climateadapt-planner:4b`

The U-Net checkpoint and logistic baseline can be reproduced using the numbered
training scripts. The planning model was QLoRA fine-tuned, merged, exported to
GGUF and imported into Ollama. Never commit model binaries or API credentials.


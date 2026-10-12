# ClimateAdapt AI – Script 42

`42_train_unet_v3.py` trains the V3 GroupNorm Compact U-Net using only frozen
training and validation tiles. It does not instantiate a test dataset or open
a test tile file.

V3 retains the V2 network topology while changing the input contract from 25
channels to 11. This lets the experiment test broader geography, improved
splitting and stable land-cover engineering without conflating those changes
with an unrelated architecture.

The checkpoint is selected using overlap-aware validation average precision.
The Dice, precision and recall at 0.5 printed during training are monitoring
metrics only. The operational threshold is selected later using validation.

Run from the project root:

```powershell
python .\scripts\42_train_unet_v3.py
```

# ClimateAdapt AI – Script 43

`43_select_validation_threshold_v3.py` loads the best V3 U-Net checkpoint and
runs inference only on Boulder Bank Glenduan and Delaware Bay estuary
validation tiles. Overlapping predictions are averaged so each geographic
cell is scored once.

The operational threshold maximises F2 subject to specificity of at least
0.80. A Dice-optimal threshold is recorded only as a secondary comparison.
Neither threshold is selected from test performance.

Run from the project root:

```powershell
python .\scripts\43_select_validation_threshold_v3.py
```

# ClimateAdapt AI – Script 41

`41_train_validate_v3_baselines.py` trains two baselines using V3 training
tiles and evaluates them on V3 validation tiles:

1. a constant training-prevalence baseline;
2. class-balanced logistic regression.

Overlapping pixels are deduplicated using their full-raster coordinates. The
logistic threshold is selected by validation F2 because missed susceptible
cells carry the larger operational cost. No file inside the test tile folder
is opened.

Run from the project root:

```powershell
python .\scripts\41_train_validate_v3_baselines.py
```

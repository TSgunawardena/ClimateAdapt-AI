# ClimateAdapt AI – Script 38

`38_audit_v3_training_areas.py` audits the nine NCC coastal-label areas before
any V3 split, tile creation or model training.

Run from the project root:

```powershell
python .\scripts\38_audit_v3_training_areas.py
```

The script reads only `data/processed/full_region/rasters`. It does not read
V1/V2 predictions or modify any trained model, threshold or previous output.

It uses a 750 m valid-land context, 128×128 candidate tiles, a 64-pixel stride
and a minimum 20% context coverage. The generated split recommendation is
provisional and must be reviewed before creating V3 tiles.

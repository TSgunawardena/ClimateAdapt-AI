# ClimateAdapt AI – Script 40

`40_create_v3_training_tiles.py` creates 128×128 tiles from the frozen V3
geographic split. Numeric normalisation is fitted exclusively on training
pixels. Validation and test tiles are retained without label-based sampling.

The model inputs are 11 channels:

- normalised elevation
- normalised slope
- normalised `log1p(distance to coastline)`
- eight one-hot semantic land-cover groups

The eight groups are project engineering aggregations, not replacements for
the official LCDB taxonomy. Their exact code mappings are written into the
tile metadata.

Run from the project root:

```powershell
python .\scripts\40_create_v3_training_tiles.py
```

This script does not train a model or evaluate the locked test split.

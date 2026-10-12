# ClimateAdapt AI – Script 39

`39_create_v3_geographic_split.py` creates and freezes the V3 area-level
training, validation and locked-test masks.

## Frozen design

- Training: Nelson CBD, Nelson Haven, Rocks Road, Tahunanui, Waimea inlet
- Validation: Boulder Bank Glenduan, Delaware Bay estuary
- Locked testing: Delaware Open Coast, Oananga Bay

Each valid-land context cell is assigned to its nearest labelled coastal area.
A 320 m total exclusion belt is inserted where different splits meet. This is
half the 640 m ground width of a 128-pixel tile on each side combined.

Run from the project root:

```powershell
python .\scripts\39_create_v3_geographic_split.py
```

The script refuses to replace an existing frozen V3 split with a different
configuration. It does not create tiles, normalise features, train a model,
select a checkpoint, or select a threshold.

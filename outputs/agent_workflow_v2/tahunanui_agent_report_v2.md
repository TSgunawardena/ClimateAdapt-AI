# ClimateAdapt AI – LangGraph V2 Planning Brief

**Run ID:** `ae812e01d36e`  
**Area:** Tāhunanui independent test area  
**Final status:** human_approved  
**Priority:** HIGH

## Multi-paradigm evidence

- Frozen operational threshold: 0.54
- Original susceptible cells: 13306
- Physically supported cells: 10561
- Human-review cells: 3542
- Terrain-rule rejections: 384
- Coast-disconnected rejections: 2361
- Uncertainty/error enrichment: 2.2920370256850364

## Planning recommendations

1. **Review physically supported susceptibility corridors** – The frozen 0.54 model threshold produced 13,306 candidate cells; 10,561 remained after terrain and coastal-connectivity checks.
2. **Inspect amber human-review locations** – 3,542 supported cells (33.54%) were flagged by epistemic uncertainty or predictive entropy.
3. **Compare conservative and precise model behaviours** – The logistic baseline emphasises sensitivity while the U-Net provides more spatially precise segmentation; disagreement must be interpreted rather than automatically averaged.
4. **Obtain qualified site-specific assessment** – The prototype does not simulate water depth, velocity, event timing, tides, rainfall, drainage capacity or exposed assets.

## Mandatory limitations

- Independent test evidence is evaluation evidence, not a local guarantee.
- Monte Carlo dropout is approximate epistemic uncertainty, not a full posterior.
- The Physicalist component is an explainable screening guardrail, not hydrodynamics.
- No emergency, engineering, funding or regulatory decision may be automated.

## Human decision

- Action: approve
- Reviewer: Prototype demo
- Notes: Approved through --auto-approve for prototype demonstration.

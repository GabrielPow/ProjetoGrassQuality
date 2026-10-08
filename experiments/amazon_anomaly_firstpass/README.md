# Amazon anomaly detection: first-pass feasibility check

Quick test of the PLAN.md hypothesis, *"does a foundation-model embedding improve unsupervised
detection of environmental disturbance in the Amazon?"*, before investing in TerraMind + a
Temporal Transformer AE. AlphaEarth (Google, precomputed annual 64-d embeddings) stands in for
TerraMind; no model is trained.

- **AOI:** Tapajós / BR-163, Pará `[-57.0, -7.5, -55.0, -5.5]`
- **Labels (evaluation only):** MapBiomas LULC. For event year T in 2021–2024: normal = stable
  forest or stable pasture over T-2..T+1; anomaly = forest at T-2 and T-1, then anthropic use
  (deforest) or mining (class 30) at **both** T and T+1.
- **Sample:** 150 points / stratum / year = 2,400 points; spatial 0.1° block split (30% test).
- **Scores:** history = mean of T-2, T-1. AlphaEarth cosine change, NDVI/NBR drop, standardized
  Sentinel-2 band distance (annual Cloud Score+ median), kNN on deltas fit on train normals only.

## Run
```
../../venv/Scripts/python.exe build_dataset.py   # ~25 min of Earth Engine calls -> data/amazon_firstpass.npz
../../venv/Scripts/python.exe analyze.py         # metrics + figures -> results/
../../venv/Scripts/python.exe map_demo.py 2022   # wall-to-wall map -> results/map_2022.png
```
numpy-only metrics because scipy/sklearn/torch DLLs are blocked by Windows App Control here.

## Results (test blocks, AUROC with block-bootstrap 95% CI)

| Task | AlphaEarth cosine change | Best spectral |
|---|---|---|
| All anomalies vs all normals | **0.89** (0.86–0.92) | 0.82 NDVI drop (0.78–0.85) |
| Deforest vs stable forest | **0.99** (0.97–0.99) | 0.93 S2 kNN (0.90–0.95) |
| Deforest vs stable pasture (hard) | **0.85** (0.80–0.90) | 0.76 NBR drop (0.70–0.82) |
| Mining vs all normals | **0.87** (0.82–0.91) | 0.82 NDVI drop (0.78–0.87) |

TPR at 5% FPR, all vs all: AlphaEarth 0.68 vs best spectral 0.49. AlphaEarth wins in every event year.
Full table: `results/metrics.csv`.

## Caveats that matter before scaling up
1. **The static baseline is a red flag.** kNN on the year-T embedding alone, with no time axis,
   scores *higher* overall (0.93) and on mining (0.95). The embedding mostly says *what the land
   is now*. A temporal model has to beat this static baseline, or the Transformer AE isn't justified.
2. **Balanced sampling (50% anomalies) hides the base rate.** Real disturbance is <1% of pixels, so at
   5% FPR a map is dominated by false positives. Report precision at realistic prevalence or at
   low FPR (≤1%).
3. **Label circularity.** MapBiomas is a Landsat-derived classification, and AlphaEarth was trained on
   related optical/radar inputs. Validate on independent events (PRODES/DETER polygons, fire scars,
   known garimpo sites).
4. **Annual granularity.** Both AlphaEarth and the labels are annual. The plan's X(t1..tn) sequence
   would need sub-annual TerraMind embeddings to add timing information.
5. One AOI and pixel-level points; no fire-only or river/water-change anomalies yet.

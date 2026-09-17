# ProjetoGrassQuality — Grassland Quality Visualizer

A prototype that estimates pasture (grassland) quality — **Low / Medium / High** — for any
point on the map in Brazil, by combining Google Earth Engine's **AlphaEarth satellite
embeddings** with a small classifier trained on **MapBiomas Pastagem (Qualidade)** labels.

Given a lat/lon, the app pulls that point's multi-year AlphaEarth embedding sequence from
Earth Engine, runs it through a trained temporal classifier, and shows the predicted class
together with the per-class probabilities.

## Two tracks in this repo

This repo covers two things:

1. **`notebooks/` — the assignment's minimum required pipeline** (IBM8924 — AC de Projeto
   2026.2, Grupo 1): raw Sentinel-2 image patches (not embeddings) over the same MapBiomas
   pasture-vigor points, a trivial baseline, and a shallow logistic-regression baseline —
   see [Minimum required pipeline](#minimum-required-pipeline-notebooks) below.
2. **`model/` + `app.py` — an additional, more advanced model** (AlphaEarth embeddings +
   an LSTM classifier, plus a Streamlit map UI), built before the raw-image pipeline above
   and kept as a bonus beyond the assignment's minimum. Described next.

## Advanced bonus model: AlphaEarth embeddings + LSTM (beyond the assignment minimum)

Not required by the assignment rubric (which asks for a raw-image tensor and a shallow
baseline — see [Minimum required pipeline](#minimum-required-pipeline-notebooks)), but kept
because it's a working, validated alternative approach worth documenting.

1. **Base representation — AlphaEarth (Google Earth Engine)**
   Instead of hand-engineering spectral indices (NDVI, etc.), the project uses Earth Engine's
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` collection — AlphaEarth's pre-trained satellite
   embeddings. Each pixel/year is already reduced by Google's foundation model to a 64-dimensional
   vector (bands `A00`…`A63`) that summarizes that location's yearly satellite signal (optical,
   radar, texture, etc.), so no raw imagery needs to be downloaded or processed by hand.

2. **Labels — MapBiomas Pastagem (Vigor)**
   Training points are labeled using MapBiomas' **Pasture Vigor** asset
   (`mapbiomas_collection90_pasture_vigor_v1`, reference year 2022), which is used as a
   stand-in for "quality" — **1/2/3 = low/medium/high vigor**. The vigor layer is masked
   to pasture-only pixels using MapBiomas' land-use/land-cover asset (class code `15` =
   Pasture) before sampling, so labels only come from land that MapBiomas itself classifies
   as pasture. Because MapBiomas' vigor map is itself a model output rather than field
   ground truth, it's best treated as a large, somewhat noisy training signal — the notebook
   flags swapping in LAPIG's field-validated points as a natural next step.

   Labeled points are drawn with `ee.Image.stratifiedSample` (100 points per vigor class,
   scale 30 m, seed 42) so the 3 classes come out balanced from the start, over a ~65 km ×
   65 km (~4,977 km²) test region around Goiânia, Goiás — chosen because published,
   field-validated pasture-degradation data exists there for later benchmarking.

3. **Dataset**
   For each of the 300 labeled sample points, the AlphaEarth embedding was pulled at each
   point (`sampleRegions`, scale 10 m) for a run of years (2018–2023, 6 years) and stacked
   into a sequence, so a training example is `(6 years × 64-dim embedding) → vigor class`.
   Only points with a complete 6-year sequence are kept (all 300/300 in this run). The
   resulting array is checked into `model/grassland_embeddings_dataset.npz` (`X`:
   `(300, 6, 64)` float32, `y`: `(300,)` int64, balanced 100/100/100 across the 3 classes;
   `years`: `[2018 2019 2020 2021 2022 2023]`; `label_map`: MapBiomas vigor code (1/2/3) →
   zero-indexed model class (0/1/2)).

4. **Model — a small temporal classifier on top of the embeddings**
   Rather than classifying a single year in isolation, a lightweight `GrasslandTemporalClassifier`
   (defined in [app.py](app.py)) consumes the whole 6-year embedding sequence for a point:
   - 1-layer **LSTM** (`input_dim=64`, `hidden_dim=64`) over the yearly embedding sequence
   - A small **MLP head** (`Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→3)`) on the LSTM's
     final hidden state, producing logits over the 3 quality classes
   - Softmax at inference time turns the logits into Low/Medium/High probabilities

   This model is trained in `model/grassland_quality_finetuning.ipynb` — the fine-tuning
   step, since AlphaEarth already did the expensive self-supervised pretraining and this
   notebook only trains the lightweight task-specific head on top of it. Training details:
   - 70/15/15 train/val/test split (210/45/45 of the 300 points), batch size 16, seed 42
   - Adam optimizer, `lr=1e-3`, `weight_decay=1e-4`, cross-entropy loss, 30 epochs
   - Reached ~80% validation accuracy by epoch 30 (`val_loss 0.693`, `val_acc 0.800`),
     evaluated on the held-out test set with a classification report and confusion matrix

   The resulting checkpoint is committed at `model/grassland_temporal_classifier.pt`. The
   notebook also includes an optional synthetic-data fallback (`USE_SYNTHETIC = True`) for
   smoke-testing the PyTorch pipeline without a working Earth Engine connection.

5. **App — Streamlit map UI ([app.py](app.py))**
   A single-file Streamlit app that:
   - Lets you click a point on a satellite/street map (via `folium` / `streamlit-folium`)
   - Fetches that point's AlphaEarth embedding for each year in the configured range directly
     from Earth Engine (`ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")`)
   - Loads a `GrasslandTemporalClassifier` checkpoint you upload (architecture parameters —
     number of classes, hidden size, number of layers — are configurable in the sidebar so they
     can be matched to whatever checkpoint you load, including `model/grassland_temporal_classifier.pt`)
   - Runs the embedding sequence through the model and displays the predicted class and a
     probability bar chart, plus the raw yearly embedding vectors

## Known limitations / next steps

This is a first working scaffold, not a finished pipeline (see the notebook's own closing
notes for more):

- Labels come from MapBiomas' *Pasture Vigor* map (itself a model output), not field
  ground truth — swapping in or blending LAPIG's field-validated points would give a
  stronger evaluation set.
- Predictions are per-point, not per-property — a natural next step is aggregating
  embeddings over CAR property polygons (mean/percentile pooling) once real property
  boundaries are available.
- "Quality" here means vegetation vigor only; it doesn't yet factor in legal/environmental
  risk signals (CAR boundaries, PRODES/DETER deforestation flags, embargo status).
- The MapBiomas Pasture class code (`15`) and asset/band names are worth reconfirming
  against the current legend at brasil.mapbiomas.org, since these can shift between
  MapBiomas collection versions.

## Minimum required pipeline (`notebooks/`)

The assignment (`TODO.md`, Grupo 1) requires a labeled **image** dataset (a raw
`X ∈ R^(N×H×W×C)` tensor, not embeddings), ≥1500 patches, a trivial baseline, a shallow
main baseline, and specific evaluation/characterization artifacts. Since the assignment's
suggested raw-image sources don't cover Brazil (EuroSAT = Europe only, UC Merced = US
only), this pipeline pulls raw **Sentinel-2 L2A** imagery via Earth Engine instead —
reusing the exact same MapBiomas pasture-vigor points and Earth Engine project as the
bonus model above.

- **`notebooks/01_dados.ipynb`** — samples pasture-vigor points across 4 years
  (2019–2022, same AOI near Goiânia) to reach ≥1500 balanced points, extracts a 64×64
  Sentinel-2 patch (bands B2/B3/B4/B8/B11/B12) per point, drops patches with too many
  missing (cloud) pixels, and splits 70/15/15 into train/val/test by spatial grid block
  (so nearby patches never land in different partitions). Writes
  `data/processed/s2_patches.npz` and `data/processed/points_metadata.csv`.
- **`notebooks/02_caracterizacao.ipynb`** — class distribution, per-band statistics,
  NDVI/GNDVI histograms, an example grid per class, and commentary on imbalance/artifacts.
- **`notebooks/03_baseline.ipynb`** — trivial baseline (majority class) vs. a logistic
  regression on aggregated per-patch features (band means/stds + NDVI/GNDVI); accuracy,
  macro F1, confusion matrix, and a commented error analysis (≥4 misclassified examples).
- **`notebooks/dataset_utils.py`** — shared helpers (`load_dataset`, NDVI/GNDVI, RGB
  composite, aggregated feature extraction) used by notebooks 02 and 03.

Run them in order: `01_dados.ipynb` → `02_caracterizacao.ipynb` → `03_baseline.ipynb`.
`01_dados.ipynb` needs the same Earth Engine access as `app.py` and is the slowest step
(patch extraction runs in small synchronous batches against Earth Engine) — *fill in
actual run time here after running it*.

## Project structure

```
ProjetoGrassQuality/
├── app.py                                    # Streamlit app: map UI + EE fetch + inference (bonus model)
├── requirements.txt                          # Python dependencies
├── data/
│   └── processed/                            # E1: s2_patches.npz + points_metadata.csv (from 01_dados.ipynb)
├── notebooks/                                # Assignment minimum pipeline (E1-E4)
│   ├── dataset_utils.py                      # Shared helpers: NDVI/GNDVI, RGB composite, aggregated features
│   ├── 01_dados.ipynb                        # Ingestion: MapBiomas points + Sentinel-2 patches + split
│   ├── 02_caracterizacao.ipynb               # Characterization: stats, NDVI histograms, example grids
│   └── 03_baseline.ipynb                     # Trivial + logistic-regression baseline, metrics, error analysis
├── model/                                    # Bonus advanced model (not required by the assignment minimum)
│   ├── grassland_quality_finetuning.ipynb    # Training notebook: EE extraction + LSTM fine-tuning
│   ├── grassland_embeddings_dataset.npz      # Training data: AlphaEarth embeddings + MapBiomas labels
│   └── grassland_temporal_classifier.pt      # Trained LSTM classifier checkpoint
└── README.md / README_pt.md
```

## Running it locally

### Prerequisites

- Python 3.10+
- A Google Cloud project with the **Earth Engine API** enabled, and access to Earth Engine
  (sign up at https://earthengine.google.com if you don't have it yet)

### Setup

```bash
# from the project root
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell: venv\Scripts\Activate.ps1)
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Authenticate Earth Engine (once)

Run this once from a terminal in the same environment — it caches credentials to disk. The
app itself does not run the interactive OAuth flow:

```bash
earthengine authenticate
```

### Reproduce the minimum required pipeline (notebooks/)

From `notebooks/`, run in order (each needs the same Earth Engine access set up above):

1. `01_dados.ipynb` — *(fill in actual run time, e.g. "~40 min for ~1800 points")*
2. `02_caracterizacao.ipynb` — *(fill in actual run time)*
3. `03_baseline.ipynb` — *(fill in actual run time)*

Library versions: see `requirements.txt` (no extra install needed beyond it —
`dataset_utils.py` is a local module imported by notebooks 02 and 03, not a package).

### Run the app

```bash
streamlit run app.py
```

This opens the app in your browser. Then:

1. Enter your Earth Engine GCP project ID in the sidebar.
2. Upload a model checkpoint — use `model/grassland_temporal_classifier.pt` for the trained
   model included in this repo.
3. Make sure the sidebar's architecture settings match the checkpoint: `num_classes=3`,
   `hidden_dim=64`, `num_layers=1`, embedding years `2018–2023` (these are the defaults, and
   match `model/grassland_temporal_classifier.pt`).
4. Click a point on the map, then click **"Fetch embeddings and predict"** to see the
   predicted pasture quality (Low / Medium / High) and class probabilities for that location.

### Retraining / re-running the fine-tuning notebook (optional)

`model/grassland_quality_finetuning.ipynb` runs standalone (in Colab or locally with
Jupyter) and needs the same Earth Engine access as the app. It re-pulls MapBiomas labels
and AlphaEarth embeddings, retrains the LSTM classifier, and writes out
`grassland_embeddings_dataset.npz` and `grassland_temporal_classifier.pt`. It also has a
`USE_SYNTHETIC = True` toggle to smoke-test the training loop without Earth Engine access.

## Team & task division

*(fill in before submitting — assignment requires this)*

| Member | Responsibilities |
|---|---|
| *Name 1* | *e.g. `01_dados.ipynb`, Earth Engine pipeline* |
| *Name 2* | *e.g. `02_caracterizacao.ipynb`, `03_baseline.ipynb`, README* |

## Generative AI usage declaration

*(fill in / confirm before submitting — assignment requires this)*

Claude Code (Anthropic) was used to: audit the existing prototype against the assignment
rubric (`TODO.md`), design the raw-Sentinel-2 patch-extraction pipeline and the
spatial-block train/val/test split, and scaffold `notebooks/01_dados.ipynb`,
`02_caracterizacao.ipynb`, `03_baseline.ipynb`, and `dataset_utils.py`. The notebooks were
not yet executed against live Earth Engine data at the time of writing — *update this
section, and the discussion below, with any further AI-assisted work plus final
human-reviewed results before submission*.

## Discussion

*(assignment requires ≤2 printed pages covering the sections below — this is a skeleton;
fill in with real numbers/observations after running the three notebooks in `notebooks/`)*

**Problem.** Estimate pasture/grassland quality (proxied by MapBiomas vigor: low/medium/
high) from satellite imagery, for a region in Brazil where the assignment's suggested
image datasets (EuroSAT, UC Merced) don't apply since they don't cover Brazil.

**Data.** *(sources, license, period, point count after QC — from `01_dados.ipynb`)*

**Characterization.** *(class balance, per-band stats, NDVI distribution, notable
artifacts — from `02_caracterizacao.ipynb`)*

**Experimental protocol.** 70/15/15 split by spatial grid block (no scene leakage across
partitions), fixed seed (42).

**Baseline and results.** *(trivial vs. logistic-regression accuracy/F1 macro, confusion
matrix, error analysis — from `03_baseline.ipynb`)*

**Limitations.** Vigor label is a model output, not field ground truth (see
[Known limitations](#known-limitations--next-steps) above); patches are single points in
time/space, not property-level aggregates.

**Future work.** Three directions (also see the bonus model's own list above): (1) swap in
or blend LAPIG field-validated points for evaluation; (2) compare this raw-image shallow
baseline against the AlphaEarth-embedding LSTM on the *same* points, as a controlled
ablation of embeddings vs. raw pixels; (3) aggregate over CAR property polygons instead of
points once boundaries are available.

# ProjetoGrassQuality — Grassland Quality Visualizer

A prototype that estimates pasture (grassland) quality — **Low / Medium / High** — for any
point on the map in Brazil, by combining Google Earth Engine's **AlphaEarth satellite
embeddings** with a small classifier trained on **MapBiomas Pastagem (Qualidade)** labels.

Given a lat/lon, the app pulls that point's multi-year AlphaEarth embedding sequence from
Earth Engine, runs it through a trained temporal classifier, and shows the predicted class
together with the per-class probabilities.

## How it works

1. **Base representation — AlphaEarth (Google Earth Engine)**
   Instead of hand-engineering spectral indices (NDVI, etc.), the project uses Earth Engine's
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` collection — AlphaEarth's pre-trained satellite
   embeddings. Each pixel/year is already reduced by Google's foundation model to a 64-dimensional
   vector (bands `A00`…`A63`) that summarizes that location's yearly satellite signal (optical,
   radar, texture, etc.), so no raw imagery needs to be downloaded or processed by hand.

2. **Labels — MapBiomas Pastagem**
   Training points are labeled using MapBiomas' Pastagem (pasture quality) classification,
   collapsed into 3 classes: **Low, Medium, High** quality. The label map stored alongside the
   training data (`model/grassland_embeddings_dataset.npz`) maps MapBiomas' original quality
   codes (1, 2, 3) to model class indices (0, 1, 2).

3. **Dataset**
   For each labeled sample point, the AlphaEarth embedding was pulled for a run of years
   (2018–2023, 6 years) and stacked into a sequence, so a training example is
   `(6 years × 64-dim embedding) → quality class`. The resulting array is checked into
   `model/grassland_embeddings_dataset.npz` (`X`: `(300, 6, 64)` float32, `y`: `(300,)` int64,
   balanced 100/100/100 across the 3 classes; `years`: `[2018 2019 2020 2021 2022 2023]`;
   `label_map`: MapBiomas code → class index).

4. **Model — a small temporal classifier on top of the embeddings**
   Rather than classifying a single year in isolation, a lightweight `GrasslandTemporalClassifier`
   (defined in [app.py](app.py)) consumes the whole 6-year embedding sequence for a point:
   - 1-layer **LSTM** (`input_dim=64`, `hidden_dim=64`) over the yearly embedding sequence
   - A small **MLP head** (`Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→3)`) on the LSTM's
     final hidden state, producing logits over the 3 quality classes
   - Softmax at inference time turns the logits into Low/Medium/High probabilities

   This model was trained (outside this repo snapshot) on the dataset above and the resulting
   checkpoint is committed at `model/grassland_temporal_classifier.pt`.

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

## Project structure

```
ProjetoGrassQuality/
├── app.py                                  # Streamlit app: map UI + EE fetch + inference
├── requirements.txt                        # Python dependencies
├── model/
│   ├── grassland_embeddings_dataset.npz     # Training data: AlphaEarth embeddings + MapBiomas labels
│   └── grassland_temporal_classifier.pt     # Trained LSTM classifier checkpoint
└── README.md
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

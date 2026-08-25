"""
Grassland Quality Visualizer — temporary Streamlit prototype

Loads a trained GrasslandTemporalClassifier checkpoint (produced by the fine-tuning
notebook), lets you click a point on a map, pulls that location's multi-year
AlphaEarth embedding sequence from Google Earth Engine, and shows the model's
predicted pasture quality class.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Before first use, authenticate Earth Engine ONCE from a terminal in this same
environment (this caches credentials to disk — the app itself does not run the
interactive OAuth flow):
    earthengine authenticate
"""

import io

import ee
import folium
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Model definition — must match the architecture used in the training notebook
# ----------------------------------------------------------------------------


class GrasslandTemporalClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1, num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


st.set_page_config(page_title="Grassland Quality Visualizer", layout="wide")
st.title("🌾 Grassland Quality Visualizer")
st.caption(
    "Temporary prototype — click a point, pull its AlphaEarth embedding history, "
    "run it through your trained model."
)

# ----------------------------------------------------------------------------
# Sidebar: configuration
# ----------------------------------------------------------------------------

st.sidebar.header("Configuration")

ee_project_id = st.sidebar.text_input(
    "Earth Engine GCP project ID", value="", placeholder="your-gcp-project-id"
)

uploaded_model = st.sidebar.file_uploader("Model checkpoint (.pt)", type=["pt"])

st.sidebar.subheader("Model architecture (must match training)")
num_classes = st.sidebar.number_input("Number of classes", min_value=2, max_value=10, value=3, step=1)
hidden_dim = st.sidebar.number_input("hidden_dim", min_value=8, max_value=512, value=64, step=8)
num_layers = st.sidebar.number_input("num_layers", min_value=1, max_value=4, value=1, step=1)

class_names_raw = st.sidebar.text_input(
    "Class names (comma-separated, in model index order — check the 'Class mapping' "
    "printed in the notebook)",
    value="Low vigor, Medium vigor, High vigor",
)
class_names = [c.strip() for c in class_names_raw.split(",")]

year_start, year_end = st.sidebar.slider(
    "Embedding years (must match the training window)", 2017, 2025, (2018, 2023)
)
embed_years = list(range(year_start, year_end + 1))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Before first use, authenticate Earth Engine once from a terminal in this same "
    "environment:\n\n```\nearthengine authenticate\n```"
)

# ----------------------------------------------------------------------------
# Earth Engine init (cached so it only runs once per project ID)
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def init_ee(project_id):
    ee.Initialize(project=project_id)
    return True


ee_ready = False
if ee_project_id:
    try:
        init_ee(ee_project_id)
        ee_ready = True
    except Exception as e:
        st.sidebar.error(f"Earth Engine failed to initialize: {e}")
else:
    st.sidebar.info("Enter your Earth Engine GCP project ID to continue.")

# ----------------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(file_bytes, num_classes, hidden_dim, num_layers):
    model = GrasslandTemporalClassifier(
        input_dim=64, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes
    )
    state_dict = torch.load(io.BytesIO(file_bytes), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = None
if uploaded_model is not None:
    try:
        model = load_model(uploaded_model.getvalue(), num_classes, hidden_dim, num_layers)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(
            f"Could not load model: {e}\n\nCheck that num_classes / hidden_dim / "
            "num_layers above match what you trained with."
        )
else:
    st.sidebar.info("Upload your trained .pt checkpoint to enable predictions.")

# ----------------------------------------------------------------------------
# Map — click to pick a location
# ----------------------------------------------------------------------------

st.subheader("1. Pick a location")

if "clicked_point" not in st.session_state:
    # Default: near Goiânia, GO — same test region used in the notebook
    st.session_state.clicked_point = {"lat": -16.68, "lng": -49.25}

m = folium.Map(
    location=[st.session_state.clicked_point["lat"], st.session_state.clicked_point["lng"]],
    zoom_start=9,
)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)
folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
folium.LayerControl().add_to(m)
folium.Marker(
    [st.session_state.clicked_point["lat"], st.session_state.clicked_point["lng"]],
    tooltip="Selected point",
).add_to(m)

map_data = st_folium(m, height=450, width=None)

if map_data and map_data.get("last_clicked"):
    st.session_state.clicked_point = map_data["last_clicked"]

lat = st.session_state.clicked_point["lat"]
lon = st.session_state.clicked_point["lng"]
st.write(f"Selected point: **{lat:.5f}, {lon:.5f}**")

# ----------------------------------------------------------------------------
# Fetch embeddings + run prediction
# ----------------------------------------------------------------------------

st.subheader("2. Run the model")


@st.cache_data(show_spinner=False)
def fetch_embedding_sequence(lat, lon, years):
    point = ee.Geometry.Point([lon, lat])
    embeddings_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    rows = []
    for yr in years:
        start = ee.Date.fromYMD(yr, 1, 1)
        end = start.advance(1, "year")
        img = (
            embeddings_col.filter(ee.Filter.date(start, end))
            .filter(ee.Filter.bounds(point))
            .mosaic()
        )
        info = img.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=10).getInfo()
        if not info or info.get("A00") is None:
            rows.append(None)
            continue
        vec = [info.get(f"A{str(b).zfill(2)}") for b in range(64)]
        rows.append(vec)
    return rows


run_clicked = st.button(
    "Fetch embeddings and predict",
    type="primary",
    disabled=not (ee_ready and model is not None),
)

if not ee_ready:
    st.warning("Set a valid Earth Engine project ID in the sidebar first.")
elif model is None:
    st.warning("Upload a trained model checkpoint in the sidebar first.")

if run_clicked:
    with st.spinner("Querying Earth Engine for this location's embedding history..."):
        rows = fetch_embedding_sequence(lat, lon, embed_years)

    missing_years = [yr for yr, r in zip(embed_years, rows) if r is None]
    if missing_years:
        st.error(
            f"No embedding data found for year(s): {missing_years}. "
            "Try a different point or year range."
        )
    else:
        X = np.array([rows], dtype=np.float32)  # shape (1, T, 64)
        with torch.no_grad():
            logits = model(torch.tensor(X))
            probs = torch.softmax(logits, dim=1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

        st.success(f"Predicted class: **{pred_label}**")

        display_names = (
            class_names[: len(probs)]
            if len(class_names) >= len(probs)
            else [f"class_{i}" for i in range(len(probs))]
        )
        prob_df = pd.DataFrame({"class": display_names, "probability": probs})
        st.bar_chart(prob_df.set_index("class"))

        with st.expander("Raw embedding vectors by year"):
            embed_df = pd.DataFrame(rows, index=embed_years)
            embed_df.index.name = "year"
            st.dataframe(embed_df)

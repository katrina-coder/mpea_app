"""
NSGAN - Multi-Principal Element Alloy Generative Design Tool
Based on: Li & Birbilis (2024), Integrating Materials and Manufacturing Innovation
Re-built for local Streamlit deployment.
"""

import io
import pandas as pd
import numpy as np
import torch
from torch import nn
from joblib import load
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPEA Generative Design",
    page_icon="⚗️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ELEMENT_NAMES = [
    "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga","Ge","Hf",
    "Li","Mg","Mn","Mo","N","Nb","Nd","Ni","Pd","Re","Sc","Si",
    "Sn","Ta","Ti","V","W","Y","Zn","Zr"
]

# Atomic masses (g/mol) for each element (same order as ELEMENT_NAMES)
MASSES = [
    107.8682, 26.9815386, 10.811, 12.0107, 40.078, 58.933195, 51.9961,
    63.546, 55.845, 69.723, 72.64, 178.49, 6.941, 24.305, 54.938045,
    95.94, 14.0067, 92.90638, 144.242, 58.6934, 106.42, 186.207,
    44.955912, 28.0855, 118.71, 180.94788, 47.867, 50.9415, 183.84,
    88.90585, 65.409, 91.224
]

# Molar volumes (cm³/mol) for each element
VOLUMES = [
    10.27, 10.0, 4.39, 5.29, 26.2, 6.67, 7.23, 7.11, 7.09, 11.8,
    13.63, 13.44, 13.02, 14.0, 7.35, 9.38, 13.54, 10.83, 20.59,
    6.59, 8.56, 8.86, 15.0, 12.06, 16.29, 10.85, 10.64, 8.32,
    9.47, 19.88, 9.16, 14.02
]

PROCESS_MAP = {
    'process_1': "As-cast processes, inclusive of 'arc-melted'",
    'process_2': "Arc-melted processes followed by artificial aging",
    'process_3': "Arc-melted processes followed by annealing",
    'process_4': "Powder processing techniques (powder metallurgy)",
    'process_5': "Novel synthesis techniques (i.e., ball milling)",
    'process_6': "Arc-melted processes followed by wrought processing techniques",
    'process_7': "Cryogenic treatments",
}

MODEL_DIR = "models"

# ── Neural network definition (must match saved weights) ─────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
        )

    def forward(self, noise):
        return self.model(noise)


# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    gen = Generator()
    gen.load_state_dict(
        torch.load(f"{MODEL_DIR}/generator_net_MPEA.pt", map_location="cpu")
    )
    gen.eval()

    regressors = {
        "Tensile Strength":  load(f"{MODEL_DIR}/tensile_regressor.joblib"),
        "Elongation":        load(f"{MODEL_DIR}/elongation_regressor.joblib"),
        "Yield Strength":    load(f"{MODEL_DIR}/yield_regressor.joblib"),
        "Hardness":          load(f"{MODEL_DIR}/hardness_regressor.joblib"),
    }
    classifiers = {
        "FCC": load(f"{MODEL_DIR}/FCC_classifier.joblib"),
        "BCC": load(f"{MODEL_DIR}/BCC_classifier.joblib"),
        "HCP": load(f"{MODEL_DIR}/HCP_classifier.joblib"),
        "IM":  load(f"{MODEL_DIR}/IM_classifier.joblib"),
    }
    return gen, regressors, classifiers


@st.cache_data
def load_dataset():
    df = pd.read_excel(f"{MODEL_DIR}/MPEA_parsed_dataset.xlsx")
    data_np = df.to_numpy()
    comp_data = data_np[:, 14:53].astype(float)
    comp_min = np.min(comp_data, axis=0)
    comp_max = np.max(comp_data, axis=0)
    process_names = df.columns.values[46:53]
    return comp_min, comp_max, process_names


# ── Optimisation problem ──────────────────────────────────────────────────────
class AlloyOptimizationProblem(Problem):
    def __init__(self, selected_objectives, generator, regressors, classifiers,
                 comp_min, comp_max):
        super().__init__(n_var=10, n_obj=len(selected_objectives), xl=-3.0, xu=3.0)
        self.selected_objectives = selected_objectives
        self.generator   = generator
        self.regressors  = regressors
        self.classifiers = classifiers
        self.comp_min    = comp_min
        self.comp_max    = comp_max
        self.masses      = np.array(MASSES)
        self.volumes     = np.array(VOLUMES)

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            fake_alloys = self.generator(x_tensor).numpy()
        fake_alloys = fake_alloys * self.comp_max + self.comp_min

        densities = (np.sum(fake_alloys[:, :32] * self.masses, axis=1) /
                     np.sum(fake_alloys[:, :32] * self.volumes, axis=1))

        obj_map = {
            "Tensile Strength": -self.regressors["Tensile Strength"].predict(fake_alloys),
            "Elongation":       -self.regressors["Elongation"].predict(fake_alloys),
            "Yield Strength":   -self.regressors["Yield Strength"].predict(fake_alloys),
            "Hardness":         -self.regressors["Hardness"].predict(fake_alloys),
            "FCC":              -self.classifiers["FCC"].predict(fake_alloys).astype(float),
            "BCC":              -self.classifiers["BCC"].predict(fake_alloys).astype(float),
            "HCP":              -self.classifiers["HCP"].predict(fake_alloys).astype(float),
            "IM":               -self.classifiers["IM"].predict(fake_alloys).astype(float),
            "Density":           densities,          # minimise density
            "Aluminum Content": -fake_alloys[:, 1],  # maximise Al
        }
        f_values = [obj_map[o] for o in self.selected_objectives]
        out["F"] = np.column_stack(f_values)


# ── Helper: build composition string ─────────────────────────────────────────
def build_alloy_name(composition):
    parts = []
    for j, c in enumerate(composition):
        if c > 0.005:
            parts.append(f"{ELEMENT_NAMES[j]}{round(c, 3)}")
    return "".join(parts)


# ── Helper: decode optimal alloys ────────────────────────────────────────────
def decode_alloys(res_X, generator, comp_min, comp_max, regressors,
                  classifiers, process_names):
    result_tensor = torch.tensor(res_X, dtype=torch.float32)
    with torch.no_grad():
        optimal_alloys = generator(result_tensor).numpy()

    optimal_alloys = optimal_alloys * comp_max + comp_min
    # Renormalise compositions so they sum to 1
    row_sums = np.sum(optimal_alloys[:, :32], axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    optimal_alloys[:, :32] /= row_sums

    alloy_names = [build_alloy_name(optimal_alloys[i, :32])
                   for i in range(optimal_alloys.shape[0])]

    # Convert process soft assignment → one-hot
    process_indices = np.argmax(optimal_alloys[:, 32:], axis=1)
    one_hot = np.zeros_like(optimal_alloys[:, 32:])
    for i, idx in enumerate(process_indices):
        one_hot[i, idx] = 1
    optimal_alloys[:, 32:] = one_hot

    masses_arr  = np.array(MASSES)
    volumes_arr = np.array(VOLUMES)
    densities = (np.sum(optimal_alloys[:, :32] * masses_arr, axis=1) /
                 np.sum(optimal_alloys[:, :32] * volumes_arr, axis=1))

    props = {
        "Elongation":       regressors["Elongation"].predict(optimal_alloys),
        "Tensile Strength": regressors["Tensile Strength"].predict(optimal_alloys),
        "Yield Strength":   regressors["Yield Strength"].predict(optimal_alloys),
        "Hardness":         regressors["Hardness"].predict(optimal_alloys),
        "Density":          densities,
    }

    phase_names_list = []
    phase_labels = ["FCC", "BCC", "HCP", "IM"]
    for i in range(optimal_alloys.shape[0]):
        phases = [lbl for lbl in phase_labels
                  if classifiers[lbl].predict(optimal_alloys[i:i+1])[0] > 0]
        phase_names_list.append("+".join(phases) if phases else "Unknown")

    proc_name_list = [
        PROCESS_MAP.get(process_names[process_indices[i]], "Unknown")
        for i in range(optimal_alloys.shape[0])
    ]

    result_df = pd.DataFrame({
        "Alloy Composition":    alloy_names,
        "Processing Method":    proc_name_list,
        "Predicted Phase":      phase_names_list,
        "Hardness (HV)":        np.round(props["Hardness"], 2),
        "Tensile Strength (MPa)": np.round(props["Tensile Strength"], 2),
        "Yield Strength (MPa)": np.round(props["Yield Strength"], 2),
        "Elongation (%)":       np.round(props["Elongation"], 2),
        "Density (g/cm³)":      np.round(props["Density"], 3),
    })

    return result_df, props


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚗️ MPEA Generative Design Tool")
st.markdown(
    """
    This tool uses the **NSGAN** framework (Non-dominant Sorting optimisation-based
    Generative Adversarial Network) to generate novel Multi-Principal Element Alloy (MPEA)
    compositions optimised for user-selected mechanical and phase objectives.

    > *Based on: Li & Birbilis (2024), Integrating Materials and Manufacturing Innovation*
    """
)

# Load models once
generator, regressors, classifiers = load_models()
comp_min, comp_max, process_names = load_dataset()

st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Optimisation Settings")

    OBJECTIVE_CHOICES = [
        "Tensile Strength", "Elongation", "Yield Strength", "Hardness",
        "Density", "FCC", "BCC", "HCP", "IM", "Aluminum Content"
    ]
    selected_objectives = st.multiselect(
        "Optimisation Objectives",
        OBJECTIVE_CHOICES,
        default=["Tensile Strength", "Elongation"],
        help="Select 2 or more objectives. All are maximised except Density (minimised)."
    )

    pop_size = st.slider("Population Size", min_value=10, max_value=200, value=50, step=10)
    n_gen    = st.slider("Number of Generations", min_value=10, max_value=500, value=200, step=10)
    seed_val = st.number_input("Random Seed", min_value=0, value=2, step=1)

    st.divider()
    st.markdown(
        "**Objective guide:**\n"
        "- Mechanical: maximised\n"
        "- Phase (FCC/BCC/HCP/IM): maximised probability\n"
        "- Density: **minimised**\n"
        "- Aluminum Content: maximised"
    )

    run_btn = st.button("🚀 Start Optimisation", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
if not selected_objectives:
    st.warning("Please select at least one optimisation objective in the sidebar.")
    st.stop()

if run_btn:
    if len(selected_objectives) < 1:
        st.error("Select at least one objective.")
        st.stop()

    progress = st.progress(0, text="Setting up optimisation...")

    problem = AlloyOptimizationProblem(
        selected_objectives, generator, regressors, classifiers, comp_min, comp_max
    )
    algorithm   = NSGA2(pop_size=pop_size, mutation=PM(prob=0.1, eta=20))
    termination = get_termination("n_gen", n_gen)

    progress.progress(10, text="Running NSGA-II optimisation…")

    res = minimize(
        problem, algorithm, termination,
        save_history=False, seed=int(seed_val), verbose=False
    )

    progress.progress(80, text="Decoding optimal alloys…")

    result_df, props = decode_alloys(
        res.X, generator, comp_min, comp_max,
        regressors, classifiers, process_names
    )

    st.session_state["result_df"] = result_df
    st.session_state["props"]     = props

    progress.progress(100, text="Done!")
    progress.empty()

# ── Display results ───────────────────────────────────────────────────────────
if "result_df" in st.session_state and st.session_state["result_df"] is not None:
    result_df = st.session_state["result_df"]
    props     = st.session_state["props"]

    st.success(f"✅ Optimisation complete — {len(result_df)} alloy candidates generated.")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tensile Strength vs Elongation")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(props["Tensile Strength"], props["Elongation"],
                    c="#1f77b4", alpha=0.7, edgecolors="white", s=60)
        ax1.set_xlabel("Tensile Strength (MPa)")
        ax1.set_ylabel("Elongation (%)")
        ax1.set_title("Tensile Strength vs Elongation")
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)

    with col2:
        st.subheader("Yield Strength vs Elongation")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(props["Yield Strength"], props["Elongation"],
                    c="#ff7f0e", alpha=0.7, edgecolors="white", s=60)
        ax2.set_xlabel("Yield Strength (MPa)")
        ax2.set_ylabel("Elongation (%)")
        ax2.set_title("Yield Strength vs Elongation")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)

    st.divider()

    # Results table
    st.subheader("📋 Generated Alloy Candidates")
    st.dataframe(result_df, use_container_width=True)

    # Download button
    buffer = io.BytesIO()
    result_df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    st.download_button(
        label="⬇️ Download Results as Excel",
        data=buffer,
        file_name="MPEA_optimised_alloys.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

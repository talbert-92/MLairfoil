import streamlit as st
import numpy as np
import joblib

# 1) Load models & scaler (use st.cache_resource for objects like models)
@st.cache_resource
def load_pipeline():
    scaler      = joblib.load('full_scaler.pkl')
    multi_clcm  = joblib.load('multi_clcm_rf.pkl')
    cd_model    = joblib.load('cd_model_rf.pkl')
    return scaler, multi_clcm, cd_model

scaler, multi_clcm, cd_model = load_pipeline()

st.title("Airfoil Aerodynamic Coefficient Predictor")
st.markdown("Select your inputs below and click **Predict**.")

# 2) Sidebar inputs
st.sidebar.header("Input Parameters")
alpha        = st.sidebar.slider("Angle of attack (α) [deg]", -10.0, 20.0, 5.0, 0.1)
Re           = st.sidebar.number_input("Reynolds number", min_value=1e4, max_value=1e7, value=2e5, step=1e4, format="%.0f")
max_thick    = st.sidebar.slider("Max thickness (t/c)", 0.02, 0.2, 0.08, 0.005)
x_thick      = st.sidebar.slider("Location of max thickness (x/c)", 0.0, 1.0, 0.3, 0.01)
max_camb     = st.sidebar.slider("Max camber (m/c)", 0.0, 0.1, 0.03, 0.005)
x_camb       = st.sidebar.slider("Location of max camber (x/c)", 0.0, 1.0, 0.4, 0.01)
le_radius    = st.sidebar.slider("Leading-edge radius (c)", 0.005, 0.05, 0.02, 0.001)

# 3) Package inputs & engineer features
features = np.array([[alpha, Re, max_thick, x_thick, max_camb, x_camb, le_radius]])
# physics features
alpha_sq      = alpha**2
inv_Re        = 1.0 / Re
thick_over_Re = max_thick / Re
X_raw = np.hstack([features, [[alpha_sq, inv_Re, thick_over_Re]]])

# 4) Scale and predict
X_scaled = scaler.transform(X_raw)
cl, cm   = multi_clcm.predict(X_scaled)[0]
cd       = cd_model.predict(X_scaled)[0]

# 5) Display
st.header("Predicted Coefficients")
col1, col2, col3 = st.columns(3)
col1.metric("CL", f"{cl:.4f}")
col2.metric("CD", f"{cd:.5f}")
col3.metric("CM", f"{cm:.4f}")

# 6) Optional: visualization
st.bar_chart({"Coefficient": ["CL","CD","CM"], "Value": [cl, cd, cm]})
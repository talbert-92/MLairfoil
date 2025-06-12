import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Connect to SQLite database
DB_PATH = 'airfoil_polars.db'
if not os.path.exists(DB_PATH):
    st.error(f"Database file not found at {DB_PATH}. Please run the data saving script first.")
    st.stop()

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# 2. Sidebar selectors with cascading logic
st.sidebar.title("Filter Settings")

# First, select the airfoil family
families = pd.read_sql("SELECT DISTINCT airfoilFamily FROM polars ORDER BY airfoilFamily", conn)["airfoilFamily"].tolist()
selected_family = st.sidebar.selectbox("Select Airfoil Family", families)

# Then, populate the airfoil list based on the selected family
airfoils = pd.read_sql(
    "SELECT DISTINCT airfoil FROM polars WHERE airfoilFamily = ? ORDER BY airfoil",
    conn,
    params=(selected_family,)
)["airfoil"].tolist()
selected_airfoil = st.sidebar.selectbox("Select Airfoil", airfoils)

# Populate Reynolds number list based on selected family and airfoil
res_list = pd.read_sql(
    "SELECT DISTINCT Re FROM polars WHERE airfoilFamily = ? AND airfoil = ? ORDER BY Re",
    conn,
    params=(selected_family, selected_airfoil)
)["Re"].tolist()
selected_Re = st.sidebar.selectbox("Select Reynolds Number", res_list)

# Slider limits based on the full selection
query_params_alpha = (selected_family, selected_airfoil, selected_Re)
min_alpha_df = pd.read_sql("SELECT MIN(alpha) FROM polars WHERE airfoilFamily=? AND airfoil=? AND Re=?", conn, params=query_params_alpha)
max_alpha_df = pd.read_sql("SELECT MAX(alpha) FROM polars WHERE airfoilFamily=? AND airfoil=? AND Re=?", conn, params=query_params_alpha)

if min_alpha_df.empty or max_alpha_df.empty or min_alpha_df.iloc[0,0] is None:
    st.sidebar.warning("No data for selection.")
    st.stop()

min_alpha = float(min_alpha_df['MIN(alpha)'][0])
max_alpha = float(max_alpha_df['MAX(alpha)'][0])

alpha_min, alpha_max = st.sidebar.slider(
    "Angle of Attack Range (°)",
    min_value=min_alpha,
    max_value=max_alpha,
    value=(max(min_alpha, 0.0), min(max_alpha, 10.0)),
    step=0.5
)

# 3. Query filtered data for polar analysis
query = """
SELECT alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
  FROM polars
 WHERE airfoilFamily = ?
   AND airfoil = ?
   AND Re = ?
   AND alpha BETWEEN ? AND ?
 ORDER BY alpha
"""
params = (selected_family, selected_airfoil, selected_Re, alpha_min, alpha_max)
df = pd.read_sql(query, conn, params=params)

# 4. Query geometric features for the selected airfoil
geo_query = """
SELECT DISTINCT max_thickness, x_thickness, max_camber, x_camber, le_radius
  FROM polars
 WHERE airfoilFamily = ?
   AND airfoil = ?
   AND max_thickness IS NOT NULL
 LIMIT 1
"""
geo_df = pd.read_sql(geo_query, conn, params=(selected_family, selected_airfoil))

# 5. Display results
st.title(f"Polar Data: {selected_family} {selected_airfoil.upper()} at Re={int(selected_Re):,}")

# Display geometric features table
if not geo_df.empty and not geo_df.isna().all().all():
    st.subheader("Airfoil Geometric Features")
    geo_data = {
        'Parameter': ['Maximum Thickness (t/c)', 'Thickness Location (x/c)', 'Maximum Camber (f/c)', 'Camber Location (x/c)', 'Leading Edge Radius'],
        'Value': [
            f"{geo_df['max_thickness'].iloc[0]:.4f}", f"{geo_df['x_thickness'].iloc[0]:.4f}",
            f"{geo_df['max_camber'].iloc[0]:.4f}", f"{geo_df['x_camber'].iloc[0]:.4f}", f"{geo_df['le_radius'].iloc[0]:.4f}"
        ]
    }
    st.table(pd.DataFrame(geo_data))
else:
    st.info("No geometric features available for this airfoil.")

if df.empty:
    st.warning("No polar data available for the selected filters.")
else:
    st.subheader("Polar Performance Data")
    st.dataframe(df.set_index('alpha'))

    st.subheader("Coefficients vs Angle of Attack")
    st.line_chart(df.set_index("alpha")[['CL', 'CD', 'CM']])

    st.subheader("Boundary Layer Transition Points")
    st.line_chart(df.set_index('alpha')[['Top_Xtr', 'Bot_Xtr']])

# 6. Load and plot airfoil shape
def load_airfoil_coordinates(family_name, airfoil_name, coords_folder='generated_airfoils'):
    # Construct the correct filename, e.g., "NACA4_NACA2412.dat"
    composite_name = f'{family_name}_{airfoil_name}'
    file_path = os.path.join(coords_folder, f'{composite_name}.dat')
    if not os.path.exists(file_path):
        return None
    try:
        coords = pd.read_csv(file_path, sep=r'\s+', skiprows=1, names=['x', 'y'], on_bad_lines='skip')
        # Ensure data is numeric
        coords['x'] = pd.to_numeric(coords['x'], errors='coerce')
        coords['y'] = pd.to_numeric(coords['y'], errors='coerce')
        return coords.dropna()
    except Exception:
        return None

coords_df = load_airfoil_coordinates(selected_family, selected_airfoil)

if coords_df is not None:
    st.subheader(f"Airfoil Shape: {selected_family} {selected_airfoil}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(coords_df['x'], coords_df['y'], 'b-')
    ax.fill(coords_df['x'], coords_df['y'], 'b', alpha=0.1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    st.pyplot(fig)
else:
    st.warning(f"No geometry file found for {selected_family}_{selected_airfoil}.dat")

# 7. Summary statistics
if not df.empty:
    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)

    max_cl = df.loc[df['CL'].idxmax()]
    min_cd = df.loc[df['CD'].idxmin()]
    
    df_ld = df[df['CD'] > 0.0001].copy() # Avoid division by zero/small numbers
    if not df_ld.empty:
        df_ld['L/D'] = df_ld['CL'] / df_ld['CD']
        max_ld = df_ld.loc[df_ld['L/D'].idxmax()]
    else:
        max_ld = None

    zero_lift_alpha = df.iloc[(df['CL'] - 0).abs().argmin()]['alpha'] if not df.empty else 0

    col1.metric("Max CL", f"{max_cl['CL']:.3f}", f"at α={max_cl['alpha']:.1f}°")
    col2.metric("Min CD", f"{min_cd['CD']:.4f}", f"at α={min_cd['alpha']:.1f}°")
    if max_ld is not None:
        col3.metric("Max L/D", f"{max_ld['L/D']:.1f}", f"at α={max_ld['alpha']:.1f}°")
    col4.metric("Zero-lift α", f"{zero_lift_alpha:.2f}°")

# 8. Close database connection
conn.close()
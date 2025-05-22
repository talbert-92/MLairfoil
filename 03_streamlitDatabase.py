import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt  # ❗ You forgot to import this
import os  # ❗ Also missing for `os.path.join`

# 1. Connect to SQLite database
DB_PATH = 'airfoil_polars.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# 2. Sidebar selectors
airfoils = pd.read_sql("SELECT DISTINCT airfoil FROM polars ORDER BY airfoil", conn)["airfoil"].tolist()
res_list = pd.read_sql("SELECT DISTINCT Re FROM polars ORDER BY Re", conn)["Re"].tolist()

st.sidebar.title("Filter Settings")
selected_airfoil = st.sidebar.selectbox("Select Airfoil", airfoils)
selected_Re = st.sidebar.selectbox("Select Reynolds Number", res_list)

# Slider limits must be cast carefully (and inside a safe range)
min_alpha = float(pd.read_sql(
    "SELECT MIN(alpha) FROM polars WHERE airfoil=? AND Re=?", conn,
    params=(selected_airfoil, selected_Re))['MIN(alpha)'][0])

max_alpha = float(pd.read_sql(
    "SELECT MAX(alpha) FROM polars WHERE airfoil=? AND Re=?", conn,
    params=(selected_airfoil, selected_Re))['MAX(alpha)'][0])

alpha_min, alpha_max = st.sidebar.slider(
    "Angle of Attack Range (°)",
    min_value=min_alpha,
    max_value=max_alpha,
    value=(max(min_alpha, 0.0), min(max_alpha, 10.0)),  # ensure default range is within limits
    step=0.5
)

# 3. Query filtered data
query = """
SELECT alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
  FROM polars
 WHERE airfoil = ?
   AND Re = ?
   AND alpha BETWEEN ? AND ?
 ORDER BY alpha
"""
params = (selected_airfoil, selected_Re, alpha_min, alpha_max)
df = pd.read_sql(query, conn, params=params)

# 4. Display results
st.title(f"Polar Data: {selected_airfoil.upper()} at Re={int(selected_Re):,}")

if df.empty:
    st.warning("No data available for the selected filters.")
else:
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Coefficients vs Angle of Attack")
    chart_df = df.set_index("alpha")[['CL', 'CD', 'CDp', 'CM']]
    st.line_chart(chart_df)

    st.subheader("Boundary Layer Transition Points")
    st.line_chart(df.set_index('alpha')[['Top_Xtr', 'Bot_Xtr']])

# 5. Load and plot airfoil shape
def load_airfoil_coordinates(airfoil_name, coords_folder='airfoil_profiles'):
    file_path = os.path.join(coords_folder, f'{airfoil_name}.dat')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coords = [line.strip().split() for line in lines[1:] if len(line.strip().split()) == 2]
    x, y = zip(*[(float(a), float(b)) for a, b in coords])
    return pd.DataFrame({'x': x, 'y': y})

coords_df = load_airfoil_coordinates(selected_airfoil)  # ❗ Variable name was wrong

if coords_df is not None:
    st.subheader(f"Airfoil Shape: {selected_airfoil}")
    fig, ax = plt.subplots()
    ax.plot(coords_df['x'], coords_df['y'], marker='o', markersize=2)
    ax.set_aspect('equal')
    ax.set_title(selected_airfoil)
    st.pyplot(fig)
else:
    st.warning(f"No geometry file found for {selected_airfoil}.")

# 6. Close database connection
conn.close()
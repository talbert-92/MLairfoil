import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# 3. Query filtered data for polar analysis
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

# 4. Query geometric features for the selected airfoil
geo_query = """
SELECT DISTINCT max_thickness, x_thickness, max_camber, x_camber, le_radius
  FROM polars
 WHERE airfoil = ?
   AND max_thickness IS NOT NULL
 LIMIT 1
"""
geo_df = pd.read_sql(geo_query, conn, params=(selected_airfoil,))

# 5. Display results
st.title(f"Polar Data: {selected_airfoil.upper()} at Re={int(selected_Re):,}")

# Display geometric features table
if not geo_df.empty and not geo_df.isna().all().all():
    st.subheader("Airfoil Geometric Features")
    
    # Create a nicely formatted table
    geo_data = {
        'Parameter': [
            'Maximum Thickness (t/c)',
            'Thickness Location (x/c)',
            'Maximum Camber (f/c)',
            'Camber Location (x/c)',
            'Leading Edge Radius'
        ],
        'Value': [
            f"{geo_df['max_thickness'].iloc[0]:.4f}" if pd.notna(geo_df['max_thickness'].iloc[0]) else "N/A",
            f"{geo_df['x_thickness'].iloc[0]:.4f}" if pd.notna(geo_df['x_thickness'].iloc[0]) else "N/A",
            f"{geo_df['max_camber'].iloc[0]:.4f}" if pd.notna(geo_df['max_camber'].iloc[0]) else "N/A",
            f"{geo_df['x_camber'].iloc[0]:.4f}" if pd.notna(geo_df['x_camber'].iloc[0]) else "N/A",
            f"{geo_df['le_radius'].iloc[0]:.4f}" if pd.notna(geo_df['le_radius'].iloc[0]) else "N/A"
        ],
        'Description': [
            'Ratio of maximum thickness to chord length',
            'Chordwise position of maximum thickness',
            'Maximum deviation from chord line',
            'Chordwise position of maximum camber',
            'Curvature radius at leading edge'
        ]
    }
    
    geo_table = pd.DataFrame(geo_data)
    st.table(geo_table)
else:
    st.info("No geometric features available for this airfoil.")

if df.empty:
    st.warning("No polar data available for the selected filters.")
else:
    st.subheader("Polar Performance Data")
    st.dataframe(df)

    st.subheader("Coefficients vs Angle of Attack")
    chart_df = df.set_index("alpha")[['CL', 'CD', 'CDp', 'CM']]
    st.line_chart(chart_df)

    st.subheader("Boundary Layer Transition Points")
    st.line_chart(df.set_index('alpha')[['Top_Xtr', 'Bot_Xtr']])

# 6. Load and plot airfoil shape
def load_airfoil_coordinates(airfoil_name, coords_folder='generated_airfoils'):
    file_path = os.path.join(coords_folder, f'{airfoil_name}.dat')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coords = [line.strip().split() for line in lines[1:] if len(line.strip().split()) == 2]
    try:
        x, y = zip(*[(float(a), float(b)) for a, b in coords])
        return pd.DataFrame({'x': x, 'y': y})
    except:
        return None

coords_df = load_airfoil_coordinates(selected_airfoil)

if coords_df is not None:
    st.subheader(f"Airfoil Shape: {selected_airfoil}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(coords_df['x'], coords_df['y'], 'b-', linewidth=2)
    ax.fill_between(coords_df['x'], coords_df['y'], alpha=0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(f'{selected_airfoil} Airfoil Profile')
    st.pyplot(fig)
    
    # Add some airfoil statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chord Length", "1.000", help="Normalized to unit chord")
    with col2:
        if not geo_df.empty:
            thickness_pct = geo_df['max_thickness'].iloc[0] * 100 if pd.notna(geo_df['max_thickness'].iloc[0]) else 0
            st.metric("Max Thickness", f"{thickness_pct:.1f}%", help="As percentage of chord")
    with col3:
        if not geo_df.empty:
            camber_pct = geo_df['max_camber'].iloc[0] * 100 if pd.notna(geo_df['max_camber'].iloc[0]) else 0
            st.metric("Max Camber", f"{camber_pct:.1f}%", help="As percentage of chord")
else:
    st.warning(f"No geometry file found for {selected_airfoil}.")

# 7. Summary statistics
if not df.empty:
    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_cl = df['CL'].max()
        st.metric("Max CL", f"{max_cl:.3f}")
    
    with col2:
        min_cd = df['CD'].min()
        st.metric("Min CD", f"{min_cd:.4f}")
    
    with col3:
        # L/D ratio
        df_ld = df[df['CD'] > 0]  # Avoid division by zero
        if not df_ld.empty:
            max_ld = (df_ld['CL'] / df_ld['CD']).max()
            st.metric("Max L/D", f"{max_ld:.1f}")
    
    with col4:
        # Zero-lift angle
        zero_lift_alpha = df.loc[df['CL'].abs().idxmin(), 'alpha']
        st.metric("Zero-lift α", f"{zero_lift_alpha:.1f}°")

# 8. Close database connection
conn.close()
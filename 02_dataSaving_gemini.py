import os
import glob
import sqlite3
import pandas as pd
import numpy as np

def extract_geo_features_robust(dat_path):
    """
    Extracts geometric features from an airfoil .dat file using robust methods
    based on established aerodynamic theory.
    
    This function correctly:
    1. Identifies upper and lower surfaces regardless of file format.
    2. Sorts coordinates before interpolation to ensure correctness.
    3. Calculates max camber while preserving its sign, which is critical for ML.

    Args:
        dat_path (str): Path to the airfoil .dat file
        
    Returns:
        dict: Dictionary containing geometric features
    """
    try:
        # 1. Read airfoil coordinates, skipping header
        with open(dat_path, 'r') as f:
            lines = f.readlines()
        
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue
        
        if len(coords) < 10: # Basic check for a valid airfoil
            raise ValueError("Not enough valid coordinates found")
        
        coords = np.array(coords)
        airfoil_name = os.path.splitext(os.path.basename(dat_path))[0]
        
        # 2. Robustly separate upper and lower surfaces
        # Find the leading edge (point with minimum x-coordinate)
        le_idx = np.argmin(coords[:, 0])
        
        # Split coordinates into two segments at the leading edge
        segment1 = coords[:le_idx+1]
        segment2 = coords[le_idx:]

        # Identify upper/lower surfaces by their mean y-value. This is robust.
        # The surface with a higher average y-value is the upper surface.
        if np.mean(segment1[1:, 1]) > np.mean(segment2[1:, 1]):
            upper_coords = segment1
            lower_coords = segment2
        else:
            upper_coords = segment2
            lower_coords = segment1
            
        # 3. Correctly interpolate surfaces onto a common x-grid
        # CRITICAL FIX: Sort coordinates by x-value before interpolating.
        # np.interp requires the 'xp' array (x-coordinates) to be monotonic.
        upper_coords_sorted = upper_coords[np.argsort(upper_coords[:, 0])]
        lower_coords_sorted = lower_coords[np.argsort(lower_coords[:, 0])]

        # Create a common, high-resolution x-coordinate basis
        x_common = np.linspace(0, 1.0, 201)
        
        upper_interp = np.interp(x_common, upper_coords_sorted[:, 0], upper_coords_sorted[:, 1])
        lower_interp = np.interp(x_common, lower_coords_sorted[:, 0], lower_coords_sorted[:, 1])
        
        # 4. Calculate geometric features based on established theory
        
        # Thickness distribution t(x) = y_upper(x) - y_lower(x)
        thickness = upper_interp - lower_interp
        max_thickness = np.max(thickness)
        x_thickness = x_common[np.argmax(thickness)]
        
        # Mean camber line c(x) = (y_upper(x) + y_lower(x)) / 2
        camber_line = (upper_interp + lower_interp) / 2
        
        # CRITICAL FIX: Find max camber while preserving its sign
        max_camber_idx = np.argmax(np.abs(camber_line))
        max_camber = camber_line[max_camber_idx] # Get the actual value, not its absolute
        x_camber = x_common[max_camber_idx]
        
        # Leading Edge Radius
        le_radius = estimate_leading_edge_radius(coords, le_idx)
        
        return {
            'airfoil': airfoil_name,
            'max_thickness': round(max_thickness, 5),
            'x_thickness': round(x_thickness, 4),
            'max_camber': round(max_camber, 5),
            'x_camber': round(x_camber, 4),
            'le_radius': round(le_radius, 6)
        }
        
    except Exception as e:
        print(f"Error processing {dat_path}: {e}")
        return {
            'airfoil': os.path.splitext(os.path.basename(dat_path))[0],
            'max_thickness': np.nan,
            'x_thickness': np.nan,
            'max_camber': np.nan,
            'x_camber': np.nan,
            'le_radius': np.nan
        }

def estimate_leading_edge_radius(coords, le_idx):
    """
    Estimate leading edge radius by fitting a circle to points near the LE.
    This version uses a more standard least-squares circle fitting method.
    Args:
        coords (np.array): Airfoil coordinates
        le_idx (int): Index of leading edge point
    Returns:
        float: Estimated leading edge radius
    """
    try:
        # Select a few points around the leading edge for a robust fit
        num_points_side = min(5, le_idx, len(coords) - le_idx - 1)
        if num_points_side < 2: return np.nan
        
        local_coords = coords[le_idx - num_points_side : le_idx + num_points_side + 1]
        x = local_coords[:, 0]
        y = local_coords[:, 1]
        
        # Fit a circle x^2 + y^2 + ax + by + c = 0
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x**2 + y**2)
        
        # Solve for the circle parameters [a, b, c]
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Radius is sqrt((a/2)^2 + (b/2)^2 - c)
        radius = np.sqrt((params[0]**2 + params[1]**2)/4 - params[2])
        return radius if np.isreal(radius) else np.nan
            
    except Exception:
        return np.nan

# --- SCRIPT MAINTAINED FROM THIS POINT ONWARDS AS REQUESTED ---

# Configuration
airfoil_folder = "generated_airfoils"  # folder containing airfoil .dat files
polar_folder = "polar_outputs"       # folder containing polar .txt files
output_folder = "data_EDA"          # folder for parquet/csv outputs
db_path = "airfoil_polars.db"       # SQLite database path

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# 1. Locate all polar files
polar_files = glob.glob(os.path.join(polar_folder, "*.txt"))
if not polar_files:
    raise FileNotFoundError(f"No polar files found in {polar_folder}")
print(f"Found {len(polar_files)} polar files.")

# 2. Process each file once and build a master DataFrame
df_list = []
for fn in polar_files:
    basename = os.path.basename(fn)
    # From your file examples, we see patterns like:
    # polar_apex16_nasa_cr-201062_Re300000.txt
    # polar_nasasc2-0714_Re300000.txt
    
    # Split by underscore
    parts = basename.rstrip(".txt").split("_")
    if len(parts) < 3:  # Need at least "polar", airfoil name, and something with "Re"
        print(f"Skipping file with too few parts: {basename}")
        continue
    
    # Take the second part as the beginning of the airfoil name
    airfoil_parts = []
    airfoil_parts.append(parts[1])
    
    # Find the part containing "Re" to mark the end of airfoil name
    re_part = None
    for i, part in enumerate(parts[2:], 2):
        if "Re" in part:
            re_part = part
            break
        else:
            # Keep adding to airfoil name until we find the Re part
            airfoil_parts.append(part)
    
    # Combine airfoil parts to get the full airfoil name
    airfoil = "_".join(airfoil_parts)
    
    # Extract Reynolds number
    if re_part is None:
        print(f"Warning: No Reynolds number found in filename: {basename}")
        print(f"Using default Reynolds number for {basename}")
        re_value = 0  # Default value
    else:
        try:
            # Extract just the numeric part after "Re"
            re_str = re_part.split("Re")[1]
            re_value = float(re_str)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse Reynolds number from {re_part} in {basename}")
            print(f"Using default Reynolds number for {basename}")
            re_value = 0
    
    try:
        # Load the data table, skipping XFoil header
        temp_df = pd.read_csv(
            fn,
            sep=r'\s+',
            skiprows=12,
            names=["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
        )
        
        # Add metadata columns
        temp_df["airfoil"] = airfoil
        temp_df["Re"] = re_value
        
        # Only add if dataframe has actual data
        if not temp_df.empty and not temp_df.isna().all().all():
            df_list.append(temp_df)
        else:
            print(f"Warning: No valid data found in {basename}")
            
    except Exception as e:
        print(f"Error processing {basename}: {e}")

# Combine all dataframes
if not df_list:
    raise ValueError("No valid data found in any polar file")
    
master_df = pd.concat(df_list, ignore_index=True)
print(f"Master DataFrame created with {master_df.shape[0]} rows and {master_df.shape[1]} columns")

# 3. Extract geometry features from .dat files
print("Extracting geometry features from .dat files...")
airfoils_used = master_df['airfoil'].unique().tolist()
print(f"Found {len(airfoils_used)} unique airfoils in polar data")

geo_feats = []
for foil in airfoils_used:
    # Construct path to .dat file
    dat_path = os.path.join(airfoil_folder, f"{foil}.dat")
    
    if not os.path.isfile(dat_path):
        print(f"  • Skipping geometry for '{foil}': .dat file not found at {dat_path}")
        # Add empty geometry data
        geo_feats.append({
            'airfoil': foil,
            'max_thickness': np.nan,
            'x_thickness': np.nan,
            'max_camber': np.nan,
            'x_camber': np.nan,
            'le_radius': np.nan
        })
        continue
    
    try:
        # *** CALLING THE CORRECTED, ROBUST FUNCTION ***
        feat = extract_geo_features_robust(dat_path)
        geo_feats.append(feat)
        print(f"  • Extracted geometry for '{foil}'")
    except Exception as e:
        print(f"  • Error parsing '{dat_path}': {e}")
        # Add empty geometry data
        geo_feats.append({
            'airfoil': foil,
            'max_thickness': np.nan,
            'x_thickness': np.nan,
            'max_camber': np.nan,
            'x_camber': np.nan,
            'le_radius': np.nan
        })

# Create a DataFrame of geometry features
geo_df = pd.DataFrame(geo_feats).set_index('airfoil')
print(f"Extracted geometry for {len(geo_df)} airfoils.")

# 4. Merge geometry features into master polar DataFrame
print("Merging geometry features with polar data...")
master_df = master_df.merge(
    geo_df,
    how='left',
    left_on='airfoil',
    right_index=True
)
print(f"Enhanced DataFrame created with {master_df.shape[0]} rows and {master_df.shape[1]} columns")

# 5. Create SQLite database
print(f"Creating SQLite database at {db_path}...")
conn = sqlite3.connect(db_path)
try:
    # Drop existing table if it exists
    with conn:
        conn.execute("DROP TABLE IF EXISTS polars;")
        conn.execute("""
            CREATE TABLE polars (
                airfoil TEXT,
                Re REAL,
                alpha REAL,
                CL REAL,
                CD REAL,
                CDp REAL,
                CM REAL,
                Top_Xtr REAL,
                Bot_Xtr REAL,
                max_thickness REAL,
                x_thickness REAL,
                max_camber REAL,
                x_camber REAL,
                le_radius REAL
            );
        """)
        conn.execute("CREATE INDEX idx_airfoil_Re ON polars(airfoil, Re);")
        conn.execute("CREATE INDEX idx_geometry ON polars(max_thickness, max_camber);")
    
    # Insert data from master DataFrame
    master_df.to_sql('polars', conn, if_exists='append', index=False, 
                    method='multi', chunksize=1000)
    
    print(f"SQLite database created successfully at {db_path}")
    
    # Print some statistics
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM polars")
    total_rows = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT airfoil) FROM polars")
    unique_airfoils = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM polars WHERE max_thickness IS NOT NULL")
    geo_rows = cursor.fetchone()[0]
    
    print(f"Database contains:")
    print(f"  • {total_rows:,} total data points")
    print(f"  • {unique_airfoils} unique airfoils")
    print(f"  • {geo_rows:,} rows with geometry data ({geo_rows/total_rows*100:.1f}%)")
    
except Exception as e:
    print(f"Error creating SQLite database: {e}")
finally:
    conn.close()

# 6. Save to Parquet and CSV
parquet_path = os.path.join(output_folder, "airfoil_polars.parquet")
csv_path = os.path.join(output_folder, "airfoil_polars.csv")

try:
    master_df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet file to: {parquet_path}")
except Exception as e:
    print(f"Error saving Parquet file: {e}")

try:
    master_df.to_csv(csv_path, index=False)
    print(f"Saved CSV file to: {csv_path}")
except Exception as e:
    print(f"Error saving CSV file: {e}")

print("\nProcessing completed successfully!")
print(f"Final dataset columns: {list(master_df.columns)}")
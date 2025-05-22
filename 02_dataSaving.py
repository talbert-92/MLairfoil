"""
Unified script to process XFoil polar text files and output to multiple formats:
- SQLite database
- Parquet file
- CSV file

Usage:
    python consolidated_airfoil_processing.py

Processes files with naming pattern: polar_<airfoil>_Re<value>.txt
"""
import os
import glob
import sqlite3
import pandas as pd

# Configuration
polar_folder = "polar_outputs"  # folder containing polar .txt files
output_folder = "data_EDA"      # folder for parquet/csv outputs
db_path = "airfoil_polars.db"   # SQLite database path

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

# 3. Create SQLite database
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
                Bot_Xtr REAL
            );
        """)
        conn.execute("CREATE INDEX idx_airfoil_Re ON polars(airfoil, Re);")
    
    # Insert data from master DataFrame
    # Use DataFrame's to_sql method for efficient bulk insertion
    master_df.to_sql('polars', conn, if_exists='append', index=False, 
                    method='multi', chunksize=1000)
    
    print(f"SQLite database created successfully at {db_path}")
except Exception as e:
    print(f"Error creating SQLite database: {e}")
finally:
    conn.close()

# 4. Save to Parquet and CSV
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

print("Processing completed successfully!")
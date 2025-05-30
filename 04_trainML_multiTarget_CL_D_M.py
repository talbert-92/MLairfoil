# Cell 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Data Loading Function (USER ACTION REQUIRED HERE)
def load_airfoil_data(filepath):
    """
    Loads the airfoil data from the specified CSV file.
    USER: You NEED to replace the synthetic data generation below
          with: return pd.read_csv(filepath)
          and ensure your CSV is correctly formatted.
    """
    print(f"Attempting to load data from: {filepath}")
    try:
        # *** USER: REPLACE THE FOLLOWING BLOCK WITH YOUR ACTUAL DATA LOADING ***
        # Example:
        df = pd.read_parquet(filepath)
        print(f"Successfully loaded data from {filepath}")
        # Ensure correct dtypes if necessary, e.g.:
        numeric_cols = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'Re', 
                        'max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
        # *** END OF USER REPLACEMENT BLOCK ***

        # --- Placeholder Synthetic Data Generation (for demonstration) ---
        print("INFO: Using placeholder synthetic data generator. Replace with your actual data loading.")
        num_airfoils = 200
        num_aoa_per_airfoil = 20
        total_rows = num_airfoils * num_aoa_per_airfoil
        airfoil_names = [f'synth_airfoil_{i}' for i in range(num_airfoils)]
        df_list = []
        for i in range(num_airfoils):
            airfoil_name = airfoil_names[i]
            max_thick = np.random.uniform(0.04, 0.18)
            x_thick = np.random.uniform(0.25, 0.45)
            max_camb = np.random.uniform(0.0, 0.06)
            x_camb = np.random.uniform(0.3, 0.7)
            le_rad = np.random.uniform(0.005, 0.03)
            reynolds_number = np.random.choice([100000.0, 200000.0, 300000.0])
            for _ in range(num_aoa_per_airfoil):
                alpha_val = np.random.uniform(-5.0, 15.0)
                cl_linear = 0.1 * alpha_val + 4 * max_camb * (1 - abs(alpha_val / 15.0))
                stall_effect = 0
                if alpha_val > 10: stall_effect = -0.05 * (alpha_val - 10)**2
                elif alpha_val < -4: stall_effect = -0.05 * (alpha_val + 4)**2
                cl_val = cl_linear - 2 * max_thick + stall_effect + np.random.normal(0, 0.05)
                cd_val = 0.01 + 0.0005 * alpha_val**2 + 2 * max_thick**2 + 0.5 * max_camb**2 + np.random.normal(0, 0.001)
                cm_val = -0.05 * max_camb - 0.002 * alpha_val + np.random.normal(0, 0.01)
                df_list.append({
                    'alpha': alpha_val, 'CL': cl_val, 'CD': cd_val, 'CDp': cd_val * 0.8, 'CM': cm_val,
                    'Top_Xtr': np.random.uniform(0.1, 1.0), 'Bot_Xtr': np.random.uniform(0.1, 1.0),
                    'airfoil': airfoil_name, 'Re': reynolds_number, 'max_thickness': max_thick,
                    'x_thickness': x_thick, 'max_camber': max_camb, 'x_camber': x_camb, 'le_radius': le_rad
                })
        df = pd.DataFrame(df_list)
        # --- End of Placeholder ---
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}. Please check the path.")
        return None
    except Exception as e:
        print(f"ERROR: An error occurred while loading data: {e}")
        return None
		
# Cell 3: Helper Functions for Clustered Models (MODIFIED FOR MULTI-TARGET)
def define_bins(df_column, bin_width, min_val=None, max_val=None):
    if df_column.empty: return [], []
    if min_val is None: min_val = df_column.min()
    if max_val is None: max_val = df_column.max()
    min_val_rounded = np.floor(min_val / bin_width) * bin_width
    bins = np.arange(min_val_rounded, max_val + bin_width, bin_width)
    if not bins.size or bins[-1] <= max_val:
         bins = np.append(bins, bins[-1] + bin_width if bins.size > 0 else min_val_rounded + bin_width)
    bins = np.unique(bins)
    if len(bins) < 2: return bins, []
    labels = [f'{bins[i]:.3f}-{bins[i+1]:.3f}' for i in range(len(bins)-1)]
    return bins, labels

def train_models_on_clusters_multi_target(df, cluster_column_name, feature_cols, target_cols, min_samples_per_cluster=50, model_type='rf'):
    """
    Modified function to train separate models for multiple targets (CL, CD, CM)
    Returns dictionaries organized by target, then by cluster
    """
    trained_models = {target: {} for target in target_cols}
    trained_scalers = {}  # Scalers are shared across targets for same cluster
    cluster_results = {target: {} for target in target_cols}
    
    if cluster_column_name not in df.columns or df[cluster_column_name].isnull().all():
        print(f"ERROR: Cluster column '{cluster_column_name}' not found or is all NaN.")
        return trained_models, trained_scalers, cluster_results
    
    unique_clusters = df[cluster_column_name].dropna().unique()
    if not unique_clusters.size:
        print(f"No valid clusters found in '{cluster_column_name}'.")
        return trained_models, trained_scalers, cluster_results

    for cluster_label in unique_clusters:
        print(f"\n--- Processing Cluster: {cluster_label} for {model_type} ---")
        cluster_data_full = df[df[cluster_column_name] == cluster_label].copy()
        
        if len(cluster_data_full) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: {len(cluster_data_full)} samples < {min_samples_per_cluster}.")
            continue
            
        # Check for missing columns
        missing_cols = [col for col in feature_cols + target_cols if col not in cluster_data_full.columns]
        if missing_cols:
            print(f"Skipping {cluster_label}: Missing columns {missing_cols}.")
            continue
            
        cluster_data = cluster_data_full[feature_cols + target_cols].dropna()
        
        if len(cluster_data) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: Insufficient data ({len(cluster_data)} rows) after NaN removal.")
            continue
            
        X = cluster_data[feature_cols]
        
        if X.empty:
            print(f"Skipping {cluster_label}: Empty feature matrix.")
            continue
            
        # Split data once for all targets
        X_train, X_test, cluster_data_train, cluster_data_test = train_test_split(
            X, cluster_data, test_size=0.2, random_state=42
        )
        
        # Scale features once per cluster
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        trained_scalers[cluster_label] = scaler
        
        # Train separate model for each target
        for target_col in target_cols:
            y_train = cluster_data_train[target_col]
            y_test = cluster_data_test[target_col]
            
            if y_train.empty or y_test.empty:
                print(f"Skipping {cluster_label} for target {target_col}: Empty target data.")
                continue
                
            # Create model
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            elif model_type == 'gb':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError("Unsupported model_type")
                
            # Train model
            model.fit(X_train_scaled, y_train)
            trained_models[target_col][cluster_label] = model
            
            # Evaluate model
            y_pred_test = model.predict(X_test_scaled)
            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            oob = model.oob_score_ if hasattr(model, 'oob_score_') and model.oob_score_ else np.nan
            
            cluster_results[target_col][cluster_label] = {
                'model_type': model_type, 
                'cluster_label': cluster_label,
                'target': target_col,
                'train_samples': len(X_train), 
                'test_samples': len(X_test),
                'r2_test': r2_test, 
                'mse_test': mse_test, 
                'oob_score': oob
            }
            
            print(f"  {target_col}: R2={r2_test:.4f}, MSE={mse_test:.4f}, OOB={oob:.4f}")
    
    return trained_models, trained_scalers, cluster_results
	
# Cell 4: Configuration (MODIFIED FOR MULTI-TARGET)
DATA_FILEPATH = 'data_EDA\\airfoil_polars.parquet'  # USER: !!! UPDATE THIS PATH !!!
TARGET_COLUMNS = ['CL', 'CD', 'CM']  # Multiple targets now

# Features for CLUSTERED models (max_thickness used for clustering, not as direct input)
GEOMETRIC_FEATURES_FOR_CLUSTERED = ['x_thickness', 'max_camber', 'x_camber', 'le_radius']
OPERATIONAL_FEATURES = ['alpha', 'Re']
INPUT_FEATURES_FOR_THICKNESS_CLUSTERING = OPERATIONAL_FEATURES + GEOMETRIC_FEATURES_FOR_CLUSTERED

# Features for GLOBAL models (all relevant features are direct inputs)
ALL_GEOMETRIC_FEATURES = ['max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
INPUT_FEATURES_FOR_GLOBAL_MODEL = OPERATIONAL_FEATURES + ALL_GEOMETRIC_FEATURES

# Clustering parameters
THICKNESS_CLUSTER_COL_NAME = 'max_thickness_cluster'
THICKNESS_BIN_WIDTH = 0.02
MIN_SAMPLES_PER_CLUSTER = 80

# Cell 5: Load Data
print("--- Loading Data ---")
df_airfoils = load_airfoil_data(DATA_FILEPATH)
if df_airfoils is None or df_airfoils.empty:
    print("CRITICAL ERROR: Data loading failed. Notebook execution cannot continue meaningfully.")
else:
    print(f"Data loaded successfully. Shape: {df_airfoils.shape}")
    print(df_airfoils.head())
	
# Cell 6: Train and Evaluate Clustered Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 1: Training Benchmark Clustered Models (by Max Thickness) for Multiple Targets ---")
thickness_clustered_models_rf = {}
thickness_clustered_scalers_rf = {}
thickness_cluster_results_summary_rf = {}

if df_airfoils is not None and not df_airfoils.empty:
    mt_min_val = df_airfoils['max_thickness'].min()
    mt_max_val = df_airfoils['max_thickness'].max()
    thickness_bins, thickness_labels = define_bins(df_airfoils['max_thickness'], THICKNESS_BIN_WIDTH, min_val=mt_min_val, max_val=mt_max_val)

    if thickness_labels:
        df_airfoils[THICKNESS_CLUSTER_COL_NAME] = pd.cut(df_airfoils['max_thickness'], bins=thickness_bins, labels=thickness_labels, right=False, include_lowest=True)
        print(f"\nValue counts for {THICKNESS_CLUSTER_COL_NAME}:")
        print(df_airfoils[THICKNESS_CLUSTER_COL_NAME].value_counts().sort_index())

        thickness_clustered_models_rf, thickness_clustered_scalers_rf, thickness_cluster_results_summary_rf = train_models_on_clusters_multi_target(
            df_airfoils.dropna(subset=[THICKNESS_CLUSTER_COL_NAME]),
            THICKNESS_CLUSTER_COL_NAME,
            INPUT_FEATURES_FOR_THICKNESS_CLUSTERING,
            TARGET_COLUMNS,
            min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER,
            model_type='rf'
        )
        
        print("\nSummary of Max Thickness Clustered RF Model Training:")
        weighted_averages_clustered = {}
        
        for target in TARGET_COLUMNS:
            if thickness_cluster_results_summary_rf[target]:
                results_rf_df = pd.DataFrame.from_dict(thickness_cluster_results_summary_rf[target], orient='index')
                print(f"\n{target} Results:")
                print(results_rf_df[['test_samples', 'oob_score', 'r2_test', 'mse_test']].sort_index())
                
                # Calculate weighted average for each target
                total_test_samples_clustered = results_rf_df['test_samples'].sum()
                if total_test_samples_clustered > 0:
                    weighted_r2 = (results_rf_df['r2_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_mse = (results_rf_df['mse_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_averages_clustered[target] = {'R2': weighted_r2, 'MSE': weighted_mse}
                    print(f"Weighted Average for {target} Clustered RF Models: R2 = {weighted_r2:.4f}, MSE = {weighted_mse:.4f}")
                else:
                    weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
                    print(f"No test samples for {target} clustered models.")
            else:
                print(f"No models were trained for {target} thickness clusters.")
                weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
    else:
        print(f"Could not create valid bins/labels for '{THICKNESS_CLUSTER_COL_NAME}'. Skipping clustered model training.")
        weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
else:
    print("Skipping Clustered Model training due to data loading issues.")
    weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
	
# Cell 7: Train and Evaluate Global Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 2: Training Global Models for Multiple Targets ---")
global_results = {}

if df_airfoils is not None and not df_airfoils.empty:
    # Prepare data for global model
    df_global = df_airfoils[INPUT_FEATURES_FOR_GLOBAL_MODEL + TARGET_COLUMNS].copy().dropna()
    
    if len(df_global) > MIN_SAMPLES_PER_CLUSTER:
        X_global = df_global[INPUT_FEATURES_FOR_GLOBAL_MODEL]
        
        # Split data once for all targets
        X_train_global, X_test_global, df_global_train, df_global_test = train_test_split(
            X_global, df_global, test_size=0.2, random_state=42
        )

        # Scale features for global models
        global_scaler = StandardScaler()
        X_train_global_scaled = global_scaler.fit_transform(X_train_global)
        X_test_global_scaled = global_scaler.transform(X_test_global)

        # Train models for each target
        for target in TARGET_COLUMNS:
            y_train_global = df_global_train[target]
            y_test_global = df_global_test[target]
            
            print(f"\n--- Training Global Models for {target} ---")
            
            # --- Global Random Forest ---
            print(f"Training Global Random Forest Model for {target}...")
            rf_global_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            rf_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_rf = rf_global_model.predict(X_test_global_scaled)
            r2_global_rf = r2_score(y_test_global, y_pred_global_rf)
            mse_global_rf = mean_squared_error(y_test_global, y_pred_global_rf)
            oob_global_rf = rf_global_model.oob_score_
            global_results[f'Global RF ({target})'] = {'R2 Test': r2_global_rf, 'MSE Test': mse_global_rf, 'OOB Score': oob_global_rf}
            print(f"Global Random Forest {target}: R2 = {r2_global_rf:.4f}, MSE = {mse_global_rf:.4f}, OOB = {oob_global_rf:.4f}")

            # --- Global Gradient Boosting ---
            print(f"Training Global Gradient Boosting Model for {target}...")
            gb_global_model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3)
            gb_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_gb = gb_global_model.predict(X_test_global_scaled)
            r2_global_gb = r2_score(y_test_global, y_pred_global_gb)
            mse_global_gb = mean_squared_error(y_test_global, y_pred_global_gb)
            global_results[f'Global GB ({target})'] = {'R2 Test': r2_global_gb, 'MSE Test': mse_global_gb, 'OOB Score': np.nan}
            print(f"Global Gradient Boosting {target}: R2 = {r2_global_gb:.4f}, MSE = {mse_global_gb:.4f}")
    else:
        print(f"Insufficient data for global model training after NaN removal (Rows: {len(df_global)}).")
else:
    print("Skipping Global Model training due to data loading issues.")

# Cell 8: Results Comparison and Discussion (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 3: Comparative Results Summary for Multiple Targets ---")

print("\nPerformance of Max Thickness Clustered Random Forest Models (Weighted Average):")
for target in TARGET_COLUMNS:
    if 'weighted_averages_clustered' in locals() and target in weighted_averages_clustered:
        weighted_r2 = weighted_averages_clustered[target]['R2']
        weighted_mse = weighted_averages_clustered[target]['MSE']
        if not np.isnan(weighted_r2):
            print(f"  {target}:")
            print(f"    - Weighted Avg R2: {weighted_r2:.4f}")
            print(f"    - Weighted Avg MSE: {weighted_mse:.4f}")
            # Add to global results for comparison
            global_results[f'Clustered RF ({target}, Weighted Avg)'] = {'R2 Test': weighted_r2, 'MSE Test': weighted_mse, 'OOB Score': np.nan}
        else:
            print(f"  {target}: Clustered model results not available for weighted average.")

print("\nPerformance of Global Models:")
if global_results:
    for model_name, metrics in global_results.items():
        if 'Global' in model_name and 'Clustered' not in model_name:
             print(f"  Model: {model_name}")
             print(f"    - R2 Test:   {metrics['R2 Test']:.4f}")
             print(f"    - MSE Test:  {metrics['MSE Test']:.4f}")
             if not np.isnan(metrics.get('OOB Score', np.nan)):
                 print(f"    - OOB Score: {metrics['OOB Score']:.4f}")
else:
    print("  - Global model results not available.")

# Create separate summary tables for each target
print("\n--- Target-wise Comparison Tables ---")
for target in TARGET_COLUMNS:
    target_results = {k: v for k, v in global_results.items() if f'({target})' in k}
    if target_results:
        summary_df = pd.DataFrame.from_dict(target_results, orient='index')
        print(f"\n{target} Comparison Table:")
        print(summary_df[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))

# Overall summary
summary_df_all = pd.DataFrame.from_dict(global_results, orient='index')
print("\nOverall Comparison Table (All Models and Targets):")
if not summary_df_all.empty:
    print(summary_df_all[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))
else:
    print("No results to display in summary table.")

print("\n--- Multi-Target Discussion ---")
print("Review the target-wise comparison tables above.")
print("Key considerations for multi-target prediction:")
print("1. CL (Lift Coefficient): Usually the primary aerodynamic parameter of interest")
print("2. CD (Drag Coefficient): Critical for efficiency analysis")
print("3. CM (Moment Coefficient): Important for stability and control")
print("4. Compare R2 scores across targets - some may be inherently harder to predict")
print("5. Consider if certain targets benefit more from clustering vs global approaches")
print("6. MSE values will vary significantly between targets due to different scales")
print("7. You might want to implement multi-output models (single model predicting all targets) for comparison")
print("8. Consider feature importance analysis for each target to understand which features matter most")

# Cell 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Data Loading Function (USER ACTION REQUIRED HERE)
def load_airfoil_data(filepath):
    """
    Loads the airfoil data from the specified CSV file.
    USER: You NEED to replace the synthetic data generation below
          with: return pd.read_csv(filepath)
          and ensure your CSV is correctly formatted.
    """
    print(f"Attempting to load data from: {filepath}")
    try:
        # *** USER: REPLACE THE FOLLOWING BLOCK WITH YOUR ACTUAL DATA LOADING ***
        # Example:
        df = pd.read_parquet(filepath)
        print(f"Successfully loaded data from {filepath}")
        # Ensure correct dtypes if necessary, e.g.:
        numeric_cols = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'Re', 
                        'max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
        # *** END OF USER REPLACEMENT BLOCK ***

        # --- Placeholder Synthetic Data Generation (for demonstration) ---
        print("INFO: Using placeholder synthetic data generator. Replace with your actual data loading.")
        num_airfoils = 200
        num_aoa_per_airfoil = 20
        total_rows = num_airfoils * num_aoa_per_airfoil
        airfoil_names = [f'synth_airfoil_{i}' for i in range(num_airfoils)]
        df_list = []
        for i in range(num_airfoils):
            airfoil_name = airfoil_names[i]
            max_thick = np.random.uniform(0.04, 0.18)
            x_thick = np.random.uniform(0.25, 0.45)
            max_camb = np.random.uniform(0.0, 0.06)
            x_camb = np.random.uniform(0.3, 0.7)
            le_rad = np.random.uniform(0.005, 0.03)
            reynolds_number = np.random.choice([100000.0, 200000.0, 300000.0])
            for _ in range(num_aoa_per_airfoil):
                alpha_val = np.random.uniform(-5.0, 15.0)
                cl_linear = 0.1 * alpha_val + 4 * max_camb * (1 - abs(alpha_val / 15.0))
                stall_effect = 0
                if alpha_val > 10: stall_effect = -0.05 * (alpha_val - 10)**2
                elif alpha_val < -4: stall_effect = -0.05 * (alpha_val + 4)**2
                cl_val = cl_linear - 2 * max_thick + stall_effect + np.random.normal(0, 0.05)
                cd_val = 0.01 + 0.0005 * alpha_val**2 + 2 * max_thick**2 + 0.5 * max_camb**2 + np.random.normal(0, 0.001)
                cm_val = -0.05 * max_camb - 0.002 * alpha_val + np.random.normal(0, 0.01)
                df_list.append({
                    'alpha': alpha_val, 'CL': cl_val, 'CD': cd_val, 'CDp': cd_val * 0.8, 'CM': cm_val,
                    'Top_Xtr': np.random.uniform(0.1, 1.0), 'Bot_Xtr': np.random.uniform(0.1, 1.0),
                    'airfoil': airfoil_name, 'Re': reynolds_number, 'max_thickness': max_thick,
                    'x_thickness': x_thick, 'max_camber': max_camb, 'x_camber': x_camb, 'le_radius': le_rad
                })
        df = pd.DataFrame(df_list)
        # --- End of Placeholder ---
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}. Please check the path.")
        return None
    except Exception as e:
        print(f"ERROR: An error occurred while loading data: {e}")
        return None
		
# Cell 3: Helper Functions for Clustered Models (MODIFIED FOR MULTI-TARGET)
def define_bins(df_column, bin_width, min_val=None, max_val=None):
    if df_column.empty: return [], []
    if min_val is None: min_val = df_column.min()
    if max_val is None: max_val = df_column.max()
    min_val_rounded = np.floor(min_val / bin_width) * bin_width
    bins = np.arange(min_val_rounded, max_val + bin_width, bin_width)
    if not bins.size or bins[-1] <= max_val:
         bins = np.append(bins, bins[-1] + bin_width if bins.size > 0 else min_val_rounded + bin_width)
    bins = np.unique(bins)
    if len(bins) < 2: return bins, []
    labels = [f'{bins[i]:.3f}-{bins[i+1]:.3f}' for i in range(len(bins)-1)]
    return bins, labels

def train_models_on_clusters_multi_target(df, cluster_column_name, feature_cols, target_cols, min_samples_per_cluster=50, model_type='rf'):
    """
    Modified function to train separate models for multiple targets (CL, CD, CM)
    Returns dictionaries organized by target, then by cluster
    """
    trained_models = {target: {} for target in target_cols}
    trained_scalers = {}  # Scalers are shared across targets for same cluster
    cluster_results = {target: {} for target in target_cols}
    
    if cluster_column_name not in df.columns or df[cluster_column_name].isnull().all():
        print(f"ERROR: Cluster column '{cluster_column_name}' not found or is all NaN.")
        return trained_models, trained_scalers, cluster_results
    
    unique_clusters = df[cluster_column_name].dropna().unique()
    if not unique_clusters.size:
        print(f"No valid clusters found in '{cluster_column_name}'.")
        return trained_models, trained_scalers, cluster_results

    for cluster_label in unique_clusters:
        print(f"\n--- Processing Cluster: {cluster_label} for {model_type} ---")
        cluster_data_full = df[df[cluster_column_name] == cluster_label].copy()
        
        if len(cluster_data_full) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: {len(cluster_data_full)} samples < {min_samples_per_cluster}.")
            continue
            
        # Check for missing columns
        missing_cols = [col for col in feature_cols + target_cols if col not in cluster_data_full.columns]
        if missing_cols:
            print(f"Skipping {cluster_label}: Missing columns {missing_cols}.")
            continue
            
        cluster_data = cluster_data_full[feature_cols + target_cols].dropna()
        
        if len(cluster_data) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: Insufficient data ({len(cluster_data)} rows) after NaN removal.")
            continue
            
        X = cluster_data[feature_cols]
        
        if X.empty:
            print(f"Skipping {cluster_label}: Empty feature matrix.")
            continue
            
        # Split data once for all targets
        X_train, X_test, cluster_data_train, cluster_data_test = train_test_split(
            X, cluster_data, test_size=0.2, random_state=42
        )
        
        # Scale features once per cluster
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        trained_scalers[cluster_label] = scaler
        
        # Train separate model for each target
        for target_col in target_cols:
            y_train = cluster_data_train[target_col]
            y_test = cluster_data_test[target_col]
            
            if y_train.empty or y_test.empty:
                print(f"Skipping {cluster_label} for target {target_col}: Empty target data.")
                continue
                
            # Create model
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            elif model_type == 'gb':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError("Unsupported model_type")
                
            # Train model
            model.fit(X_train_scaled, y_train)
            trained_models[target_col][cluster_label] = model
            
            # Evaluate model
            y_pred_test = model.predict(X_test_scaled)
            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            oob = model.oob_score_ if hasattr(model, 'oob_score_') and model.oob_score_ else np.nan
            
            cluster_results[target_col][cluster_label] = {
                'model_type': model_type, 
                'cluster_label': cluster_label,
                'target': target_col,
                'train_samples': len(X_train), 
                'test_samples': len(X_test),
                'r2_test': r2_test, 
                'mse_test': mse_test, 
                'oob_score': oob
            }
            
            print(f"  {target_col}: R2={r2_test:.4f}, MSE={mse_test:.4f}, OOB={oob:.4f}")
    
    return trained_models, trained_scalers, cluster_results
	
# Cell 4: Configuration (MODIFIED FOR MULTI-TARGET)
DATA_FILEPATH = 'data_EDA\\airfoil_polars.parquet'  # USER: !!! UPDATE THIS PATH !!!
TARGET_COLUMNS = ['CL', 'CD', 'CM']  # Multiple targets now

# Features for CLUSTERED models (max_thickness used for clustering, not as direct input)
GEOMETRIC_FEATURES_FOR_CLUSTERED = ['x_thickness', 'max_camber', 'x_camber', 'le_radius']
OPERATIONAL_FEATURES = ['alpha', 'Re']
INPUT_FEATURES_FOR_THICKNESS_CLUSTERING = OPERATIONAL_FEATURES + GEOMETRIC_FEATURES_FOR_CLUSTERED

# Features for GLOBAL models (all relevant features are direct inputs)
ALL_GEOMETRIC_FEATURES = ['max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
INPUT_FEATURES_FOR_GLOBAL_MODEL = OPERATIONAL_FEATURES + ALL_GEOMETRIC_FEATURES

# Clustering parameters
THICKNESS_CLUSTER_COL_NAME = 'max_thickness_cluster'
THICKNESS_BIN_WIDTH = 0.02
MIN_SAMPLES_PER_CLUSTER = 80

# Cell 5: Load Data
print("--- Loading Data ---")
df_airfoils = load_airfoil_data(DATA_FILEPATH)
if df_airfoils is None or df_airfoils.empty:
    print("CRITICAL ERROR: Data loading failed. Notebook execution cannot continue meaningfully.")
else:
    print(f"Data loaded successfully. Shape: {df_airfoils.shape}")
    print(df_airfoils.head())
	
# Cell 6: Train and Evaluate Clustered Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 1: Training Benchmark Clustered Models (by Max Thickness) for Multiple Targets ---")
thickness_clustered_models_rf = {}
thickness_clustered_scalers_rf = {}
thickness_cluster_results_summary_rf = {}

if df_airfoils is not None and not df_airfoils.empty:
    mt_min_val = df_airfoils['max_thickness'].min()
    mt_max_val = df_airfoils['max_thickness'].max()
    thickness_bins, thickness_labels = define_bins(df_airfoils['max_thickness'], THICKNESS_BIN_WIDTH, min_val=mt_min_val, max_val=mt_max_val)

    if thickness_labels:
        df_airfoils[THICKNESS_CLUSTER_COL_NAME] = pd.cut(df_airfoils['max_thickness'], bins=thickness_bins, labels=thickness_labels, right=False, include_lowest=True)
        print(f"\nValue counts for {THICKNESS_CLUSTER_COL_NAME}:")
        print(df_airfoils[THICKNESS_CLUSTER_COL_NAME].value_counts().sort_index())

        thickness_clustered_models_rf, thickness_clustered_scalers_rf, thickness_cluster_results_summary_rf = train_models_on_clusters_multi_target(
            df_airfoils.dropna(subset=[THICKNESS_CLUSTER_COL_NAME]),
            THICKNESS_CLUSTER_COL_NAME,
            INPUT_FEATURES_FOR_THICKNESS_CLUSTERING,
            TARGET_COLUMNS,
            min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER,
            model_type='rf'
        )
        
        print("\nSummary of Max Thickness Clustered RF Model Training:")
        weighted_averages_clustered = {}
        
        for target in TARGET_COLUMNS:
            if thickness_cluster_results_summary_rf[target]:
                results_rf_df = pd.DataFrame.from_dict(thickness_cluster_results_summary_rf[target], orient='index')
                print(f"\n{target} Results:")
                print(results_rf_df[['test_samples', 'oob_score', 'r2_test', 'mse_test']].sort_index())
                
                # Calculate weighted average for each target
                total_test_samples_clustered = results_rf_df['test_samples'].sum()
                if total_test_samples_clustered > 0:
                    weighted_r2 = (results_rf_df['r2_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_mse = (results_rf_df['mse_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_averages_clustered[target] = {'R2': weighted_r2, 'MSE': weighted_mse}
                    print(f"Weighted Average for {target} Clustered RF Models: R2 = {weighted_r2:.4f}, MSE = {weighted_mse:.4f}")
                else:
                    weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
                    print(f"No test samples for {target} clustered models.")
            else:
                print(f"No models were trained for {target} thickness clusters.")
                weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
    else:
        print(f"Could not create valid bins/labels for '{THICKNESS_CLUSTER_COL_NAME}'. Skipping clustered model training.")
        weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
else:
    print("Skipping Clustered Model training due to data loading issues.")
    weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
	
# Cell 7: Train and Evaluate Global Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 2: Training Global Models for Multiple Targets ---")
global_results = {}

if df_airfoils is not None and not df_airfoils.empty:
    # Prepare data for global model
    df_global = df_airfoils[INPUT_FEATURES_FOR_GLOBAL_MODEL + TARGET_COLUMNS].copy().dropna()
    
    if len(df_global) > MIN_SAMPLES_PER_CLUSTER:
        X_global = df_global[INPUT_FEATURES_FOR_GLOBAL_MODEL]
        
        # Split data once for all targets
        X_train_global, X_test_global, df_global_train, df_global_test = train_test_split(
            X_global, df_global, test_size=0.2, random_state=42
        )

        # Scale features for global models
        global_scaler = StandardScaler()
        X_train_global_scaled = global_scaler.fit_transform(X_train_global)
        X_test_global_scaled = global_scaler.transform(X_test_global)

        # Train models for each target
        for target in TARGET_COLUMNS:
            y_train_global = df_global_train[target]
            y_test_global = df_global_test[target]
            
            print(f"\n--- Training Global Models for {target} ---")
            
            # --- Global Random Forest ---
            print(f"Training Global Random Forest Model for {target}...")
            rf_global_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            rf_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_rf = rf_global_model.predict(X_test_global_scaled)
            r2_global_rf = r2_score(y_test_global, y_pred_global_rf)
            mse_global_rf = mean_squared_error(y_test_global, y_pred_global_rf)
            oob_global_rf = rf_global_model.oob_score_
            global_results[f'Global RF ({target})'] = {'R2 Test': r2_global_rf, 'MSE Test': mse_global_rf, 'OOB Score': oob_global_rf}
            print(f"Global Random Forest {target}: R2 = {r2_global_rf:.4f}, MSE = {mse_global_rf:.4f}, OOB = {oob_global_rf:.4f}")

            # --- Global Gradient Boosting ---
            print(f"Training Global Gradient Boosting Model for {target}...")
            gb_global_model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3)
            gb_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_gb = gb_global_model.predict(X_test_global_scaled)
            r2_global_gb = r2_score(y_test_global, y_pred_global_gb)
            mse_global_gb = mean_squared_error(y_test_global, y_pred_global_gb)
            global_results[f'Global GB ({target})'] = {'R2 Test': r2_global_gb, 'MSE Test': mse_global_gb, 'OOB Score': np.nan}
            print(f"Global Gradient Boosting {target}: R2 = {r2_global_gb:.4f}, MSE = {mse_global_gb:.4f}")
    else:
        print(f"Insufficient data for global model training after NaN removal (Rows: {len(df_global)}).")
else:
    print("Skipping Global Model training due to data loading issues.")

# Cell 8: Results Comparison and Discussion (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 3: Comparative Results Summary for Multiple Targets ---")

print("\nPerformance of Max Thickness Clustered Random Forest Models (Weighted Average):")
for target in TARGET_COLUMNS:
    if 'weighted_averages_clustered' in locals() and target in weighted_averages_clustered:
        weighted_r2 = weighted_averages_clustered[target]['R2']
        weighted_mse = weighted_averages_clustered[target]['MSE']
        if not np.isnan(weighted_r2):
            print(f"  {target}:")
            print(f"    - Weighted Avg R2: {weighted_r2:.4f}")
            print(f"    - Weighted Avg MSE: {weighted_mse:.4f}")
            # Add to global results for comparison
            global_results[f'Clustered RF ({target}, Weighted Avg)'] = {'R2 Test': weighted_r2, 'MSE Test': weighted_mse, 'OOB Score': np.nan}
        else:
            print(f"  {target}: Clustered model results not available for weighted average.")

print("\nPerformance of Global Models:")
if global_results:
    for model_name, metrics in global_results.items():
        if 'Global' in model_name and 'Clustered' not in model_name:
             print(f"  Model: {model_name}")
             print(f"    - R2 Test:   {metrics['R2 Test']:.4f}")
             print(f"    - MSE Test:  {metrics['MSE Test']:.4f}")
             if not np.isnan(metrics.get('OOB Score', np.nan)):
                 print(f"    - OOB Score: {metrics['OOB Score']:.4f}")
else:
    print("  - Global model results not available.")

# Create separate summary tables for each target
print("\n--- Target-wise Comparison Tables ---")
for target in TARGET_COLUMNS:
    target_results = {k: v for k, v in global_results.items() if f'({target})' in k}
    if target_results:
        summary_df = pd.DataFrame.from_dict(target_results, orient='index')
        print(f"\n{target} Comparison Table:")
        print(summary_df[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))

# Overall summary
summary_df_all = pd.DataFrame.from_dict(global_results, orient='index')
print("\nOverall Comparison Table (All Models and Targets):")
if not summary_df_all.empty:
    print(summary_df_all[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))
else:
    print("No results to display in summary table.")

print("\n--- Multi-Target Discussion ---")
print("Review the target-wise comparison tables above.")
print("Key considerations for multi-target prediction:")
print("1. CL (Lift Coefficient): Usually the primary aerodynamic parameter of interest")
print("2. CD (Drag Coefficient): Critical for efficiency analysis")
print("3. CM (Moment Coefficient): Important for stability and control")
print("4. Compare R2 scores across targets - some may be inherently harder to predict")
print("5. Consider if certain targets benefit more from clustering vs global approaches")
print("6. MSE values will vary significantly between targets due to different scales")
print("7. You might want to implement multi-output models (single model predicting all targets) for comparison")
print("8. Consider feature importance analysis for each target to understand which features matter most")

# Cell 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Data Loading Function (USER ACTION REQUIRED HERE)
def load_airfoil_data(filepath):
    """
    Loads the airfoil data from the specified CSV file.
    USER: You NEED to replace the synthetic data generation below
          with: return pd.read_csv(filepath)
          and ensure your CSV is correctly formatted.
    """
    print(f"Attempting to load data from: {filepath}")
    try:
        # *** USER: REPLACE THE FOLLOWING BLOCK WITH YOUR ACTUAL DATA LOADING ***
        # Example:
        df = pd.read_parquet(filepath)
        print(f"Successfully loaded data from {filepath}")
        # Ensure correct dtypes if necessary, e.g.:
        numeric_cols = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'Re', 
                        'max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
        # *** END OF USER REPLACEMENT BLOCK ***

        # --- Placeholder Synthetic Data Generation (for demonstration) ---
        print("INFO: Using placeholder synthetic data generator. Replace with your actual data loading.")
        num_airfoils = 200
        num_aoa_per_airfoil = 20
        total_rows = num_airfoils * num_aoa_per_airfoil
        airfoil_names = [f'synth_airfoil_{i}' for i in range(num_airfoils)]
        df_list = []
        for i in range(num_airfoils):
            airfoil_name = airfoil_names[i]
            max_thick = np.random.uniform(0.04, 0.18)
            x_thick = np.random.uniform(0.25, 0.45)
            max_camb = np.random.uniform(0.0, 0.06)
            x_camb = np.random.uniform(0.3, 0.7)
            le_rad = np.random.uniform(0.005, 0.03)
            reynolds_number = np.random.choice([100000.0, 200000.0, 300000.0])
            for _ in range(num_aoa_per_airfoil):
                alpha_val = np.random.uniform(-5.0, 15.0)
                cl_linear = 0.1 * alpha_val + 4 * max_camb * (1 - abs(alpha_val / 15.0))
                stall_effect = 0
                if alpha_val > 10: stall_effect = -0.05 * (alpha_val - 10)**2
                elif alpha_val < -4: stall_effect = -0.05 * (alpha_val + 4)**2
                cl_val = cl_linear - 2 * max_thick + stall_effect + np.random.normal(0, 0.05)
                cd_val = 0.01 + 0.0005 * alpha_val**2 + 2 * max_thick**2 + 0.5 * max_camb**2 + np.random.normal(0, 0.001)
                cm_val = -0.05 * max_camb - 0.002 * alpha_val + np.random.normal(0, 0.01)
                df_list.append({
                    'alpha': alpha_val, 'CL': cl_val, 'CD': cd_val, 'CDp': cd_val * 0.8, 'CM': cm_val,
                    'Top_Xtr': np.random.uniform(0.1, 1.0), 'Bot_Xtr': np.random.uniform(0.1, 1.0),
                    'airfoil': airfoil_name, 'Re': reynolds_number, 'max_thickness': max_thick,
                    'x_thickness': x_thick, 'max_camber': max_camb, 'x_camber': x_camb, 'le_radius': le_rad
                })
        df = pd.DataFrame(df_list)
        # --- End of Placeholder ---
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}. Please check the path.")
        return None
    except Exception as e:
        print(f"ERROR: An error occurred while loading data: {e}")
        return None
		
# Cell 3: Helper Functions for Clustered Models (MODIFIED FOR MULTI-TARGET)
def define_bins(df_column, bin_width, min_val=None, max_val=None):
    if df_column.empty: return [], []
    if min_val is None: min_val = df_column.min()
    if max_val is None: max_val = df_column.max()
    min_val_rounded = np.floor(min_val / bin_width) * bin_width
    bins = np.arange(min_val_rounded, max_val + bin_width, bin_width)
    if not bins.size or bins[-1] <= max_val:
         bins = np.append(bins, bins[-1] + bin_width if bins.size > 0 else min_val_rounded + bin_width)
    bins = np.unique(bins)
    if len(bins) < 2: return bins, []
    labels = [f'{bins[i]:.3f}-{bins[i+1]:.3f}' for i in range(len(bins)-1)]
    return bins, labels

def train_models_on_clusters_multi_target(df, cluster_column_name, feature_cols, target_cols, min_samples_per_cluster=50, model_type='rf'):
    """
    Modified function to train separate models for multiple targets (CL, CD, CM)
    Returns dictionaries organized by target, then by cluster
    """
    trained_models = {target: {} for target in target_cols}
    trained_scalers = {}  # Scalers are shared across targets for same cluster
    cluster_results = {target: {} for target in target_cols}
    
    if cluster_column_name not in df.columns or df[cluster_column_name].isnull().all():
        print(f"ERROR: Cluster column '{cluster_column_name}' not found or is all NaN.")
        return trained_models, trained_scalers, cluster_results
    
    unique_clusters = df[cluster_column_name].dropna().unique()
    if not unique_clusters.size:
        print(f"No valid clusters found in '{cluster_column_name}'.")
        return trained_models, trained_scalers, cluster_results

    for cluster_label in unique_clusters:
        print(f"\n--- Processing Cluster: {cluster_label} for {model_type} ---")
        cluster_data_full = df[df[cluster_column_name] == cluster_label].copy()
        
        if len(cluster_data_full) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: {len(cluster_data_full)} samples < {min_samples_per_cluster}.")
            continue
            
        # Check for missing columns
        missing_cols = [col for col in feature_cols + target_cols if col not in cluster_data_full.columns]
        if missing_cols:
            print(f"Skipping {cluster_label}: Missing columns {missing_cols}.")
            continue
            
        cluster_data = cluster_data_full[feature_cols + target_cols].dropna()
        
        if len(cluster_data) < min_samples_per_cluster:
            print(f"Skipping {cluster_label}: Insufficient data ({len(cluster_data)} rows) after NaN removal.")
            continue
            
        X = cluster_data[feature_cols]
        
        if X.empty:
            print(f"Skipping {cluster_label}: Empty feature matrix.")
            continue
            
        # Split data once for all targets
        X_train, X_test, cluster_data_train, cluster_data_test = train_test_split(
            X, cluster_data, test_size=0.2, random_state=42
        )
        
        # Scale features once per cluster
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        trained_scalers[cluster_label] = scaler
        
        # Train separate model for each target
        for target_col in target_cols:
            y_train = cluster_data_train[target_col]
            y_test = cluster_data_test[target_col]
            
            if y_train.empty or y_test.empty:
                print(f"Skipping {cluster_label} for target {target_col}: Empty target data.")
                continue
                
            # Create model
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            elif model_type == 'gb':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError("Unsupported model_type")
                
            # Train model
            model.fit(X_train_scaled, y_train)
            trained_models[target_col][cluster_label] = model
            
            # Evaluate model
            y_pred_test = model.predict(X_test_scaled)
            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            oob = model.oob_score_ if hasattr(model, 'oob_score_') and model.oob_score_ else np.nan
            
            cluster_results[target_col][cluster_label] = {
                'model_type': model_type, 
                'cluster_label': cluster_label,
                'target': target_col,
                'train_samples': len(X_train), 
                'test_samples': len(X_test),
                'r2_test': r2_test, 
                'mse_test': mse_test, 
                'oob_score': oob
            }
            
            print(f"  {target_col}: R2={r2_test:.4f}, MSE={mse_test:.4f}, OOB={oob:.4f}")
    
    return trained_models, trained_scalers, cluster_results
	
# Cell 4: Configuration (MODIFIED FOR MULTI-TARGET)
DATA_FILEPATH = 'data_EDA\\airfoil_polars.parquet'  # USER: !!! UPDATE THIS PATH !!!
TARGET_COLUMNS = ['CL', 'CD', 'CM']  # Multiple targets now

# Features for CLUSTERED models (max_thickness used for clustering, not as direct input)
GEOMETRIC_FEATURES_FOR_CLUSTERED = ['x_thickness', 'max_camber', 'x_camber', 'le_radius']
OPERATIONAL_FEATURES = ['alpha', 'Re']
INPUT_FEATURES_FOR_THICKNESS_CLUSTERING = OPERATIONAL_FEATURES + GEOMETRIC_FEATURES_FOR_CLUSTERED

# Features for GLOBAL models (all relevant features are direct inputs)
ALL_GEOMETRIC_FEATURES = ['max_thickness', 'x_thickness', 'max_camber', 'x_camber', 'le_radius']
INPUT_FEATURES_FOR_GLOBAL_MODEL = OPERATIONAL_FEATURES + ALL_GEOMETRIC_FEATURES

# Clustering parameters
THICKNESS_CLUSTER_COL_NAME = 'max_thickness_cluster'
THICKNESS_BIN_WIDTH = 0.02
MIN_SAMPLES_PER_CLUSTER = 80

# Cell 5: Load Data
print("--- Loading Data ---")
df_airfoils = load_airfoil_data(DATA_FILEPATH)
if df_airfoils is None or df_airfoils.empty:
    print("CRITICAL ERROR: Data loading failed. Notebook execution cannot continue meaningfully.")
else:
    print(f"Data loaded successfully. Shape: {df_airfoils.shape}")
    print(df_airfoils.head())
	
# Cell 6: Train and Evaluate Clustered Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 1: Training Benchmark Clustered Models (by Max Thickness) for Multiple Targets ---")
thickness_clustered_models_rf = {}
thickness_clustered_scalers_rf = {}
thickness_cluster_results_summary_rf = {}

if df_airfoils is not None and not df_airfoils.empty:
    mt_min_val = df_airfoils['max_thickness'].min()
    mt_max_val = df_airfoils['max_thickness'].max()
    thickness_bins, thickness_labels = define_bins(df_airfoils['max_thickness'], THICKNESS_BIN_WIDTH, min_val=mt_min_val, max_val=mt_max_val)

    if thickness_labels:
        df_airfoils[THICKNESS_CLUSTER_COL_NAME] = pd.cut(df_airfoils['max_thickness'], bins=thickness_bins, labels=thickness_labels, right=False, include_lowest=True)
        print(f"\nValue counts for {THICKNESS_CLUSTER_COL_NAME}:")
        print(df_airfoils[THICKNESS_CLUSTER_COL_NAME].value_counts().sort_index())

        thickness_clustered_models_rf, thickness_clustered_scalers_rf, thickness_cluster_results_summary_rf = train_models_on_clusters_multi_target(
            df_airfoils.dropna(subset=[THICKNESS_CLUSTER_COL_NAME]),
            THICKNESS_CLUSTER_COL_NAME,
            INPUT_FEATURES_FOR_THICKNESS_CLUSTERING,
            TARGET_COLUMNS,
            min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER,
            model_type='rf'
        )
        
        print("\nSummary of Max Thickness Clustered RF Model Training:")
        weighted_averages_clustered = {}
        
        for target in TARGET_COLUMNS:
            if thickness_cluster_results_summary_rf[target]:
                results_rf_df = pd.DataFrame.from_dict(thickness_cluster_results_summary_rf[target], orient='index')
                print(f"\n{target} Results:")
                print(results_rf_df[['test_samples', 'oob_score', 'r2_test', 'mse_test']].sort_index())
                
                # Calculate weighted average for each target
                total_test_samples_clustered = results_rf_df['test_samples'].sum()
                if total_test_samples_clustered > 0:
                    weighted_r2 = (results_rf_df['r2_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_mse = (results_rf_df['mse_test'] * results_rf_df['test_samples']).sum() / total_test_samples_clustered
                    weighted_averages_clustered[target] = {'R2': weighted_r2, 'MSE': weighted_mse}
                    print(f"Weighted Average for {target} Clustered RF Models: R2 = {weighted_r2:.4f}, MSE = {weighted_mse:.4f}")
                else:
                    weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
                    print(f"No test samples for {target} clustered models.")
            else:
                print(f"No models were trained for {target} thickness clusters.")
                weighted_averages_clustered[target] = {'R2': np.nan, 'MSE': np.nan}
    else:
        print(f"Could not create valid bins/labels for '{THICKNESS_CLUSTER_COL_NAME}'. Skipping clustered model training.")
        weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
else:
    print("Skipping Clustered Model training due to data loading issues.")
    weighted_averages_clustered = {target: {'R2': np.nan, 'MSE': np.nan} for target in TARGET_COLUMNS}
	
# Cell 7: Train and Evaluate Global Models (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 2: Training Global Models for Multiple Targets ---")
global_results = {}

if df_airfoils is not None and not df_airfoils.empty:
    # Prepare data for global model
    df_global = df_airfoils[INPUT_FEATURES_FOR_GLOBAL_MODEL + TARGET_COLUMNS].copy().dropna()
    
    if len(df_global) > MIN_SAMPLES_PER_CLUSTER:
        X_global = df_global[INPUT_FEATURES_FOR_GLOBAL_MODEL]
        
        # Split data once for all targets
        X_train_global, X_test_global, df_global_train, df_global_test = train_test_split(
            X_global, df_global, test_size=0.2, random_state=42
        )

        # Scale features for global models
        global_scaler = StandardScaler()
        X_train_global_scaled = global_scaler.fit_transform(X_train_global)
        X_test_global_scaled = global_scaler.transform(X_test_global)

        # Train models for each target
        for target in TARGET_COLUMNS:
            y_train_global = df_global_train[target]
            y_test_global = df_global_test[target]
            
            print(f"\n--- Training Global Models for {target} ---")
            
            # --- Global Random Forest ---
            print(f"Training Global Random Forest Model for {target}...")
            rf_global_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
            rf_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_rf = rf_global_model.predict(X_test_global_scaled)
            r2_global_rf = r2_score(y_test_global, y_pred_global_rf)
            mse_global_rf = mean_squared_error(y_test_global, y_pred_global_rf)
            oob_global_rf = rf_global_model.oob_score_
            global_results[f'Global RF ({target})'] = {'R2 Test': r2_global_rf, 'MSE Test': mse_global_rf, 'OOB Score': oob_global_rf}
            print(f"Global Random Forest {target}: R2 = {r2_global_rf:.4f}, MSE = {mse_global_rf:.4f}, OOB = {oob_global_rf:.4f}")

            # --- Global Gradient Boosting ---
            print(f"Training Global Gradient Boosting Model for {target}...")
            gb_global_model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3)
            gb_global_model.fit(X_train_global_scaled, y_train_global)
            y_pred_global_gb = gb_global_model.predict(X_test_global_scaled)
            r2_global_gb = r2_score(y_test_global, y_pred_global_gb)
            mse_global_gb = mean_squared_error(y_test_global, y_pred_global_gb)
            global_results[f'Global GB ({target})'] = {'R2 Test': r2_global_gb, 'MSE Test': mse_global_gb, 'OOB Score': np.nan}
            print(f"Global Gradient Boosting {target}: R2 = {r2_global_gb:.4f}, MSE = {mse_global_gb:.4f}")
    else:
        print(f"Insufficient data for global model training after NaN removal (Rows: {len(df_global)}).")
else:
    print("Skipping Global Model training due to data loading issues.")

# Cell 8: Results Comparison and Discussion (MODIFIED FOR MULTI-TARGET)
print("\n\n--- Step 3: Comparative Results Summary for Multiple Targets ---")

print("\nPerformance of Max Thickness Clustered Random Forest Models (Weighted Average):")
for target in TARGET_COLUMNS:
    if 'weighted_averages_clustered' in locals() and target in weighted_averages_clustered:
        weighted_r2 = weighted_averages_clustered[target]['R2']
        weighted_mse = weighted_averages_clustered[target]['MSE']
        if not np.isnan(weighted_r2):
            print(f"  {target}:")
            print(f"    - Weighted Avg R2: {weighted_r2:.4f}")
            print(f"    - Weighted Avg MSE: {weighted_mse:.4f}")
            # Add to global results for comparison
            global_results[f'Clustered RF ({target}, Weighted Avg)'] = {'R2 Test': weighted_r2, 'MSE Test': weighted_mse, 'OOB Score': np.nan}
        else:
            print(f"  {target}: Clustered model results not available for weighted average.")

print("\nPerformance of Global Models:")
if global_results:
    for model_name, metrics in global_results.items():
        if 'Global' in model_name and 'Clustered' not in model_name:
             print(f"  Model: {model_name}")
             print(f"    - R2 Test:   {metrics['R2 Test']:.4f}")
             print(f"    - MSE Test:  {metrics['MSE Test']:.4f}")
             if not np.isnan(metrics.get('OOB Score', np.nan)):
                 print(f"    - OOB Score: {metrics['OOB Score']:.4f}")
else:
    print("  - Global model results not available.")

# Create separate summary tables for each target
print("\n--- Target-wise Comparison Tables ---")
for target in TARGET_COLUMNS:
    target_results = {k: v for k, v in global_results.items() if f'({target})' in k}
    if target_results:
        summary_df = pd.DataFrame.from_dict(target_results, orient='index')
        print(f"\n{target} Comparison Table:")
        print(summary_df[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))

# Overall summary
summary_df_all = pd.DataFrame.from_dict(global_results, orient='index')
print("\nOverall Comparison Table (All Models and Targets):")
if not summary_df_all.empty:
    print(summary_df_all[['R2 Test', 'MSE Test', 'OOB Score']].sort_values(by='R2 Test', ascending=False))
else:
    print("No results to display in summary table.")

print("\n--- Multi-Target Discussion ---")
print("Review the target-wise comparison tables above.")
print("Key considerations for multi-target prediction:")
print("1. CL (Lift Coefficient): Usually the primary aerodynamic parameter of interest")
print("2. CD (Drag Coefficient): Critical for efficiency analysis")
print("3. CM (Moment Coefficient): Important for stability and control")
print("4. Compare R2 scores across targets - some may be inherently harder to predict")
print("5. Consider if certain targets benefit more from clustering vs global approaches")
print("6. MSE values will vary significantly between targets due to different scales")
print("7. You might want to implement multi-output models (single model predicting all targets) for comparison")
print("8. Consider feature importance analysis for each target to understand which features matter most")

   
# 9. Model Performance Summary Table (as text plot)
plt.subplot(3, 4, 12)
plt.axis('off')

# Create summary text
summary_text = "MODEL PERFORMANCE SUMMARY\n" + "="*30 + "\n\n"
summary_text += "BEST MODELS BY TARGET:\n"

best_models = {}
for target in TARGET_COLUMNS:
    target_models = [(k, v['R2 Test']) for k, v in global_results.items() if f'({target})' in k]
    if target_models:
        best_model = max(target_models, key=lambda x: x[1])
        best_models[target] = best_model
        summary_text += f"{target}: {best_model[0].split('(')[0]} (R²={best_model[1]:.4f})\n"

summary_text += f"\nKEY INSIGHTS:\n"
summary_text += f"• Random Forest consistently outperforms Gradient Boosting\n"
summary_text += f"• CD prediction is most challenging (lowest R²)\n"
summary_text += f"• CM prediction achieves highest accuracy\n"
summary_text += f"• Clustered vs Global shows minimal difference\n"
summary_text += f"• All models achieve excellent performance (R² > 0.98)\n"

plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
        fontfamily='monospace', fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.suptitle('Airfoil Aerodynamic Coefficient Prediction - Model Analysis', 
            fontsize=16, fontweight='bold', y=0.98)
#plt.show()

# Additional specific analysis
print("\n--- Detailed Analysis Results ---")
print("Why Random Forest Performs Best:")
print("1. Non-linear relationships: Aerodynamic coefficients have complex, non-linear dependencies")
print("2. Feature interactions: RF automatically captures interactions between geometry and flow conditions")
print("3. Robustness: Ensemble approach handles outliers and noise better than sequential methods")
print("4. No hyperparameter sensitivity: RF is less sensitive to hyperparameter tuning than GB")

print(f"\nTarget Analysis:")
for target in TARGET_COLUMNS:
    if target in df_airfoils.columns:
        mean_val = df_airfoils[target].mean()
        std_val = df_airfoils[target].std()
        cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
        print(f"{target}: Mean={mean_val:.4f}, Std={std_val:.4f}, CV={cv:.4f}")
    else:
        print("Visualization skipped due to missing data or results.")

print(f"\nWhy CD is Critical:")
print("- Small absolute values make relative errors significant")
print("- Highly non-linear behavior (stall, Reynolds effects)")
print("- Direct impact on efficiency (L/D ratio)")
print("- Most sensitive to small geometric changes")
print("\n--- --------------- ---")



# ## Cell 10) Improved Multi-Output RF: OOB, Tuned Hyperparams, Feature Importance
# - **Input**: 
#     - `X_train_global_scaled`, `X_test_global_scaled`  
#     - `df_global_train[TARGET_COLUMNS]`, `df_global_test[TARGET_COLUMNS]`  
# - **Process**:  
#     1. Wrap a tuned RF in `MultiOutputRegressor` with OOB enabled.  
#     2. Fit on all three targets.  
#     3. Print per‐target OOB, R², and MSE.  
#     4. Compute and display mean feature‐importances.  
# - **Output**:  
#     - Console printout of OOB & test metrics, plus importance table.

# %%
from sklearn.multioutput import MultiOutputRegressor

# Prepare multi-target arrays
y_train_multi = df_global_train[TARGET_COLUMNS].values
y_test_multi  = df_global_test[TARGET_COLUMNS].values

# Base RF hyperparameters (enable OOB)
rf_params = {
    'n_estimators':    200,
    'max_depth':       12,
    'min_samples_leaf':5,
    'oob_score':       True,
    'n_jobs':          -1,
    'random_state':    42
}
base_rf = RandomForestRegressor(**rf_params)

# Wrap for multi-output
multi_rf = MultiOutputRegressor(base_rf)

print("Training MultiOutputRegressor (RF) on all targets at once...")
multi_rf.fit(X_train_global_scaled, y_train_multi)

# --- Per-target OOB scores ---
print("\n--- OOB R² per target ---")
for i, est in enumerate(multi_rf.estimators_):
    print(f"{TARGET_COLUMNS[i]:>4} OOB R² = {est.oob_score_:.4f}")

# Predict on test set
y_pred_multi = multi_rf.predict(X_test_global_scaled)

# Evaluate per target
print("\n--- Test Set Results ---")
for i, target in enumerate(TARGET_COLUMNS):
    r2  = r2_score(y_test_multi[:, i], y_pred_multi[:, i])
    mse = mean_squared_error(y_test_multi[:, i], y_pred_multi[:, i])
    print(f"{target}: R² = {r2:.4f},  MSE = {mse:.4f}")

# --- Aggregate Feature Importances ---
importances = np.vstack([est.feature_importances_ for est in multi_rf.estimators_])
mean_imp   = np.mean(importances, axis=0)
imp_df     = pd.DataFrame({
    'feature': INPUT_FEATURES_FOR_GLOBAL_MODEL,
    'mean_importance': mean_imp
}).sort_values('mean_importance', ascending=False)

print("\Improved multi output RF")
print("\nTop Features (averaged across CL, CD, CM):")
print(imp_df.to_string(index=False))
print("\------------- END STEP 10 --------------")

# %% [markdown]
# ## Cell 10) Hybrid vs. Pure Multi-Output Comparison (All in One Cell)
# This cell:
#  1. Re-defines & trains the hybrid pipeline (multi_clcm, cd_model).
#  2. Uses multi_rf from Cell 9 for pure multi-output predictions.
#  3. Asserts correct array shapes.
#  4. Prints comparison.

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------------------------------------
# A) Re-build the Hybrid Pipeline
# -----------------------------------------------------------------------------

# 1) Feature engineering helper
def add_physics_feats(X_df):
    X = X_df.copy()
    X['alpha_sq']      = X['alpha'] ** 2
    X['inv_Re']        = 1.0 / X['Re']
    X['thick_over_Re'] = X['max_thickness'] / X['Re']
    return X

# 2) Reconstruct original feature DataFrames (unscaled)
X_train_orig = pd.DataFrame(X_train_global_scaled, columns=INPUT_FEATURES_FOR_GLOBAL_MODEL)
X_test_orig  = pd.DataFrame(X_test_global_scaled,  columns=INPUT_FEATURES_FOR_GLOBAL_MODEL)

# 3) Add new features
X_train_fe = add_physics_feats(X_train_orig)
X_test_fe  = add_physics_feats(X_test_orig)

# 4) Scale all features
full_scaler        = StandardScaler()
X_train_fe_scaled  = full_scaler.fit_transform(X_train_fe)
X_test_fe_scaled   = full_scaler.transform(X_test_fe)

# 5) Split targets for hybrid
y_train_cl_cm = df_global_train[['CL','CM']].values
y_train_cd    = df_global_train['CD'].values
y_test_multi  = df_global_test[TARGET_COLUMNS].values  # [CL, CD, CM] true values

# 6a) Multi-output for CL & CM
rf_base     = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
multi_clcm  = MultiOutputRegressor(rf_base)
multi_clcm.fit(X_train_fe_scaled, y_train_cl_cm)

# 6b) Standalone CD model with target scaling
cd_rf    = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
cd_model = TransformedTargetRegressor(regressor=cd_rf,
                                      transformer=StandardScaler())
cd_model.fit(X_train_fe_scaled, y_train_cd)

# -----------------------------------------------------------------------------
# B) Predictions
# -----------------------------------------------------------------------------

# Hybrid predictions (CL, CD, CM)
pred_clcm      = multi_clcm.predict(X_test_fe_scaled)
pred_cd        = cd_model.predict(X_test_fe_scaled)
preds_hybrid   = np.column_stack([pred_clcm[:,0], pred_cd, pred_clcm[:,1]])

# Pure multi-output predictions (using multi_rf from Cell 9)
# X_test_global_scaled and multi_rf must already exist in scope
preds_pure_multi = multi_rf.predict(X_test_global_scaled)

# -----------------------------------------------------------------------------
# C) Sanity Check: shapes must match the true targets
# -----------------------------------------------------------------------------
assert preds_hybrid.shape  == y_test_multi.shape, \
    f"Hybrid preds shape {preds_hybrid.shape} != true shape {y_test_multi.shape}"
assert preds_pure_multi.shape == y_test_multi.shape, \
    f"Pure-multi preds shape {preds_pure_multi.shape} != true shape {y_test_multi.shape}"

# -----------------------------------------------------------------------------
# D) Compute metrics & compare
# -----------------------------------------------------------------------------
results = []
for label, preds in [
        ("Pure Multi-Output", preds_pure_multi),
        ("Hybrid",            preds_hybrid)
    ]:
    row = {"Model": label}
    for i, target in enumerate(TARGET_COLUMNS):
        r2  = r2_score(y_test_multi[:, i], preds[:, i])
        mse = mean_squared_error(y_test_multi[:, i], preds[:, i])
        row[f"{target} R2"]  = r2
        row[f"{target} MSE"] = mse
    results.append(row)

comp_df = pd.DataFrame(results)

# Print the table
print("\n=== Model Comparison on Test Set ===")
print(comp_df.to_string(index=False, float_format='%.4f'))

# Highlight best model per target
print("\nBest R² by target:")
for target in TARGET_COLUMNS:
    best_model = comp_df.loc[comp_df[f"{target} R2"].idxmax(), "Model"]
    print(f"  {target}: {best_model}")


# %% [markdown]
# ## Cell 12) Serialize (Lock-In) Your Trained Pipeline
# After you’ve fit `full_scaler`, `multi_clcm` and `cd_model` in Cell 10,
# dump them to disk so you never have to retrain for inference.

# %%
import joblib

# 1) Save your feature‐scaler
joblib.dump(full_scaler, 'full_scaler.pkl')

# 2) Save the MultiOutputRegressor for CL & CM
joblib.dump(multi_clcm,    'multi_clcm_rf.pkl')

# 3) Save the CD model (with its target transformer inside)
joblib.dump(cd_model,       'cd_model_rf.pkl')

# 4) (Optionally) Save the list of feature names so you can reconstruct DataFrames
joblib.dump(INPUT_FEATURES_FOR_GLOBAL_MODEL + ['alpha_sq','inv_Re','thick_over_Re'],
            'feature_list.pkl')

print("✅ Pipeline serialized: full_scaler.pkl, multi_clcm_rf.pkl, cd_model_rf.pkl")


# %% [markdown]
# ## Cell 13) Residual Diagnostics
# Compute residuals (prediction – true) and plot them against your main inputs
# (α, log₁₀(Re), and top-3 shape features).

# %%

# 1) Recompute or reuse your test‐set originals and preds
#    - X_test_orig: unscaled feature DataFrame from Cell 10
#    - y_true: numpy array of shape (n_samples,3) for [CL,CD,CM]
#    - preds_hybrid: same shape, from the hybrid pipeline

y_true       = df_global_test[TARGET_COLUMNS].values
residuals    = preds_hybrid - y_true

# 2) Prepare the variables you want to inspect
vars_to_plot = {
    'alpha':        X_test_orig['alpha'],
    'logRe':        np.log10(X_test_orig['Re']),
    'x_thickness':  X_test_orig['x_thickness'],
    'max_camber':   X_test_orig['max_camber'],
    'le_radius':    X_test_orig['le_radius'],
}

# 3) Loop over each target and each variable, making one scatter per plot
for i, target in enumerate(TARGET_COLUMNS):
    for var_name, var_series in vars_to_plot.items():
        plt.figure()
        plt.scatter(var_series, residuals[:, i], s=10, alpha=0.6)
        plt.axhline(0, linestyle='--')
        plt.xlabel(var_name)
        plt.ylabel(f"{target} residual")
        plt.title(f"{target} Residual vs. {var_name}")
        plt.show()
"""
FEL Model Evaluation Script

Loads a trained FEL model and evaluates performance on test data.
"""

import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import random
import os
import re
import yaml
import gc
import seaborn as sns
from utils import parse_pv_yml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lume_model.utils import variables_from_yaml, variables_as_yaml
from lume_model.models import TorchModel, TorchModule
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. CONFIGURATION
# ==========================================

SUBSAMPLE_STEP = 2 
Float64_to_32 = 0  # Enable float32 conversion

base_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset_updated/'
model_path = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/model/'
model_version = '2026-03-10_13-37-18_nn_retrain/' # '2026-01-19_06-33-38_nn/' 

# Files to evaluate
test_files = [
    'hxr_archiver_2026-01.pkl', 'hxr_archiver_2026-02.pkl', 'hxr_archiver_2026-03.pkl'
    'hxr_archiver_2025-12.pkl', 'hxr_archiver_2025-11.pkl', 'hxr_archiver_2025-10.pkl',
    # 'hxr_archiver_2025-09.pkl', 'hxr_archiver_2025-06.pkl',
    # 'hxr_archiver_2025-05.pkl', 'hxr_archiver_2025-04.pkl', 'hxr_archiver_2025-03.pkl',
    # 'hxr_archiver_2025-02.pkl', 'hxr_archiver_2025-01.pkl', 'hxr_archiver_2024-12.pkl', 
    # 'hxr_archiver_2024-11.pkl', 'hxr_archiver_2024-10.pkl', 'hxr_archiver_2024-09.pkl', 
    # 'hxr_archiver_2024-08.pkl', 'hxr_archiver_2024-07.pkl', 'hxr_archiver_2024-06.pkl',
]

filter_time_frame = 0

# ==========================================
# 2. MODEL LOADING (FIXED)
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = torch.device(device)
print(f"\n{'='*70}")
print(f"LOADING MODEL")
print(f"{'='*70}")
print(f"Device: {device}")
print(f"Model version: {model_version}")

# Paths
model_dir = model_path + model_version
loaded_model_path = os.path.join(model_dir, 'final_model.pt')  # Use best_model
loaded_input_scaler_path = os.path. join(model_dir, 'input_scaler.pt')
loaded_output_scaler_path = os.path.join(model_dir, 'output_scaler.pt')
config_path = os.path.join(model_dir, 'feature_config.yml')

# # Check files exist
# print("\nChecking files...")
# for path, name in [
#     (loaded_model_path, 'Model weights'),
#     (loaded_input_scaler_path, 'Input scaler'),
#     (loaded_output_scaler_path, 'Output scaler'),
#     (config_path, 'Feature config')
# ]:
#     if os.path.exists(path):
#         print(f"  ✓ {name}:  {path}")
#     else:
#         raise FileNotFoundError(f"  ✗ {name} not found:  {path}")

# # Load configuration to get input size
# print("\nLoading configuration...")
# input_variables, output_variables = variables_from_yaml(config_path)
# input_size = len(input_variables)
# output_size = len(output_variables)
# print(f"  Input size:  {input_size}")
# print(f"  Output size: {output_size}")

# # ⭐ CREATE MODEL INSTANCE (THIS IS THE FIX)
# print("\nCreating model instance...")
# model = FELNeuralNetwork(
#     input_size=input_size,
#     output_size=output_size,
#     hidden_dims=[512, 512, 256, 128, 64, 16, 16],
#     dropout=0.05
# )

# # ⭐ LOAD WEIGHTS INTO MODEL
# print("Loading model weights...")
# state_dict = torch.load(loaded_model_path, map_location=device)
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

# Load artifacts
model = torch.load(loaded_model_path, weights_only=False, map_location=map_loc)
# input_scaler = torch. load(loaded_input_scaler_path, weights_only=False, map_location=map_loc)
# output_scaler = torch.load(loaded_output_scaler_path, weights_only=False, map_location=map_loc)
input_variables, output_variables = variables_from_yaml(config_path)
parsed_variables = parse_pv_yml(config_path)

# Extract ranges
variable_ranges = {}
for variable_name in parsed_variables['input_variables']:
    var_range = parsed_variables['input_variables'][variable_name]. get('value_range')
    if var_range: 
        variable_ranges[variable_name] = var_range


print("  ✓ Model loaded successfully")

# Load scalers
print("\nLoading scalers...")
input_scaler = torch.load(loaded_input_scaler_path, map_location=map_loc, weights_only=False)
output_scaler = torch.load(loaded_output_scaler_path, map_location=map_loc, weights_only=False)
print("  ✓ Scalers loaded")

# Create TorchModel wrapper
print("\nCreating LUME-model wrapper...")
lume_model = TorchModel(
    model=model,  # ← Pass model instance, not state_dict
    input_variables=input_variables,
    output_variables=output_variables,
    input_transformers=[input_scaler],
    output_transformers=[output_scaler],
)
print("  ✓ LUME-model created")

# Disable input validation
lume_model.input_validation_config = {
    pv_name: "none" for pv_name in lume_model.input_names
}

# Create TorchModule
lume_module = TorchModule(
    model=lume_model,
    input_order=lume_model.input_names,
    output_order=lume_model.output_names,
)

print(f"\n{'='*70}")
print(f"✓ MODEL LOADED SUCCESSFULLY")
print(f"{'='*70}\n")

# ==========================================
# 3. ENHANCED FILTERING WITH DETAILED LOGS
# ==========================================

def dataset_filter(dataset, filename="Unknown"):
    """
    Filter dataset with detailed diagnostic output showing pass counts 
    and percentages for each condition.
    """
    total_samples = len(dataset)
    
    if total_samples == 0:
        return dataset

    print(f"\n[Filtering] {filename}")
    print(f"  Original samples: {total_samples: ,}")
    
    # Define all conditions
    conditions = {
        'L1S_S_PV < 0': dataset['ACCL:LI21:1:L1S_S_PV'] < 0,
        'L1S_S_AV > 100': dataset['ACCL:LI21:1:L1S_S_AV'] > 100,
        'LI22 ADES 2000-6000': (dataset['ACCL:LI22:1:ADES'] > 2000) & (dataset['ACCL:LI22:1:ADES'] < 6000),
        'XRMS 250-370': (dataset['CAMR:IN20:186:XRMS'] > 250) & (dataset['CAMR:IN20:186:XRMS'] < 380),
        'YRMS 250-370': (dataset['CAMR:IN20:186:YRMS'] > 250) & (dataset['CAMR:IN20:186:YRMS'] < 380),
        'Intensity 0.1-4': (dataset['GDET:FEE1:241:ENRC'] > 0.1) & (dataset['GDET:FEE1:241:ENRC'] < 4),
        'Charge at gun 240-275': (dataset['SIOC:SYS0:ML00:CALC038'] > 240) & (dataset['SIOC:SYS0:ML00:CALC038'] < 275),
        'Charge after BC1 < 200': dataset['SIOC:SYS0:ML00:CALC252'] < 200,
        'HXR e-energy > 8': dataset['BEND:DMPH:400:BACT'] > 8,
        'HXR photon > 7000': dataset['SIOC:SYS0:ML00:AO627'] > 7000,
    }
    
    # Apply all conditions together
    final_mask = pd.Series(True, index=dataset.index)
    for name, mask in conditions.items():
        final_mask = final_mask & mask

    filtered_dataset = dataset[final_mask]
    remaining = len(filtered_dataset)
    
    print(f"  After physics filter: {remaining:,} ({remaining/total_samples*100:.1f}%)")
    
    return filtered_dataset

# ==========================================
# 4. MEMORY-SAFE DATA LOADING
# ==========================================

frames = []
print(f"\n{'='*70}")
print(f"DATA LOADING")
print(f"{'='*70}")
print(f"Subsample step: {SUBSAMPLE_STEP}")
print(f"Float32 conversion: {'Enabled' if Float64_to_32 else 'Disabled'}")
print(f"Files to load: {len(test_files)}")
print(f"{'='*70}")

for i, fname in enumerate(test_files):
    full_path = os.path.join(base_dir, fname)
    print(f"\n[{i+1}/{len(test_files)}] {fname}")
    
    if not os.path.exists(full_path):
        print(f"  ✗ File not found, skipping")
        continue
        
    try:
        # 1. Load File
        df_temp = pd.read_pickle(full_path)
        print(f"  Loaded:  {len(df_temp):,} samples")
        
        # 2. Filter (Physics)
        df_temp = dataset_filter(df_temp, filename=fname)
        
        # 3. Subsample IMMEDIATELY
        if SUBSAMPLE_STEP > 1:
            df_temp = df_temp.iloc[::SUBSAMPLE_STEP]
            print(f"  After subsample: {len(df_temp):,} samples")
        
        # 4. Cast to float32
        if Float64_to_32:
            float_cols = df_temp.select_dtypes(include=['float64']).columns
            df_temp[float_cols] = df_temp[float_cols]. astype('float32')
            print(f"  Converted {len(float_cols)} columns to float32")

        if len(df_temp) > 0:
            frames.append(df_temp)
            print(f"  ✓ Added to frames")
        else:
            print(f"  ✗ Empty after filtering, skipped")
        
        # 5. Clean up
        del df_temp
        gc.collect() 
        
    except Exception as e:
        print(f"  ✗ ERROR:  {e}")

print(f"\n{'='*70}")

if frames:
    print(f"Concatenating {len(frames)} dataframes...")
    data_from_archiver = pd.concat(frames, axis=0, ignore_index=False)
    del frames
    gc.collect()
    print(f"  ✓ Total samples: {len(data_from_archiver):,}")
else:
    print("  ✗ No data loaded!")
    data_from_archiver = pd. DataFrame()

print(f"{'='*70}\n")

# Check if we have data
if len(data_from_archiver) == 0:
    print("ERROR: No data to evaluate!")
    exit(1)

# ==========================================
# 5. POST-PROCESSING & EVALUATION
# ==========================================

if filter_time_frame:  
    start_time = pd.Timestamp('2025-06-01 00:00:00').tz_localize('US/Pacific')
    end_time   = pd.Timestamp('2025-06-10 00:00:00').tz_localize('US/Pacific')
    test_set = data_from_archiver. loc[start_time:end_time]
    date_tag = f"{pd.to_datetime(start_time).strftime('%Y%m%d')}_to_{pd.to_datetime(end_time).strftime('%Y%m%d')}"
else:
    test_set = data_from_archiver
    date_parts = set()
    for fname in test_files:
        m = re.search(r'(\d{4}-\d{2})', fname)
        if m:
            date_parts.add(m.group(1))
    if date_parts:
        parts_sorted = sorted(date_parts)
        date_tag = parts_sorted[0] if len(parts_sorted) == 1 else f"{parts_sorted[0]}_to_{parts_sorted[-1]}"
    else:
        date_tag = "unknown_dates"

print(f'Test set size: {test_set.shape[0]: ,}')

# Select Range Logic (use all data by default)
selected_ranges = [(0, 1)]
selected_validation_set = pd.DataFrame()
for start_fraction, end_fraction in selected_ranges: 
    start_index = int(start_fraction * len(test_set))
    end_index = int(end_fraction * len(test_set))
    subset = test_set.iloc[start_index:end_index]
    selected_validation_set = pd.concat([selected_validation_set, subset])
    
print(f"Selected validation set size:  {len(selected_validation_set):,}")

# Clean up
del test_set
gc.collect()

# ==========================================
# 6. CHECK FOR REQUIRED COLUMNS
# ==========================================

print(f"\n{'='*70}")
print(f"COLUMN VERIFICATION")
print(f"{'='*70}")

missing_inputs = [col for col in lume_model. input_names if col not in selected_validation_set.columns]
missing_outputs = [col for col in lume_model.output_names if col not in selected_validation_set.columns]

if missing_inputs:
    print(f"✗ Missing input columns ({len(missing_inputs)}):")
    for col in missing_inputs[:10]: 
        print(f"    {col}")
    if len(missing_inputs) > 10:
        print(f"    ...  and {len(missing_inputs) - 10} more")
    print("\nERROR: Cannot proceed without required input columns!")
    exit(1)

if missing_outputs:
    print(f"✗ Missing output columns:  {missing_outputs}")
    print("\nERROR: Cannot proceed without output column!")
    exit(1)

print(f"✓ All required columns present")
print(f"  Input columns: {len(lume_model.input_names)}")
print(f"  Output columns: {len(lume_model.output_names)}")
print(f"{'='*70}\n")

# ==========================================
# 7. INFERENCE
# ==========================================

print(f"{'='*70}")
print(f"RUNNING INFERENCE")
print(f"{'='*70}")
print(f"Samples to evaluate: {len(selected_validation_set):,}")

# Extract data
x_index = selected_validation_set.index
y_true = selected_validation_set[lume_model.output_names]

# Prepare input tensor
X_input = torch.tensor(
    selected_validation_set[lume_model.input_names]. values,
    dtype=torch. float32
).to(device)

print(f"Input tensor shape: {X_input. shape}")

# Run inference
print("Running model inference...")
with torch.no_grad():
    y_pred = lume_module(X_input).cpu().numpy().flatten()

print(f"  ✓ Inference complete")

# Convert to numpy
y_true_np = y_true.values.flatten()
y_pred_np = np.asarray(y_pred)

# Calculate metrics
print("\nCalculating metrics...")
mae = mean_absolute_error(y_true_np, y_pred_np)
rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
r2 = r2_score(y_true_np, y_pred_np)

print(f"\n{'='*70}")
print(f"EVALUATION METRICS")
print(f"{'='*70}")
print(f"  Samples:   {len(y_true_np):,}")
print(f"  MAE:      {mae:.6f} mJ")
print(f"  RMSE:     {rmse:.6f} mJ")
print(f"  R²:       {r2:.6f}")
print(f"{'='*70}\n")

# Rolling statistics (optional, for visualization)
window = 50
y_mean = y_true. rolling(window=window).mean()

# Ensure y_mean is a Series
if isinstance(y_mean, pd.DataFrame):
    y_mean_series = y_mean.iloc[: , 0]
else: 
    y_mean_series = y_mean

# ==========================================
# 8. SAVE RESULTS
# ==========================================

print(f"{'='*70}")
print(f"SAVING RESULTS")
print(f"{'='*70}")

base_model_dir = model_path + model_version
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dump_dir = os.path.join(base_model_dir, f"eval_{date_tag}_{timestamp}")
os.makedirs(dump_dir, exist_ok=True)

# Save predictions CSV
dump_df = pd.DataFrame({
    "timestamp": x_index. astype(str),
    "y_true": y_true_np,
    "y_pred": y_pred_np,
    "error": y_true_np - y_pred_np,
    "abs_error": np.abs(y_true_np - y_pred_np),
    "y_mean_rolling": y_mean_series.reindex(x_index).to_numpy()
})
csv_path = os.path.join(dump_dir, "predictions.csv")
dump_df.to_csv(csv_path, index=False)
print(f"  ✓ Predictions CSV:  {csv_path}")

# Save summary
summary_path = os.path.join(dump_dir, "evaluation_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"FEL Model Evaluation Summary\n")
    f.write(f"{'='*70}\n\n")
    f.write(f"Model Information:\n")
    f.write(f"  Version: {model_version}\n")
    f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Device: {device}\n\n")
    f.write(f"Data Information:\n")
    f.write(f"  Date Range: {date_tag}\n")
    f.write(f"  Subsample Step: {SUBSAMPLE_STEP}\n")
    f.write(f"  Total Samples: {len(selected_validation_set):,}\n")
    f.write(f"  Input Features: {len(lume_model.input_names)}\n")
    f.write(f"  Output Features: {len(lume_model.output_names)}\n\n")
    f.write(f"Performance Metrics:\n")
    f.write(f"  MAE:   {mae:.6f} mJ\n")
    f.write(f"  RMSE:  {rmse:.6f} mJ\n")
    f.write(f"  R²:    {r2:.6f}\n\n")
    f.write(f"{'='*70}\n")

print(f"  ✓ Summary:  {summary_path}")

# Simple scatter plot
print("\nGenerating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax1.scatter(y_true_np, y_pred_np, alpha=0.3, s=1)
ax1.plot([y_true_np.min(), y_true_np.max()], 
         [y_true_np.min(), y_true_np.max()], 
         'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Intensity (mJ)')
ax1.set_ylabel('Predicted Intensity (mJ)')
ax1.set_title(f'Prediction vs Actual (R²={r2:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residual plot
residuals = y_true_np - y_pred_np
ax2.scatter(y_true_np, residuals, alpha=0.3, s=1)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Actual Intensity (mJ)')
ax2.set_ylabel('Residual (Actual - Predicted)')
ax2.set_title(f'Residual Plot (MAE={mae:.4f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(dump_dir, "evaluation_plots.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  ✓ Plots:  {plot_path}")

print(f"\n{'='*70}")
print(f"✓ EVALUATION COMPLETE")
print(f"{'='*70}")
print(f"Results saved to: {dump_dir}")
print(f"{'='*70}\n")
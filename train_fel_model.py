"""
FEL Neural Network Model Training

Loads historical machine archiver data (PVs including quadrupole and RF settings),
applies filtering and preprocessing, and trains a neural network to predict FEL intensity.  

Usage:
    python train_fel_model.py --epochs 50 --batch_size 512
    python train_fel_model.py --resume_from /path/to/checkpoint.pt
    sbatch train_fel_model.slurm  # On SLURM cluster
"""
import os
import gc
import logging
import argparse
import warnings
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch. models. transforms. input import AffineInputTransform
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level='INFO'):
    """Configure logging for better output management."""
    logging.basicConfig(
        level=getattr(logging, log_level. upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a FEL neural network model for intensity prediction."
    )
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model to load")
    parser.add_argument("--subsample_step", type=int, default=1,
                        help="Keep only every Nth sample (>1 to subsample)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory for saving checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=30,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()
    
    if args.model_path and args.resume_from:
        parser.error("Cannot specify both --model_path and --resume_from.  Choose one.")
    
    if args.epochs <= 0:
        parser. error("--epochs must be positive")
    
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    
    if args.subsample_step < 1:
        parser.error("--subsample_step must be >= 1")
    
    return args
    return parser.parse_args()

# ============================================================================
# CHECKPOINT & DIRECTORY SETUP
# ============================================================================

def setup_checkpoint_dir(checkpoint_dir: Optional[str]) -> str:
    """Setup checkpoint directory with safe defaults for cluster environments."""
    if checkpoint_dir:
        ckpt_dir = checkpoint_dir
    else:
        base = os.environ.get("SCRATCH", os.getcwd())
        job_id = os.environ.get("SLURM_JOB_ID", "manual")
        ckpt_dir = os.path.join(base, "fel_tuning", "checkpoints", job_id)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {ckpt_dir}")
    return ckpt_dir

# ============================================================================
# DATA FILTERING & PREPROCESSING
# ============================================================================
def dataset_filter(dataset: pd.DataFrame, log_top_n: int = 5, logger=None) -> pd.DataFrame:
    """
    Filter dataset based on machine operational criteria AND physical PV bounds.
    Now with detailed diagnostic logging to identify strictest conditions.
    
    Args:
        dataset: Input dataframe with PV values
        log_top_n: Number of top strictest conditions to log (default: 5)
        logger: Logger instance (optional)
        
    Returns: 
        Filtered dataframe
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    total_samples = len(dataset)
    
    # Track each condition's impact
    condition_stats = {}
    
    # ============================================================================
    # BASIC PHYSICS FILTER (tracked individually)
    # ============================================================================
    
    # Initialize with all True
    condition = pd.Series(True, index=dataset.index)
    
    # Track each filter individually
    filters = {
        'L1S_S_PV < 0': dataset['ACCL:LI21:1:L1S_S_PV'] < 0,
        'L1S_S_AV > 100': dataset['ACCL:LI21:1:L1S_S_AV'] > 100,
        'LI22 ADES 2000-6000': (dataset['ACCL:LI22:1:ADES'] > 2000) & (dataset['ACCL:LI22:1:ADES'] < 6000),
        'XRMS 250-370': (dataset['XRMS on VCC'] > 250) & (dataset['XRMS on VCC'] < 370),
        'YRMS 250-370': (dataset['YRMS on VCC'] > 250) & (dataset['YRMS on VCC'] < 370),
        'Intensity 0.1-4': (dataset['hxr_pulse_intensity'] > 0.1) & (dataset['hxr_pulse_intensity'] < 4),
        'Charge at gun 240-275': (dataset['Charge at gun [pC]'] > 240) & (dataset['Charge at gun [pC]'] < 275),
        'Charge after BC1 < 200': dataset['Charge after BC1 [pC]'] < 200,
        'HXR e-energy > 8': dataset['HXR electron energy [GeV]'] > 8,
        'HXR photon > 7000': dataset['HXR photon energy [eV]'] > 7000,
    }
    
    # Apply each filter and track impact
    for name, filt in filters.items():
        samples_passing = filt.sum()
        samples_failing = total_samples - samples_passing
        pct_passing = (samples_passing / total_samples) * 100
        
        condition_stats[name] = {
            'passing': samples_passing,
            'failing': samples_failing,
            'pct_passing': pct_passing,
            'pct_failing': 100 - pct_passing,
        }
        
        condition &= filt
    
    # ============================================================================
    # PHYSICAL PV BOUNDS (grouped by category)
    # ============================================================================
    
    XCOR_EXCLUSIONS = ['XCOR:UNDH:4780:BCTRL']
    YCOR_EXCLUSIONS = []
    
    # ---- Undulator X Correctors ----
    xcor_cols = [col for col in dataset.columns 
                 if 'XCOR:UNDH:' in col 
                 and ':BCTRL' in col 
                 and col not in XCOR_EXCLUSIONS]
    
    if xcor_cols:
        xcor_condition = pd.Series(True, index=dataset.index)
        for col in xcor_cols:
            xcor_condition &= (dataset[col] >= -0.001) & (dataset[col] <= 0.001)
        
        samples_passing = xcor_condition.sum()
        condition_stats[f'XCOR bounds ({len(xcor_cols)} correctors)'] = {
            'passing': samples_passing,
            'failing': total_samples - samples_passing,
            'pct_passing': (samples_passing / total_samples) * 100,
            'pct_failing': 100 - (samples_passing / total_samples) * 100,
        }
        condition &= xcor_condition
    
    # ---- Undulator Y Correctors ----
    ycor_cols = [col for col in dataset.columns 
                 if 'YCOR:UNDH:' in col 
                 and ':BCTRL' in col 
                 and col not in YCOR_EXCLUSIONS]
    
    if ycor_cols:
        ycor_condition = pd.Series(True, index=dataset.index)
        for col in ycor_cols:
            ycor_condition &= (dataset[col] >= -0.0013) & (dataset[col] <= 0.0025)
        
        samples_passing = ycor_condition.sum()
        condition_stats[f'YCOR bounds ({len(ycor_cols)} correctors)'] = {
            'passing': samples_passing,
            'failing': total_samples - samples_passing,
            'pct_passing': (samples_passing / total_samples) * 100,
            'pct_failing': 100 - (samples_passing / total_samples) * 100,
        }
        condition &= ycor_condition
    
    # ---- Phase Shifter Gaps ----
    phas_cols = [col for col in dataset.columns if 'PHAS:UNDH:' in col and ':GapDes' in col]
    if phas_cols:
        phas_condition = pd.Series(True, index=dataset.index)
        for col in phas_cols:
            phas_condition &= (dataset[col] >= 12) & (dataset[col] <= 22)
        
        samples_passing = phas_condition.sum()
        condition_stats[f'PHAS gaps ({len(phas_cols)} shifters)'] = {
            'passing': samples_passing,
            'failing': total_samples - samples_passing,
            'pct_passing': (samples_passing / total_samples) * 100,
            'pct_failing': 100 - (samples_passing / total_samples) * 100,
        }
        condition &= phas_condition
    
    # ---- Undulator Segment Gaps ----
    useg_cols = [col for col in dataset.columns if 'USEG:UNDH:' in col and ':GapDes' in col]
    if useg_cols:
        useg_condition = pd.Series(True, index=dataset.index)
        for col in useg_cols:
            useg_condition &= (dataset[col] >= 7.1) & (dataset[col] <= 8)
        
        samples_passing = useg_condition.sum()
        condition_stats[f'USEG gaps ({len(useg_cols)} segments)'] = {
            'passing': samples_passing,
            'failing': total_samples - samples_passing,
            'pct_passing': (samples_passing / total_samples) * 100,
            'pct_failing': 100 - (samples_passing / total_samples) * 100,
        }
        condition &= useg_condition
    
    # ============================================================================
    # LOGGING: Show top N strictest conditions
    # ============================================================================
    
    # Sort by percentage failing (descending)
    sorted_conditions = sorted(
        condition_stats.items(), 
        key=lambda x: x[1]['pct_failing'], 
        reverse=True
    )
    
    logger.info(f"  📊 Filter diagnostics (total: {total_samples:,} samples):")
    logger.info(f"     Top {log_top_n} strictest conditions:")
    
    for i, (name, stats) in enumerate(sorted_conditions[:log_top_n], 1):
        logger.info(
            f"       {i}. {name:40s} → "
            f"Rejected: {stats['failing']:8,} ({stats['pct_failing']:5.1f}%) | "
            f"Kept: {stats['passing']:8,} ({stats['pct_passing']:5.1f}%)"
        )
    
    # Show combined effect
    final_passing = condition.sum()
    final_pct = (final_passing / total_samples) * 100
    logger.info(f"     {'─' * 80}")
    logger.info(
        f"     Combined filter:                             → "
        f"Rejected: {total_samples - final_passing:8,} ({100-final_pct:5.1f}%) | "
        f"Kept: {final_passing:8,} ({final_pct:5.1f}%)"
    )
    
    return dataset[condition]
    
def detect_low_variability_pvs_percentile(df, input_cols, percentile_threshold=0.01, absolute_threshold=1e-6):
    """
    Detect low-variability PVs using percentile range.  
    
    Checks if (P75 - P25) / (P99 - P1) < threshold
    (i.e., middle 50% of data spans < 1% of full range)
    
    Args:
        df: DataFrame with PV data
        input_cols: List of columns to check
        percentile_threshold: Relative IQR threshold (default 0.01 = 1%)
        absolute_threshold: Absolute range threshold for constant detection
        
    Returns:
        dict with: 
            - 'low_variability': List of PV names with low variability
            - 'reasons': Dict mapping PV → reason for removal
            - 'stats': Dict mapping PV → statistics
    """
    low_variability_cols = []
    removal_reasons = {}
    pv_stats = {}
    
    for col in input_cols:  
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        if len(data) < 10:
            logger.warning(f"{col}:  Insufficient data ({len(data)} samples)")
            continue
        
        # Calculate percentiles
        p1, p25, p50, p75, p99 = data. quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        
        # Interquartile range (middle 50% of data)
        iqr = p75 - p25
        
        # Full range (excluding extreme outliers)
        full_range = p99 - p1
        
        # Store statistics for all PVs
        pv_stats[col] = {
            'p1': float(p1),
            'p25': float(p25),
            'median': float(p50),
            'p75': float(p75),
            'p99': float(p99),
            'iqr': float(iqr),
            'range': float(full_range),
            'n_samples': int(len(data))
        }
        
        # Check 1: Absolutely constant (range ≈ 0)
        if full_range < absolute_threshold:
            low_variability_cols.append(col)
            removal_reasons[col] = f"Constant (range={full_range:.2e} < {absolute_threshold:.2e})"
            continue
        
        # Check 2: Relative IQR (low variability relative to range)
        relative_iqr = iqr / full_range
        pv_stats[col]['relative_iqr'] = float(relative_iqr)
        
        if relative_iqr < percentile_threshold:
            low_variability_cols.append(col)
            removal_reasons[col] = f"Low variability (IQR/range={relative_iqr:.4f} < {percentile_threshold}, IQR={iqr:.3f}, range={full_range:.3f})"
    
    return {
        'low_variability': low_variability_cols,
        'reasons': removal_reasons,
        'stats': pv_stats
    }

def clip_outliers_iqr(df, columns, iqr_multiplier=3.0, min_samples=100):
    """
    Clip outliers using IQR method for each column independently. 
    
    Removes values outside [Q25 - k*IQR, Q75 + k*IQR]
    
    Args:
        df: DataFrame
        columns: List of column names to clip
        iqr_multiplier: How many IQRs to allow (default 3.0 = ~99.7% coverage)
        min_samples:  Minimum samples required per PV
        
    Returns:
        df_cleaned: DataFrame with outliers removed
        outlier_report: Dict with statistics per PV
    """
    import numpy as np
    import pandas as pd
    
    df_cleaned = df.copy()
    outlier_report = {}
    total_rows_original = len(df)
    
    for col in columns:
        if col not in df. columns:
            continue
        
        data = df[col].replace([np.inf, -np. inf], np.nan).dropna()
        
        if len(data) < min_samples:
            logger.warning(f"{col}: Too few samples ({len(data)}), skipping outlier filter")
            continue
        
        # Calculate IQR bounds
        q25, q75 = data.quantile([0.25, 0.75])
        iqr = q75 - q25
        
        lower_bound = q25 - iqr_multiplier * iqr
        upper_bound = q75 + iqr_multiplier * iqr
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        n_outliers = outliers.sum()
        outlier_pct = (n_outliers / len(df)) * 100
        
        # Create mask for this column
        valid_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        
        # Apply filter (accumulate masks across all columns)
        if col == columns[0]: 
            combined_mask = valid_mask
        else:
            combined_mask = combined_mask & valid_mask
        
        outlier_report[col] = {
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'q25': q25,
            'q75': q75,
            'iqr': iqr
        }
        
        if n_outliers > 0:
            logger.info(f"{col}:  Removed {n_outliers} outliers ({outlier_pct:.2f}%) "
                       f"outside [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Apply combined mask
    df_cleaned = df[combined_mask]. copy()
    
    total_removed = total_rows_original - len(df_cleaned)
    removal_pct = (total_removed / total_rows_original) * 100
    
    logger.info("=" * 70)
    logger.info(f"OUTLIER FILTERING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Original samples: {total_rows_original}")
    logger.info(f"Cleaned samples:   {len(df_cleaned)}")
    logger.info(f"Removed samples:  {total_removed} ({removal_pct:.2f}%)")
    logger.info("=" * 70)
    
    return df_cleaned, outlier_report


def print_invalid_pv_summary(
    manually_invalid_pvs, 
    low_variance_cols, 
    removal_reasons, 
    available_cols,
    output_file=None
):
    """
    Print comprehensive summary of all invalid PVs. 
    
    Args:
        manually_invalid_pvs: List of manually specified invalid PVs
        low_variance_cols: List of auto-detected low-variance PVs
        removal_reasons: Dict mapping PV → reason
        available_cols: List of all available PVs
        output_file: Optional file path to save summary
    """
    
    # Separate manual invalids that are actually in the data
    manual_in_data = [pv for pv in manually_invalid_pvs if pv in available_cols]
    manual_not_in_data = [pv for pv in manually_invalid_pvs if pv not in available_cols]
    
    # Build summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("INVALID PV SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Section 1: Manually specified invalid PVs
    summary_lines.append("-" * 80)
    summary_lines.append(f"1.  MANUALLY SPECIFIED INVALID PVs (Total: {len(manually_invalid_pvs)})")
    summary_lines.append("-" * 80)
    summary_lines.append(f"   - Found in data:      {len(manual_in_data)}")
    summary_lines.append(f"   - Not in data:       {len(manual_not_in_data)}")
    summary_lines.append("")
    
    if manual_in_data:
        summary_lines.append("   PVs found in data (WILL BE REMOVED):")
        for i, pv in enumerate(sorted(manual_in_data), 1):
            summary_lines.append(f"      {i: 3d}. {pv}")
        summary_lines.append("")
    
    if manual_not_in_data:
        summary_lines.append("   PVs NOT in data (already missing):")
        for i, pv in enumerate(sorted(manual_not_in_data), 1):
            summary_lines.append(f"      {i:3d}. {pv}")
        summary_lines.append("")
    
    # Section 2: Auto-detected low-variance PVs
    summary_lines.append("-" * 80)
    summary_lines.append(f"2. AUTO-DETECTED LOW-VARIANCE PVs (Total:  {len(low_variance_cols)})")
    summary_lines.append("-" * 80)
    summary_lines.append("")
    
    if low_variance_cols:
        # Separate by reason type
        constant_pvs = [pv for pv in low_variance_cols if 'Constant' in removal_reasons. get(pv, '')]
        low_var_pvs = [pv for pv in low_variance_cols if 'Low variability' in removal_reasons.get(pv, '')]
        
        if constant_pvs: 
            summary_lines.append(f"   2a. Constant PVs (range ≈ 0): {len(constant_pvs)}")
            summary_lines.append("")
            for i, pv in enumerate(sorted(constant_pvs), 1):
                reason = removal_reasons.get(pv, 'Unknown')
                summary_lines.append(f"      {i:3d}. {pv}")
                summary_lines.append(f"           → {reason}")
            summary_lines.append("")
        
        if low_var_pvs:
            summary_lines. append(f"   2b.  Low-Variance PVs (IQR < {percentile_threshold*100}% of range): {len(low_var_pvs)}")
            summary_lines.append("")
            for i, pv in enumerate(sorted(low_var_pvs), 1):
                reason = removal_reasons.get(pv, 'Unknown')
                summary_lines.append(f"      {i:3d}. {pv}")
                summary_lines.append(f"           → {reason}")
            summary_lines. append("")
    else:
        summary_lines.append("   No low-variance PVs detected.")
        summary_lines.append("")
    
    # Section 3: Total summary
    total_invalid = len(set(manual_in_data + low_variance_cols))
    summary_lines.append("-" * 80)
    summary_lines.append("3. TOTAL INVALID PVs")
    summary_lines.append("-" * 80)
    summary_lines.append(f"   - Manually specified (in data): {len(manual_in_data)}")
    summary_lines.append(f"   - Auto-detected low-variance:    {len(low_variance_cols)}")
    summary_lines.append(f"   - Overlap (if any):              {len(manual_in_data) + len(low_variance_cols) - total_invalid}")
    summary_lines.append(f"   = TOTAL UNIQUE INVALID:           {total_invalid}")
    summary_lines.append("")
    
    # Section 4: Complete list of all invalid PVs
    all_invalid = sorted(set(manual_in_data + low_variance_cols))
    summary_lines.append("-" * 80)
    summary_lines.append(f"4. COMPLETE LIST OF ALL INVALID PVs TO BE REMOVED ({len(all_invalid)})")
    summary_lines.append("-" * 80)
    summary_lines.append("")
    for i, pv in enumerate(all_invalid, 1):
        if pv in manual_in_data and pv in low_variance_cols:
            tag = "[MANUAL + AUTO]"
        elif pv in manual_in_data:
            tag = "[MANUAL]"
        else:
            tag = "[AUTO]"
        summary_lines.append(f"   {i:3d}. {tag: 20s} {pv}")
    summary_lines.append("")
    
    summary_lines.append("=" * 80)
    
    # Print to console
    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)
    
    # Optionally save to file
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f. write(summary_text)
            logger.info(f"Invalid PV summary saved to:  {output_file}")
        except Exception as e:
            logger. error(f"Failed to save summary to file: {e}")
    
    return summary_text


def load_and_preprocess_data(args, input_cols_override=None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Load pickle files, apply filters, exclusions, and validation splits.
    
    Args:
        args: Command-line arguments
        input_cols_override: If provided, skip feature detection and use these columns
    
    Returns: 
        (final_df, val_df, input_cols, output_cols)
    """
    logger.info("=" * 70)
    logger.info("DATA LOADING & PREPROCESSING")
    logger.info("=" * 70)
    
    file_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset/'
    pickle_files = [
        'hxr_archiver_2025-11.pkl', 'hxr_archiver_2025-10.pkl', 'hxr_archiver_2025-09.pkl',
        'hxr_archiver_2025-06.pkl', 'hxr_archiver_2025-05.pkl', 'hxr_archiver_2025-04.pkl',
        'hxr_archiver_2025-03.pkl', 'hxr_archiver_2025-02.pkl', 'hxr_archiver_2025-01.pkl',
        'hxr_archiver_2024-12.pkl', 'hxr_archiver_2024-11.pkl', 'hxr_archiver_2024-10.pkl',
        'hxr_archiver_2024-09.pkl', 'hxr_archiver_2024-08.pkl', 'hxr_archiver_2024-07.pkl',
        'hxr_archiver_2024-06.pkl', 'hxr_archiver_2024-05.pkl', 'hxr_archiver_2024-04.pkl',
        'hxr_archiver_2024-03.pkl', 'hxr_archiver_2024-02.pkl', 'hxr_archiver_2024-01.pkl',
        'hxr_archiver_2023-11.pkl', 'hxr_archiver_2023-10.pkl', 'hxr_archiver_2023-09.pkl',
        'hxr_archiver_2023-08.pkl', 'hxr_archiver_2023-07.pkl',
    ]
    
    logger.info(f"Loading {len(pickle_files)} pickle files")
    
    dfs = []
    # In load_and_preprocess_data(), update the loop:
    
    for i, file in enumerate(pickle_files):
        full_path = os.path.join(file_dir, file)
        status_msg = f"  [{i+1}/{len(pickle_files)}] {file}"
        
        try:
            logger.info(f"\n{status_msg}")
            logger.info(f"  {'─' * 80}")
            
            temp_df = pd.read_pickle(full_path)
            original_count = len(temp_df)
            
            temp_df = dataset_filter(temp_df, log_top_n=5, logger=logger)
            
            filtered_count = len(temp_df)
            
            # Memory optimization
            float_cols = temp_df.select_dtypes(include=['float64']).columns
            temp_df[float_cols] = temp_df[float_cols].astype('float32')
            
            if not temp_df.empty:
                dfs.append(temp_df)
                retention_pct = (filtered_count / original_count * 100) if original_count > 0 else 0
                logger.info(f"  ✓ Retained: {filtered_count:,} / {original_count:,} ({retention_pct:.1f}%)")
            else:
                logger.info(f"  ✗ All samples filtered out")
            
            del temp_df
            gc.collect()
        except Exception as e:
            logger.error(f"{status_msg} ✗ Error: {e}")
    
    if not dfs:
        raise ValueError("No data remained after filtering!  Check filter conditions.")
    
    final_df = pd.concat(dfs, axis=0, ignore_index=False)
    logger.info(f"Total samples after filtering: {len(final_df)}")
    logger.info(f"Total columns in combined data: {len(final_df. columns)}")
    del dfs
    gc.collect()
    
    # ---- Timezone Conversion ----
    logger.info("Converting timezone to US/Pacific...")
    final_df. index = pd.to_datetime(final_df.index)
    if final_df.index. tz is None:
        final_df.index = final_df. index.tz_localize("UTC")
    final_df. index = final_df.index. tz_convert("US/Pacific")
    
    # ---- Apply Exclusion Windows (Maintenance, etc.) ----
    logger.info("Applying exclusion windows (MDs, maintenance, downtime)...")
    exclusion_windows = [
        ("2025-11-27 10:00", "2025-11-28 18:00"),
        ("2025-11-20 07:00", "2025-11-21 06:00"),
        ("2025-10-22 12:00", "2025-10-24 06:00"),
        ("2025-10-01 06:00", "2025-10-02 06:00"),
        ("2025-09-16 06:00", "2025-09-17 06:00"),
        ("2025-09-09 06:00", "2025-09-11 06:00"),
        ("2025-06-25 17:00", "2025-06-26 04:00"),
        ("2025-05-28 23:00", "2025-05-29 06:00"),
        ("2025-05-21 20:00", "2025-05-22 06:00"),
        ("2025-05-15 13:00", "2025-05-15 22:00"),
        ("2025-05-06 12:00", "2025-05-07 02:00"),
        ("2025-04-17 11:00", "2025-04-17 23:00"),
        ("2025-04-02 07:00", "2025-04-02 18:00"),
        ("2025-03-26 15:00", "2025-03-27 02:00"),
        ("2025-02-05 07:00", "2025-02-05 17:00"),
        ("2024-11-21 08:30", "2024-11-21 18:00"),
        ("2024-11-12 16:00", "2024-11-13 02:00"),
        ("2024-11-06 07:30", "2024-11-06 15:30"),
        ("2024-10-15 07:00", "2024-10-16 08:00"),
        ("2024-09-04 21:00", "2024-09-05 15:30"),
        ("2024-06-06 20:30", "2024-06-07 04:30"),
        ("2024-05-09 15:00", "2024-05-09 22:00"),
        ("2024-03-28 10:00", "2024-03-29 02:00"),
        ("2024-03-20 17:00", "2024-03-21 01:30"),
        ("2024-02-14 19:00", "2024-02-15 03:00"),
        ("2023-11-16 08:00", "2023-11-16 17:00"),
        ("2023-11-09 16:00", "2023-11-10 04:00"),
        ("2023-11-01 13:00", "2023-11-01 22:00"),
        ("2023-10-05 09:00", "2023-10-06 05:00"),
        ("2023-09-27 21:00", "2023-09-28 03:00"),
        ("2023-09-21 09:00", "2023-09-21 19:00"),
        ("2023-08-30 06:00", "2023-08-30 18:00"),
    ]
    
    exclusion_mask = pd.Series(False, index=final_df.index)
    exclusion_count = 0
    for t0, t1 in exclusion_windows:
        start = pd.Timestamp(t0, tz="US/Pacific")
        end = pd.Timestamp(t1, tz="US/Pacific")
        window_mask = (final_df. index >= start) & (final_df.index <= end)
        cnt = window_mask.sum()
        if cnt > 0:
            logger.debug(f"  Excluding {cnt} rows from {t0} to {t1}")
            exclusion_count += cnt
        exclusion_mask |= window_mask
    
    final_df = final_df[~exclusion_mask]
    logger.info(f"Removed {exclusion_count} samples in exclusion windows")
    logger.info(f"Remaining after exclusions: {len(final_df)}")
    
    # ---- Apply Validation Windows (Time-based split) ----
    logger.info("Applying validation windows...")
    validation_windows = [
        ("2025-11-25 00:00", "2025-12-05 00:00"),
        ("2025-10-25 00:00", "2025-11-05 00:00"),
        ("2025-09-25 00:00", "2025-10-05 00:00"),
        ("2025-08-25 00:00", "2025-09-05 00:00"),
        ("2025-06-25 00:00", "2025-07-05 00:00"),
        ("2025-05-25 00:00", "2025-06-05 00:00"),
        ("2025-04-25 00:00", "2025-05-05 00:00"),
        ("2025-03-25 00:00", "2025-04-05 00:00"),
        ("2025-02-25 00:00", "2025-03-05 00:00"),
        ("2024-11-25 00:00", "2024-12-05 00:00"),
        ("2024-10-25 00:00", "2024-11-05 00:00"),
        ("2024-09-25 00:00", "2024-10-05 00:00"),
        ("2024-08-25 00:00", "2024-09-05 00:00"),
        ("2024-06-25 00:00", "2024-07-05 00:00"),
        ("2024-05-25 00:00", "2024-06-05 00:00"),
        ("2024-04-25 00:00", "2024-05-05 00:00"),
        ("2024-03-25 00:00", "2024-04-05 00:00"),
        ("2024-02-25 00:00", "2024-03-05 00:00"),
        ("2023-10-25 00:00", "2023-11-05 00:00"),
        ("2023-09-25 00:00", "2023-10-05 00:00"),
        ("2023-08-25 00:00", "2023-09-05 00:00"),
        ("2023-07-25 00:00", "2023-08-05 00:00"),
    ]
    
    val_mask = pd.Series(False, index=final_df.index)
    for t0, t1 in validation_windows:
        start = pd.Timestamp(t0, tz="US/Pacific")
        end = pd.Timestamp(t1, tz="US/Pacific")
        val_mask |= (final_df.index >= start) & (final_df.index <= end)
    
    val_df = final_df[val_mask]. copy()
    final_df = final_df[~val_mask]. copy()
    logger.info(f"Validation samples:{len(val_df)}")
    logger.info(f"Training/test samples:{len(final_df)}")

    # ============================================================================
    # FEATURE SELECTION & INVALID PV REMOVAL
    # ============================================================================
    if input_cols_override is not None: 
        logger.info("=" * 70)
        logger.info("RETRAINING MODE:  Using pre-defined feature set")
        logger.info("=" * 70)
        logger.info(f"Loading {len(input_cols_override)} input features from model")
        
        input_cols = input_cols_override
        output_cols = ['hxr_pulse_intensity']
        
        # Verify all required columns exist
        missing_cols = [c for c in input_cols if c not in final_df.columns]
        if missing_cols:
            logger. error(f"❌ Missing {len(missing_cols)} required columns from pre-trained model!")
            logger.error(f"First 10 missing:  {missing_cols[:10]}")
            raise ValueError(f"Cannot retrain:  {len(missing_cols)} required features missing from data")
        
        logger.info(f"✓ All {len(input_cols)} required features present in data")
        logger.info("=" * 70)
        
        # Drop columns we don't need (optional, for memory)
        cols_to_keep = input_cols + output_cols + [c for c in final_df. columns if c in ['timestamp']]
        extra_cols = [c for c in final_df.columns if c not in cols_to_keep]
        if extra_cols:
            logger. info(f"Dropping {len(extra_cols)} unused columns to save memory")
            final_df = final_df.drop(columns=extra_cols, errors='ignore')
            val_df = val_df.drop(columns=extra_cols, errors='ignore')
        
        # Skip to the end
        return final_df, val_df, input_cols, output_cols
    
    logger.info("=" * 70)
    logger.info("FRESH TRAINING MODE:  Detecting features from data")
    logger.info("=" * 70)
    logger.info("FEATURE SELECTION & INVALID PV REMOVAL")
    logger.info("=" * 70)

    
    # ---- Step 1:Define desired input features (hardcoded lists) ----
    RF_ampls = ['ACCL:LI21:1:L1S_S_AV', 'ACCL:LI21:180:L1X_S_AV', 'ACCL:LI22:1:ADES', 'ACCL:LI25:1:ADES']
    RF_phases = ['ACCL:LI21:1:L1S_S_PV', 'ACCL:LI21:180:L1X_S_PV', 'ACCL:LI22:1:PDES', 'ACCL:LI25:1:PDES']
    vcc_profile = ['XRMS on VCC', 'YRMS on VCC']
    
    undh_corr_x = [
        'XCOR:UNDH:1380:BCTRL', 'XCOR:UNDH:1480:BCTRL', 'XCOR:UNDH:1580:BCTRL', 'XCOR:UNDH:1680:BCTRL',
        'XCOR:UNDH:1780:BCTRL', 'XCOR:UNDH:1880:BCTRL', 'XCOR:UNDH:1980:BCTRL', 'XCOR:UNDH:2080:BCTRL',
        'XCOR:UNDH:2180:BCTRL', 'XCOR:UNDH:2280:BCTRL', 'XCOR:UNDH:2380:BCTRL', 'XCOR:UNDH:2480:BCTRL',
        'XCOR:UNDH:2580:BCTRL', 'XCOR:UNDH:2680:BCTRL', 'XCOR:UNDH:2780:BCTRL', 'XCOR:UNDH:2880:BCTRL',
        'XCOR:UNDH:2980:BCTRL', 'XCOR:UNDH:3080:BCTRL', 'XCOR:UNDH:3180:BCTRL', 'XCOR:UNDH:3280:BCTRL',
        'XCOR:UNDH:3380:BCTRL', 'XCOR:UNDH:3480:BCTRL', 'XCOR:UNDH:3580:BCTRL', 'XCOR:UNDH:3680:BCTRL',
        'XCOR:UNDH:3780:BCTRL', 'XCOR:UNDH:3880:BCTRL', 'XCOR:UNDH:3980:BCTRL', 'XCOR:UNDH:4080:BCTRL',
        'XCOR:UNDH:4180:BCTRL', 'XCOR:UNDH:4280:BCTRL', 'XCOR:UNDH:4380:BCTRL', 'XCOR:UNDH:4480:BCTRL',
        'XCOR:UNDH:4580:BCTRL', 'XCOR:UNDH:4680:BCTRL', 'XCOR:UNDH:4780:BCTRL'
    ]
    
    undh_corr_y = [
        'YCOR:UNDH:1380:BCTRL', 'YCOR:UNDH:1480:BCTRL', 'YCOR:UNDH:1580:BCTRL', 'YCOR:UNDH:1680:BCTRL',
        'YCOR:UNDH:1780:BCTRL', 'YCOR:UNDH:1880:BCTRL', 'YCOR:UNDH:1980:BCTRL', 'YCOR:UNDH:2080:BCTRL',
        'YCOR:UNDH:2180:BCTRL', 'YCOR:UNDH:2280:BCTRL', 'YCOR:UNDH:2380:BCTRL', 'YCOR:UNDH:2480:BCTRL',
        'YCOR:UNDH:2580:BCTRL', 'YCOR:UNDH:2680:BCTRL', 'YCOR:UNDH:2780:BCTRL', 'YCOR:UNDH:2880:BCTRL',
        'YCOR:UNDH:2980:BCTRL', 'YCOR:UNDH:3080:BCTRL', 'YCOR:UNDH:3180:BCTRL', 'YCOR:UNDH:3280:BCTRL',
        'YCOR:UNDH:3380:BCTRL', 'YCOR:UNDH:3480:BCTRL', 'YCOR:UNDH:3580:BCTRL', 'YCOR:UNDH:3680:BCTRL',
        'YCOR:UNDH:3780:BCTRL', 'YCOR:UNDH:3880:BCTRL', 'YCOR:UNDH:3980:BCTRL', 'YCOR:UNDH:4080:BCTRL',
        'YCOR:UNDH:4180:BCTRL', 'YCOR:UNDH:4280:BCTRL', 'YCOR:UNDH:4380:BCTRL', 'YCOR:UNDH:4480:BCTRL',
        'YCOR:UNDH:4580:BCTRL', 'YCOR:UNDH:4680:BCTRL', 'YCOR:UNDH:4780:BCTRL'
    ]
    
    undh_shifter = [
        'PHAS:UNDH:1495:GapDes', 'PHAS:UNDH:1595:GapDes', 'PHAS:UNDH:1695:GapDes', 'PHAS:UNDH:1795:GapDes',
        'PHAS:UNDH:1895:GapDes', 'PHAS:UNDH:1995:GapDes', 'PHAS:UNDH:2095:GapDes', 'PHAS:UNDH:2295:GapDes',
        'PHAS:UNDH:2395:GapDes', 'PHAS:UNDH:2495:GapDes', 'PHAS:UNDH:2595:GapDes', 'PHAS:UNDH:2695:GapDes',
        'PHAS:UNDH:2795:GapDes', 'PHAS:UNDH:2995:GapDes', 'PHAS:UNDH:3095:GapDes', 'PHAS:UNDH:3195:GapDes',
        'PHAS:UNDH:3295:GapDes', 'PHAS:UNDH:3395:GapDes', 'PHAS:UNDH:3495:GapDes', 'PHAS:UNDH:3595:GapDes',
        'PHAS:UNDH:3695:GapDes', 'PHAS:UNDH:3795:GapDes', 'PHAS:UNDH:3895:GapDes', 'PHAS:UNDH:3995:GapDes',
        'PHAS:UNDH:4095:GapDes', 'PHAS:UNDH:4195:GapDes', 'PHAS:UNDH:4295:GapDes', 'PHAS:UNDH:4395:GapDes',
        'PHAS:UNDH:4495:GapDes', 'PHAS:UNDH:4595:GapDes', 'PHAS:UNDH:4695:GapDes'
    ]
    
    undh_gap = [
        'USEG:UNDH:1450:GapDes', 'USEG:UNDH:1550:GapDes', 'USEG:UNDH:1650:GapDes', 'USEG:UNDH:1750:GapDes',
        'USEG:UNDH:1850:GapDes', 'USEG:UNDH:1950:GapDes', 'USEG:UNDH:2050:GapDes', 'USEG:UNDH:2250:GapDes',
        'USEG:UNDH:2350:GapDes', 'USEG:UNDH:2450:GapDes', 'USEG:UNDH:2550:GapDes', 'USEG:UNDH:2650:GapDes',
        'USEG:UNDH:2750:GapDes', 'USEG:UNDH:2950:GapDes', 'USEG:UNDH:3050:GapDes', 'USEG:UNDH:3150:GapDes',
        'USEG:UNDH:3250:GapDes', 'USEG:UNDH:3350:GapDes', 'USEG:UNDH:3450:GapDes', 'USEG:UNDH:3550:GapDes',
        'USEG:UNDH:3650:GapDes', 'USEG:UNDH:3750:GapDes', 'USEG:UNDH:3850:GapDes', 'USEG:UNDH:3950:GapDes',
        'USEG:UNDH:4050:GapDes', 'USEG:UNDH:4150:GapDes', 'USEG:UNDH:4250:GapDes', 'USEG:UNDH:4350:GapDes',
        'USEG:UNDH:4450:GapDes', 'USEG:UNDH:4550:GapDes', 'USEG:UNDH:4650:GapDes', 'USEG:UNDH:4750:GapDes'
    ]
    
    # Load quadrupoles from CSV
    try:
        quads = pd.read_csv('quad_mapping.csv')
        quads_list = quads['device_name'].tolist()
        quads_list = [quad + ':BCTRL' for quad in quads_list]
        logger.info(f"Loaded {len(quads_list)} quads from quad_mapping.csv")
    except FileNotFoundError:
        logger.warning("quad_mapping.csv not found, using only additional quads")
        quads_list = []
    
    # Add additional quads
    quads_list.extend([
        'SOLN:IN20:121:BCTRL', 'SOLN:IN20:311:BCTRL', 'QUAD:IN20:121:BCTRL',
        'QUAD:IN20:122:BCTRL', 'QUAD:IN20:361:BCTRL', 'QUAD:IN20:371:BCTRL',
        'QUAD:IN20:425:BCTRL', 'QUAD:IN20:441:BCTRL', 'QUAD:IN20:511:BCTRL',
        'QUAD:IN20:525:BCTRL'
    ])
    
    # Combine all desired input columns
    desired_input_cols = (quads_list + RF_ampls + RF_phases + vcc_profile + 
                         undh_corr_x + undh_corr_y + undh_shifter + undh_gap)
    
    logger.info(f"Desired input features (from hardcoded lists): {len(desired_input_cols)}")
    
    # ---- Step 2: Check which desired columns actually exist ----
    available_cols = [c for c in desired_input_cols if c in final_df.columns]
    missing_cols = [c for c in desired_input_cols if c not in final_df.columns]
    
    logger.info(f"Available in data: {len(available_cols)}")
    if missing_cols:
        logger. warning(f"Missing from data: {len(missing_cols)} PVs")
        logger.debug(f"First 10 missing PVs: {missing_cols[: 10]}")
    
    # # ---- Step 3:  Manually specified always-invalid PVs ----
    # manually_invalid_pvs = [
        
    #     'QUAD:LI21:243:BCTRL', 'QUAD:LI24:713:BCTRL', 'QUAD:LI24:892:BCTRL',
    #     'QUAD:CLTH:140:BCTRL', 'QUAD:CLTH:170:BCTRL', 'QUAD:BSYH:445:BCTRL',
    #     'QUAD:LTUH:285:BCTRL', 'QUAD:LTUH:665:BCTRL', 'QUAD:DMPH:300:BCTRL',
    #     'QUAD:DMPH:380:BCTRL', 'QUAD:DMPH:500:BCTRL', 'QUAD:BSYH:465:BCTRL',
    #     'QUAD:BSYH:640:BCTRL', 'QUAD:BSYH:735:BCTRL', 'QUAD:BSYH:910:BCTRL',
    #     'QUAD:LTUH:110:BCTRL', 'QUAD:LTUH:120:BCTRL', 'QUAD:LTUH:180:BCTRL',
    #     'QUAD:LTUH:190:BCTRL', 'QUAD:LTUH:130:BCTRL', 'QUAD:LTUH:290:BCTRL',
    #     'QUAD:LTUH:250:BCTRL', 'QUAD:LTUH:720:BCTRL', 'QUAD:LTUH:820:BCTRL',
    #     'QUAD:UNDH:2780:BCTRL', 'QUAD:UNDH:2980:BCTRL', 'QUAD:UNDH:3080:BCTRL',
    #     'QUAD:UNDH:4580:BCTRL', 'QUAD:UNDH:4680:BCTRL', 'QUAD:LI24:701:BCTRL',
    #     'QUAD:LI24:601:BCTRL', 'QUAD:LI24:901:BCTRL', 'QUAD:LI25:201:BCTRL',
    #     'QUAD:IN20:631:BCTRL', 'QUAD:IN20:651:BCTRL', 'QUAD:IN20:731:BCTRL',
    #     'QUAD:LI21:315:BCTRL'
    # ]
    
    # logger.info(f"Manually specified invalid PVs: {len(manually_invalid_pvs)}")
    
    # ---- Step 4: Auto-detect low-variability PVs using PERCENTILE METHOD ----
    logger.info("Detecting low-variability PVs using percentile range method...")
    
    # Only check available columns
    cols_to_check = [c for c in available_cols if c in final_df.columns]
    
    # Run percentile-based detection
    variability_result = detect_low_variability_pvs_percentile(
        df=final_df,
        input_cols=cols_to_check,
        percentile_threshold=0.01,  # IQR < 1% of range
        absolute_threshold=1e-6      # Range < 1e-6 = constant
    )
    
    low_variance_cols = variability_result['low_variability']
    removal_reasons = variability_result['reasons']
    pv_stats = variability_result['stats']
    
    logger.info(f"Found {len(low_variance_cols)} low-variability PVs")
    
    # Log removed PVs with reasons
    if low_variance_cols:
        logger.info("All low-variability PVs and reasons:")
        for col in low_variance_cols[:]: 
            logger.info(f"  {col}: {removal_reasons[col]}")
    
    # ---- Step 5: Combine all invalid PVs ----
    # all_invalid_pvs = list(set(manually_invalid_pvs + low_variance_cols))
    logger.info(f"Total invalid PVs to remove: {len(low_variance_cols)}")
    
    # ---- Step 6: Final input columns (available - invalid) ----
    input_cols = [c for c in available_cols if c not in low_variance_cols]
    output_cols = ['hxr_pulse_intensity']
    
    logger.info("=" * 70)
    logger.info(f"FINAL FEATURE COUNT: {len(input_cols)}")
    logger.info("=" * 70)
    logger.info(f"  Desired features:          {len(desired_input_cols)}")
    logger.info(f"  - Missing from data:     {len(missing_cols)}")
    # logger.info(f"  - Manually invalid:      {len([c for c in manually_invalid_pvs if c in available_cols])}")
    logger.info(f"  - Auto-detected invalid: {len(low_variance_cols)}")
    logger.info(f"  = Final valid features:    {len(input_cols)}")
    logger.info("=" * 70)
    
    # ---- Step 7: Drop invalid PVs from dataframes ----
    cols_to_drop = [c for c in low_variance_cols if c in final_df.columns]
    if cols_to_drop:
        final_df = final_df.drop(columns=cols_to_drop)
        val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
        logger.info(f"Dropped {len(cols_to_drop)} invalid PVs from dataframes")
    
    # ---- Subsampling ----
    if args.subsample_step > 1:
        logger.info(f"Subsampling:  keeping 1 out of every {args.subsample_step} rows")
        final_df = final_df.iloc[::args.subsample_step]
        val_df = val_df.iloc[::args. subsample_step] if len(val_df) > 0 else val_df
    
    logger.info(f"Final training/test samples: {len(final_df)}")
    logger.info(f"Final validation samples:  {len(val_df)}")

    return final_df, val_df, input_cols, output_cols

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class FELNeuralNetwork(nn.Module):
    """
    Fully connected neural network for FEL intensity prediction.  
    
    Uses ELU activation with dropout for regularization.
    Output activation is Softplus to ensure positive intensity predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_dims: List[int] = None,
        dropout:  float = 0.05,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128, 64, 16, 16]
        
        layers = []
        prev_dim = input_size
        
        # Hidden layers
        for hidden_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            
            # Add dropout to certain layers
            if hidden_dim in [128, 64, 16]: 
                layers.append(nn. Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        # Output layer with Softplus to ensure positive values
        layers.append(nn.Linear(prev_dim, output_size))
        layers.append(nn.Softplus())  # Ensures output > 0 (physical for intensity)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class FELDataset(Dataset):
    """PyTorch Dataset for FEL model training."""
    
    def __init__(self, dataframe: pd.DataFrame, input_cols: List[str], output_cols: List[str]):
        self.features = dataframe[input_cols]. values. astype(np.float32)
        self.outputs = dataframe[output_cols]. values.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self. outputs[idx], dtype=torch. float32)
        return x, y

def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_cols: List[str],
    output_cols: List[str],
    batch_size: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = FELDataset(train_df, input_cols, output_cols)
    test_dataset = FELDataset(test_df, input_cols, output_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    checkpoint_path: str,
    input_cols: List[str] = None,  
):
    """Save training checkpoint with atomic writes."""
    checkpoint = {
        "epoch": epoch,
        "model":   model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "input_features": input_cols, 
        "n_features": len(input_cols) if input_cols else None, 
    }
    
    # Atomic save with temp file
    tmp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> int:
    """Load checkpoint and return starting epoch."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found:  {checkpoint_path}")
        return 0
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt. get("epoch", 0)
    
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch

def train_model(
    model: nn. Module,
    train_loader:  DataLoader,
    test_loader: DataLoader,
    criterion:  nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int = 50,
    device: torch.device = None,
    start_epoch: int = 0,
    ckpt_dir: Optional[str] = None,
    save_every: int = 30,
    input_cols: List[str] = None,  
) -> Tuple[List[float], List[float], Dict]: 
    """
    Train the FEL model.
    
    Args:
        model: Neural network model
        train_loader:  Training data loader
        test_loader: Test/validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Total number of epochs
        device: Device (cuda/cpu)
        start_epoch: Starting epoch (for resume)
        ckpt_dir: Directory for checkpoints
        save_every: Save snapshots every N epochs
        
    Returns:
        (train_losses, test_losses, best_model_state_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model_state = None
    
    t0 = time.time()
    
    logger.info("=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    logger.info(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    logger.info(f"Starting from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time() 
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= max(1, len(train_loader))
        
        # ---- Evaluation ----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_loss /= max(1, len(test_loader))
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # ---- Logging ----
        current_lr = optimizer.param_groups[0]['lr']
        
        # ETA calculation
        elapsed = time.time() - t0
        avg_time_per_epoch = elapsed / (epoch - start_epoch + 1)
        remaining_epochs = num_epochs - epoch - 1
        remaining_time = remaining_epochs * avg_time_per_epoch
        
        if remaining_time > 60: 
            eta_str = f"{remaining_time / 60:.1f} min"
        else:
            eta_str = f"{remaining_time:.0f} sec"
        epoch_time = time.time() - epoch_start 
        # Log message
        log_msg = (
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Time: {epoch_time:.1f}s | "
            f"ETA: {eta_str}"
        )
        
        # Highlight best epoch
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            logger.info(f"✓ {log_msg}")
        else:
            logger.info(log_msg)
        
        # Step scheduler
        scheduler.step(test_loss)
        
        # ---- Checkpointing ----
        if ckpt_dir: 
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Save last checkpoint
            last_ckpt_path = os.path.join(ckpt_dir, "last. pt")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, last_ckpt_path, input_cols) 
            
            # Save snapshots
            if (epoch + 1) % save_every == 0:
                snap_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch + 1, snap_path, input_cols)  

                logger.info(f"  → Saved snapshot:  {snap_path}")
            
            # Save best weights
            if test_loss == best_loss:
                best_path = os.path.join(ckpt_dir, "best_weights.pt")
                torch.save(best_model_state, best_path)

        # In train_model(), after each epoch:

        if device.type == 'cuda':
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            logger.debug(f"  GPU Memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
    
    elapsed_min = (time.time() - t0) / 60
    logger.info(f"Training completed in {elapsed_min:.1f} minutes")
    
    return train_losses, test_losses, best_model_state


def save_variable_config(
    input_cols: List[str],
    output_cols: List[str],
    train_df_unscaled: pd.DataFrame,  
    artifact_dir: str
):
    """
    Save input/output variable specifications for model deployment.
    
    Args:
        input_cols: List of input feature names
        output_cols: List of output feature names
        train_df_unscaled: UNSCALED training dataframe (original physical units)
        artifact_dir: Directory to save the configuration
    """
    logger.info("Saving variable configuration...")
    
    try:
        from lume_model.utils import variables_as_yaml
        from lume_model.variables import ScalarVariable
        
        input_variables = []
        for col in input_cols:
            # ✅ FIX: Use unscaled data to get physical ranges
            lower_bound, default_value, upper_bound = train_df_unscaled[col].quantile([0, 0.5, 1])
            input_variables.append(
                ScalarVariable(
                    name=col,
                    default_value=float(default_value),
                    value_range=[float(lower_bound), float(upper_bound)]
                )
            )
        
        output_variables = []
        for col in output_cols: 
            output_variables.append(ScalarVariable(name=col))
        
        config_path = os.path.join(artifact_dir, 'feature_config.yml')
        variables_as_yaml(input_variables, output_variables, config_path)
        logger.info(f"  ✓ Variable config:  {config_path}")
        
        return True
    
    except ImportError:
        logger.warning("lume_model not available; skipping variable config")
        return False


def save_training_config(
    args: argparse.Namespace,
    input_cols: List[str],
    output_cols: List[str],
    train_df: pd.DataFrame,
    artifact_dir: str,
    additional_info: Optional[Dict] = None
):
    """
    Save complete training configuration for reproducibility.
    
    This saves all arguments, hyperparameters, and metadata needed to
    reproduce the training run or understand the model configuration.
    
    Args:
        args: Command-line arguments
        input_cols:  List of input feature names
        output_cols: List of output feature names
        train_df: Training dataframe (for statistics)
        artifact_dir: Directory to save configuration
        additional_info: Optional dict with extra metadata
    """
    logger.info("Saving training configuration...")
    
    # Compute input statistics for reference
    input_stats = {}
    for col in input_cols[: 10]:  # Save stats for first 10 features as example
        input_stats[col] = {
            'min': float(train_df[col].min()),
            'max': float(train_df[col].max()),
            'mean': float(train_df[col].mean()),
            'std': float(train_df[col].std())
        }
    
    config = {
        # Training arguments
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'subsample_step': args.subsample_step,
            'save_every': args.save_every,
        },
        
        # Model architecture
        'model_architecture': {
            'type': 'FELNeuralNetwork',
            'input_size': len(input_cols),
            'output_size': len(output_cols),
            'hidden_dims': [512, 512, 256, 128, 64, 16, 16],
            'dropout': 0.05,
            'output_activation': 'Softplus'
        },
        
        # Optimizer settings
        'optimizer': {
            'type': 'Adam',
            'learning_rate': 1e-6,
            'weight_decay':  1e-4
        },
        
        # Scheduler settings
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.8,
            'patience': 4
        },
        
        # Data information
        'data_info': {
            'n_train_samples': len(train_df),
            'n_input_features': len(input_cols),
            'n_output_features': len(output_cols),
            'input_features': input_cols,
            'output_features': output_cols,
            'input_stats_sample': input_stats,
        },
        
        # Metadata
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'user': os.environ.get('USER', 'unknown'),
            'hostname': os.environ.get('HOSTNAME', 'unknown'),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),
        }
    }
    
    # Add any additional info
    if additional_info:
        config['additional_info'] = additional_info
    
    # Save as JSON
    config_path = os.path.join(artifact_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"  ✓ Training config: {config_path}")
    
    # Also save a human-readable version
    readme_path = os.path.join(artifact_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEL NEURAL NETWORK MODEL - TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date: {config['metadata']['timestamp']}\n")
        f.write(f"User: {config['metadata']['user']}\n")
        f.write(f"SLURM Job ID: {config['metadata']['slurm_job_id']}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"  Input size: {config['model_architecture']['input_size']}\n")
        f.write(f"  Hidden layers: {config['model_architecture']['hidden_dims']}\n")
        f.write(f"  Output size: {config['model_architecture']['output_size']}\n")
        f.write(f"  Dropout: {config['model_architecture']['dropout']}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Epochs: {config['training_args']['epochs']}\n")
        f.write(f"  Batch size:  {config['training_args']['batch_size']}\n")
        f.write(f"  Learning rate: {config['optimizer']['learning_rate']}\n")
        f.write(f"  Weight decay: {config['optimizer']['weight_decay']}\n\n")
        
        f.write("DATA:\n")
        f.write(f"  Training samples: {config['data_info']['n_train_samples']}\n")
        f.write(f"  Input features: {config['data_info']['n_input_features']}\n")
        f.write(f"  Output features: {config['data_info']['n_output_features']}\n\n")
        
        f.write("FILES IN THIS DIRECTORY:\n")
        f.write("  - best_model.pt: Best model weights (lowest validation loss)\n")
        f.write("  - final_model.pt: Final model weights after all epochs\n")
        f.write("  - input_scaler.pt: Input normalization scaler\n")
        f.write("  - output_scaler.pt: Output normalization scaler\n")
        f.write("  - feature_config.yml: LUME-model variable specifications\n")
        f.write("  - training_config.json: Complete training configuration\n")
        f.write("  - train_losses.npy: Training loss history\n")
        f.write("  - test_losses.npy: Test loss history\n")
        f.write("  - README.txt: This file\n")
    
    logger.info(f"  ✓ README: {readme_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("=" * 70)
    logger.info("FEL NEURAL NETWORK MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # Setup checkpoint directory
    ckpt_dir = setup_checkpoint_dir(args.checkpoint_dir)

    input_cols_override = None
    
    # Only apply feature override for model_path (fine-tuning/transfer learning)
    if args.model_path and not args.resume_from:
        logger.info("=" * 70)
        logger.info("🔄 FINE-TUNING MODE:   Loading from pre-trained model")
        logger.info("=" * 70)
        
        model_dir = os.path.dirname(args.model_path)
        logger.info(f"Model path provided: {args.model_path}")
        
        # Try to find feature_config.yml
        config_candidates = [
            os.path. join(model_dir, 'feature_config.yml'),
            os.path.join(model_dir, 'model_config.yml'),
        ]
        
        config_path = None
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break
        
        if config_path:
            logger.info(f"✓ Found feature config:  {config_path}")
            try:
                from lume_model.utils import variables_from_yaml
                input_variables, output_variables = variables_from_yaml(config_path)
                input_cols_override = [v. name for v in input_variables]
                logger.info(f"✓ Loaded {len(input_cols_override)} input features from config")
            except Exception as e: 
                logger.error(f"❌ Failed to load feature config: {e}")
                logger.error("   Cannot proceed with fine-tuning - feature set unknown")
                exit(1)
        else:
            logger.error("❌ No feature_config.yml found for fine-tuning!")
            logger.error("   Searched in:")
            for candidate in config_candidates:
                logger.error(f"     - {candidate}")
            logger.error("\n💡 SOLUTION:  Provide the model directory containing feature_config.yml")
            exit(1)
        
        logger.info("=" * 70)
    
    elif args.resume_from:
        logger.info("=" * 70)
        logger.info("🔄 RESUME MODE:  Continuing from checkpoint")
        logger.info("=" * 70)
        logger.info(f"Checkpoint: {args.resume_from}")
        logger.info("Will use same feature detection as original run")
        logger.info("=" * 70)
        # No feature override - let data preprocessing run normally
    
  
    final_df, val_df, input_cols, output_cols = load_and_preprocess_data(args, input_cols_override)
    
    # Train/test split
    logger.info("\nPerforming train/test split (80/20)...")
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=39)
    
   
    train_df_unscaled = train_df[input_cols + output_cols].copy()
    
    # Drop constant inputs detected in training set
    logger.info("Removing constant inputs in training set...")
    train_rng = train_df[input_cols].max() - train_df[input_cols].min()
    const_inputs = train_rng[train_rng == 0].index.tolist()
    
    if const_inputs:
        logger.info(f"Dropping {len(const_inputs)} constant PVs")
        for c in const_inputs[: 10]: 
            logger.debug(f"  {c}")
        if len(const_inputs) > 10:
            logger.debug(f"  ...and {len(const_inputs) - 10} more")
        
        train_df = train_df.drop(columns=const_inputs)
        test_df = test_df.drop(columns=const_inputs)
        val_df = val_df.drop(columns=const_inputs, errors='ignore')
        input_cols = [c for c in input_cols if c not in const_inputs]
    
    # Fit scalers ONLY on training data (avoid leakage)
    logger.info("Fitting scalers on training data only...")
    input_mins = train_df[input_cols].min()
    input_maxs = train_df[input_cols].max()
    output_mins = train_df[output_cols].min()
    output_maxs = train_df[output_cols].max()
    
    input_scaler = AffineInputTransform(
        d=len(input_cols),
        coefficient=torch.tensor((input_maxs - input_mins).values, dtype=torch.float32),
        offset=torch.tensor(input_mins.values, dtype=torch.float32),
    )
    output_scaler = AffineInputTransform(
        d=len(output_cols),
        coefficient=torch.tensor((output_maxs - output_mins).values, dtype=torch.float32),
        offset=torch.tensor(output_mins.values, dtype=torch.float32),
    )
    
    # Apply scaling
    logger.info("Applying scaling to all datasets...")
    train_df.loc[:, input_cols] = input_scaler.transform(
        torch.tensor(train_df[input_cols].to_numpy(dtype=np.float32))
    ).numpy()
    test_df.loc[:, input_cols] = input_scaler.transform(
        torch.tensor(test_df[input_cols].to_numpy(dtype=np.float32))
    ).numpy()
    train_df.loc[:, output_cols] = output_scaler.transform(
        torch.tensor(train_df[output_cols].to_numpy(dtype=np.float32))
    ).numpy()
    test_df.loc[: , output_cols] = output_scaler.transform(
        torch.tensor(test_df[output_cols].to_numpy(dtype=np.float32))
    ).numpy()
    
    if len(val_df) > 0:
        val_df.loc[:, input_cols] = input_scaler.transform(
            torch.tensor(val_df[input_cols].to_numpy(dtype=np.float32))
        ).numpy()
        val_df.loc[:, output_cols] = output_scaler.transform(
            torch.tensor(val_df[output_cols].to_numpy(dtype=np.float32))
        ).numpy()
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_df, test_df, input_cols, output_cols, batch_size=args.batch_size
    )
    
    # Create model
    logger.info("\n" + "=" * 70)
    logger.info("MODEL CREATION")
    logger.info("=" * 70)
    model = FELNeuralNetwork(
        input_size=len(input_cols),
        output_size=len(output_cols),
        hidden_dims=[512, 512, 256, 128, 64, 16, 16],
        dropout=0.05,
    ).to(device)

    logger.info(f"Model created: {len(input_cols)} inputs → {len(output_cols)} output") 

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")    
    # Load pre-trained weights if provided
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading pre-trained model:  {args.model_path}")
        try:
            # Load the Sequential model (not state_dict)
            pretrained_model = torch.load(
                args.model_path, 
                map_location=device,
                weights_only=False  # ← Explicitly allow non-weights objects
            )
            
            # If it's a Sequential, copy into our model's . net
            if isinstance(pretrained_model, nn. Sequential):
                model.net.load_state_dict(pretrained_model.state_dict())
                logger.info("✓ Pre-trained Sequential model loaded")
            # If it's a state dict, load directly
            elif isinstance(pretrained_model, dict):
                model.load_state_dict(pretrained_model)
                logger.info("✓ Pre-trained state dict loaded")
            else:
                logger. warning(f"Unexpected model type: {type(pretrained_model)}")
                logger.warning("Training from scratch.")
        except Exception as e: 
            logger.warning(f"Failed to load model:  {e}.  Training from scratch.")
    
    # Create optimizer and scheduler
    lr = 1e-5 #5e-6 #1e-5
    weight_decay = 1e-4 #1e-6
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=4)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
    # In main(), before training: 
    
    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 70)
    logger.info(f"Learning rate:      {lr}")
    logger.info(f"Weight decay:      {weight_decay}")
    logger.info(f"Batch size:         {args.batch_size}")
    logger.info(f"Loss function:     {criterion.__class__.__name__}")
    logger.info(f"Optimizer:         {optimizer.__class__.__name__}")
    logger.info(f"LR scheduler:      {scheduler.__class__.__name__}")
    logger.info(f"Scheduler patience: 4")
    logger.info(f"Scheduler factor:  0.8")
    logger.info("=" * 70)
    # Train
    train_losses, test_losses, best_model_state = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, scheduler,
        num_epochs=args.epochs,
        device=device,
        start_epoch=start_epoch,
        ckpt_dir=ckpt_dir,
        save_every=args.save_every,
        input_cols=input_cols,  
    )
    
    # Save artifacts
    logger.info("\n" + "=" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    model_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/model/'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    artifact_dir = os.path.join(model_dir, f"{timestamp}_nn")
    os.makedirs(artifact_dir, exist_ok=True)
    
    best_model_path = os.path.join(artifact_dir, 'best_model.pt')
    final_model_path = os.path.join(artifact_dir, 'final_model.pt')
    input_scaler_path = os.path.join(artifact_dir, 'input_scaler.pt')
    output_scaler_path = os.path.join(artifact_dir, 'output_scaler.pt')
    train_losses_path = os.path.join(artifact_dir, 'train_losses.npy')
    test_losses_path = os.path.join(artifact_dir, 'test_losses.npy')
    
    torch.save(best_model_state, best_model_path)
    torch.save(model.net, final_model_path)
    torch.save(input_scaler, input_scaler_path)
    torch.save(output_scaler, output_scaler_path)
    np.save(train_losses_path, np.array(train_losses))
    np.save(test_losses_path, np.array(test_losses))
    
    logger.info(f"\nArtifacts saved to: {artifact_dir}")
    logger.info(f"  ✓ Best model: {best_model_path}")
    logger.info(f"  ✓ Final model: {final_model_path}")
    logger.info(f"  ✓ Input scaler: {input_scaler_path}")
    logger.info(f"  ✓ Output scaler: {output_scaler_path}")
    logger.info(f"  ✓ Train losses: {train_losses_path}")
    logger.info(f"  ✓ Test losses: {test_losses_path}")
    
    save_variable_config(input_cols, output_cols, train_df_unscaled, artifact_dir)
    
    additional_info = {
        'best_test_loss': float(min(test_losses)),
        'final_test_loss': float(test_losses[-1]),
        'n_test_samples': len(test_df),
        'n_validation_samples': len(val_df),
    }
    save_training_config(args, input_cols, output_cols, train_df, artifact_dir, additional_info)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
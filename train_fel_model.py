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
import logging
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
from lume_model.utils import variables_from_yaml, variables_as_yaml
from lume_model.variables import ScalarVariable
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
        # ⭐ PARSE FIRST
    args = parser.parse_args()
    
    # ⭐ ADD VALIDATION
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
        'XRMS 250-370': (dataset['CAMR:IN20:186:XRMS'] > 250) & (dataset['CAMR:IN20:186:XRMS'] < 380),
        'YRMS 250-370': (dataset['CAMR:IN20:186:YRMS'] > 250) & (dataset['CAMR:IN20:186:YRMS'] < 380),
        'Intensity 0.1-4': (dataset['GDET:FEE1:241:ENRC'] > 0.1) & (dataset['GDET:FEE1:241:ENRC'] < 4),
        'Charge at gun 240-275': (dataset['SIOC:SYS0:ML00:CALC038'] > 240) & (dataset['SIOC:SYS0:ML00:CALC038'] < 275),
        'Charge after BC1 < 200': dataset['SIOC:SYS0:ML00:CALC252'] < 200,
        'HXR e-energy > 8': dataset['BEND:DMPH:400:BACT'] > 8,
        'HXR photon > 7000': dataset['SIOC:SYS0:ML00:AO627'] > 7000,
        # 'HXR photon 9.5keV-10keV': (dataset['SIOC:SYS0:ML00:AO627'] > 9500) & (dataset['SIOC:SYS0:ML00:AO627']<10000),
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
    
    XCOR_EXCLUSIONS = ['XCOR:UNDH:4780:BACT']
    YCOR_EXCLUSIONS = []
    
    # ---- Undulator X Correctors ----
    xcor_cols = [col for col in dataset.columns 
                 if 'XCOR:UNDH:' in col 
                 and ':BACT' in col 
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
                 and ':BACT' in col 
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
    phas_cols = [col for col in dataset.columns if 'PHAS:UNDH:' in col and ':GapAct' in col]
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
    useg_cols = [col for col in dataset.columns if 'USEG:UNDH:' in col and ':GapAct' in col]
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

def clip_outliers_percentile(df, columns, lower_pct=1, upper_pct=99, min_samples=100):
    """
    Remove outliers based on percentiles.
    
    Args:
        df: DataFrame
        columns: List of column names to clip
        lower_pct: Lower percentile threshold (default: 1 = P1)
        upper_pct: Upper percentile threshold (default: 99 = P99)
        min_samples: Minimum samples required per PV
        
    Returns:
        df_cleaned: DataFrame with outliers removed
        outlier_report: Dict with statistics per PV
    """
    df_cleaned = df.copy()
    outlier_report = {}
    total_rows_original = len(df)
    
    combined_mask = pd.Series(True, index=df.index)
    
    logger.info(f"Filtering {len(columns)} columns for outliers...")
    logger.info(f"Method: Percentile range [P{lower_pct}, P{upper_pct}]")
    
    for col in columns:
        if col not in df.columns:
            continue
        
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) < min_samples:
            logger.debug(f"{col}: Too few samples ({len(data)}), skipping")
            continue
        
        # Calculate percentile bounds
        lower_bound = data.quantile(lower_pct / 100)
        upper_bound = data.quantile(upper_pct / 100)
        
        # Create mask
        valid_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        combined_mask &= valid_mask
        
        # Count outliers
        n_outliers = (~valid_mask).sum()
        outlier_pct = (n_outliers / len(df)) * 100
        
        outlier_report[col] = {
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
        }
        
        if n_outliers > 0:
            logger.debug(f"{col}: {n_outliers} outliers ({outlier_pct:.2f}%) "
                        f"outside [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Apply combined mask
    df_cleaned = df[combined_mask].copy()
    
    total_removed = total_rows_original - len(df_cleaned)
    removal_pct = (total_removed / total_rows_original) * 100
    
    logger.info("=" * 70)
    logger.info(f"PERCENTILE OUTLIER FILTERING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Range: P{lower_pct} to P{upper_pct}")
    logger.info(f"Original samples: {total_rows_original:,}")
    logger.info(f"Cleaned samples:  {len(df_cleaned):,}")
    logger.info(f"Removed samples:  {total_removed:,} ({removal_pct:.2f}%)")
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
    
    file_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/dataset_updated/'
    pickle_files = [
        # '20260218_MD_1st.pkl',
        'hxr_archiver_2026-02.pkl', 'hxr_archiver_2026-03.pkl',
        'hxr_archiver_2026-01.pkl', 'hxr_archiver_2025-12.pkl',
        'hxr_archiver_2025-11.pkl', 'hxr_archiver_2025-10.pkl', 'hxr_archiver_2025-09.pkl',
        'hxr_archiver_2025-06.pkl', 'hxr_archiver_2025-05.pkl', 'hxr_archiver_2025-04.pkl',
        'hxr_archiver_2025-03.pkl', 'hxr_archiver_2025-02.pkl', 'hxr_archiver_2025-01.pkl',
        # 'hxr_archiver_2024-12.pkl', 'hxr_archiver_2024-11.pkl', 'hxr_archiver_2024-10.pkl',
        # 'hxr_archiver_2024-09.pkl', 'hxr_archiver_2024-08.pkl', 'hxr_archiver_2024-07.pkl',
        # 'hxr_archiver_2024-06.pkl', 'hxr_archiver_2024-05.pkl', 'hxr_archiver_2024-04.pkl',
        # 'hxr_archiver_2024-03.pkl', 'hxr_archiver_2024-02.pkl', 'hxr_archiver_2024-01.pkl',
        # 'hxr_archiver_2023-11.pkl', 'hxr_archiver_2023-10.pkl', 'hxr_archiver_2023-09.pkl',
        # 'hxr_archiver_2023-08.pkl', 'hxr_archiver_2023-07.pkl',
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
            
            # ⭐ Apply filter with diagnostics
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
        # ("2024-11-21 08:30", "2024-11-21 18:00"),
        # ("2024-11-12 16:00", "2024-11-13 02:00"),
        # ("2024-11-06 07:30", "2024-11-06 15:30"),
        # ("2024-10-15 07:00", "2024-10-16 08:00"),
        # ("2024-09-04 21:00", "2024-09-05 15:30"),
        # ("2024-06-06 20:30", "2024-06-07 04:30"),
        # ("2024-05-09 15:00", "2024-05-09 22:00"),
        # ("2024-03-28 10:00", "2024-03-29 02:00"),
        # ("2024-03-20 17:00", "2024-03-21 01:30"),
        # ("2024-02-14 19:00", "2024-02-15 03:00"),
        # ("2023-11-16 08:00", "2023-11-16 17:00"),
        # ("2023-11-09 16:00", "2023-11-10 04:00"),
        # ("2023-11-01 13:00", "2023-11-01 22:00"),
        # ("2023-10-05 09:00", "2023-10-06 05:00"),
        # ("2023-09-27 21:00", "2023-09-28 03:00"),
        # ("2023-09-21 09:00", "2023-09-21 19:00"),
        # ("2023-08-30 06:00", "2023-08-30 18:00"),
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
        ("2026-02-28 00:00", "2026-03-03 00:00"),
        ("2026-01-28 00:00", "2026-02-03 00:00"),
        ("2025-11-28 00:00", "2025-12-03 00:00"),
        ("2025-10-28 00:00", "2025-11-03 00:00"),
        ("2025-09-28 00:00", "2025-10-03 00:00"),
        ("2025-08-28 00:00", "2025-09-03 00:00"),
        ("2025-06-28 00:00", "2025-07-03 00:00"),
        ("2025-05-28 00:00", "2025-06-03 00:00"),
        ("2025-04-28 00:00", "2025-05-03 00:00"),
        ("2025-03-28 00:00", "2025-04-03 00:00"),
        ("2025-02-28 00:00", "2025-03-03 00:00"),
        # ("2024-11-28 00:00", "2024-12-03 00:00"),
        # ("2024-10-28 00:00", "2024-11-03 00:00"),
        # ("2024-09-28 00:00", "2024-10-03 00:00"),
        # ("2024-08-28 00:00", "2024-09-03 00:00"),
        # ("2024-06-28 00:00", "2024-07-03 00:00"),
        # ("2024-05-28 00:00", "2024-06-03 00:00"),
        # ("2024-04-28 00:00", "2024-05-03 00:00"),
        # ("2024-03-28 00:00", "2024-04-03 00:00"),
        # ("2024-02-28 00:00", "2024-03-03 00:00"),
        # ("2023-10-28 00:00", "2023-11-03 00:00"),
        # ("2023-09-28 00:00", "2023-10-03 00:00"),
        # ("2023-08-28 00:00", "2023-09-03 00:00"),
        # ("2023-07-28 00:00", "2023-08-03 00:00"),
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
    # ⭐ OUTLIER FILTERING (Apply for BOTH FRESH and RETRAIN!)
    # ============================================================================
    
    # ⭐ ALWAYS apply P1-P99 filtering (for consistency)
    logger.info("\n" + "=" * 70)
    logger.info("OUTLIER FILTERING (PERCENTILE METHOD)")
    logger.info("=" * 70)
    
    if input_cols_override is None:  # FRESH training mode
        logger.info("Mode: FRESH TRAINING - Applying outlier filtering")
    else:  # RETRAIN mode
        logger.info("Mode: RETRAIN - Applying outlier filtering")
        logger.info("  (Ensuring data distribution matches original training)")
    
    # Get all numeric columns except output
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_filter = [c for c in numeric_cols if c not in ['GDET:FEE1:241:ENRC']]
    
    logger.info(f"Filtering {len(cols_to_filter)} numeric columns (P1-P99 range)...")
    
    original_count = len(final_df)
    final_df, outlier_report = clip_outliers_percentile(
        final_df, 
        columns=cols_to_filter,
        lower_pct=0,
        upper_pct=100,
        min_samples=100
    )
    
    filtered_count = len(final_df)
    logger.info(f"\n✓ Outlier filtering complete:")
    logger.info(f"  Samples: {original_count:,} → {filtered_count:,} "
               f"({(1 - filtered_count/original_count)*100:.2f}% removed)")
    
    # Save outlier report
    if ckpt_dir := args.checkpoint_dir or os.environ.get("SCRATCH"):
        os.makedirs(ckpt_dir, exist_ok=True)
        import json
        report_path = os.path.join(ckpt_dir, 'outlier_report.json')
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            serializable_report = {
                k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                    for kk, vv in v.items()}
                for k, v in outlier_report.items()
            }
            json.dump(serializable_report, f, indent=2)
        logger.info(f"  Outlier report saved: {report_path}")
    
    
    else:
        logger.info("\n" + "=" * 70)
        logger.info("OUTLIER FILTERING")
        logger.info("=" * 70)
        logger.info("Mode: RETRAIN - Skipping outlier filtering")
        logger.info("  (Using same data distribution as original training)")
        logger.info("=" * 70)
    # ============================================================================
    # FEATURE SELECTION & INVALID PV REMOVAL
    # ============================================================================
    # In load_and_preprocess_data(), in the RETRAINING MODE section:
    if input_cols_override is not None:
        # RETRAINING MODE: Use pre-defined feature set
        logger.info("=" * 70)
        logger.info("RETRAINING MODE: Using pre-defined feature set")
        logger.info("=" * 70)
        logger.info(f"Loading {len(input_cols_override)} input features from pre-trained model")
        
        input_cols = input_cols_override
        output_cols = ['GDET:FEE1:241:ENRC']
        
        # Check which features are available
        available_features = [c for c in input_cols if c in final_df.columns]
        missing_features = [c for c in input_cols if c not in final_df.columns]
        
        if missing_features:
            logger.error(f"❌ Missing {len(missing_features)} required features from pre-trained model!")
            logger.error(f"\nMissing features:")
            for feat in missing_features[:10]:
                logger.error(f"     - {feat}")
            if len(missing_features) > 10:
                logger.error(f"     ... and {len(missing_features)-10} more")
            
            logger.error(f"\n💡 SOLUTIONS:")
            logger.error(f"   1. Use data from the same time period as original training")
            logger.error(f"   2. Check if PV names changed")
            logger.error(f"   3. Train from scratch on new data (remove --model_path)")
            
            raise ValueError(f"Cannot retrain: {len(missing_features)} required features missing")
        
        logger.info(f"✓ All {len(input_cols)} required features present in data")
        
        # Check for constant features (warning only, don't drop)
        logger.info("\nChecking for constant features in new data...")
        feature_ranges = final_df[input_cols].max() - final_df[input_cols].min()
        constant_features = feature_ranges[feature_ranges == 0].index.tolist()
        
        if constant_features:
            logger.warning(f"⚠️  {len(constant_features)} features are constant in new data:")
            for feat in constant_features[:10]:
                logger.warning(f"     - {feat}")
            if len(constant_features) > 10:
                logger.warning(f"     ... and {len(constant_features)-10} more")
            logger.warning("\n   These features were NOT constant in original training.")
            logger.warning("   They will be kept to match pre-trained model architecture.")
            logger.warning("   This may indicate data distribution shift!")
        else:
            logger.info("✓ No constant features detected")
        
        logger.info("=" * 70)
        
        # Keep ALL features from pre-trained model
        cols_to_keep = input_cols + output_cols
        final_df = final_df[cols_to_keep].copy()
        if len(val_df) > 0:
            val_df = val_df[cols_to_keep].copy()
        
        return final_df, val_df, input_cols, output_cols
    
    # ⭐ FRESH TRAINING MODE: Detect features from data
    logger.info("=" * 70)
    logger.info("FRESH TRAINING MODE: Detecting features from data")
    logger.info("=" * 70)

    
    # ---- Step 1:Define desired input features (hardcoded lists) ----
    RF_ampls = ['ACCL:LI21:1:L1S_S_AV', 'ACCL:LI21:180:L1X_S_AV', 'ACCL:LI22:1:ADES', 'ACCL:LI25:1:ADES']
    RF_phases = ['ACCL:LI21:1:L1S_S_PV', 'ACCL:LI21:180:L1X_S_PV', 'ACCL:LI22:1:PDES', 'ACCL:LI25:1:PDES']
    vcc_profile = ['CAMR:IN20:186:XRMS', 'CAMR:IN20:186:YRMS']
    blen = ['BLEN:LI21:265:AIMAX1H', 'BLEN:LI24:886:BIMAX1H']
    bcharge = ['SIOC:SYS0:ML00:CALC038', 'SIOC:SYS0:ML00:CALC252']  # at gun, after BC1
    hxr_energy = ['BEND:DMPH:400:BACT','SIOC:SYS0:ML00:AO627']  # beam energy, photon energy
    undh_corr_x = [
        'XCOR:UNDH:1380:BACT', 'XCOR:UNDH:1480:BACT', 'XCOR:UNDH:1580:BACT', 'XCOR:UNDH:1680:BACT',
        'XCOR:UNDH:1780:BACT', 'XCOR:UNDH:1880:BACT', 'XCOR:UNDH:1980:BACT', 'XCOR:UNDH:2080:BACT',
        'XCOR:UNDH:2180:BACT', 'XCOR:UNDH:2280:BACT', 'XCOR:UNDH:2380:BACT', 'XCOR:UNDH:2480:BACT',
        'XCOR:UNDH:2580:BACT', 'XCOR:UNDH:2680:BACT', 'XCOR:UNDH:2780:BACT', 'XCOR:UNDH:2880:BACT',
        'XCOR:UNDH:2980:BACT', 'XCOR:UNDH:3080:BACT', 'XCOR:UNDH:3180:BACT', 'XCOR:UNDH:3280:BACT',
        'XCOR:UNDH:3380:BACT', 'XCOR:UNDH:3480:BACT', 'XCOR:UNDH:3580:BACT', 'XCOR:UNDH:3680:BACT',
        'XCOR:UNDH:3780:BACT', 'XCOR:UNDH:3880:BACT', 'XCOR:UNDH:3980:BACT', 'XCOR:UNDH:4080:BACT',
        'XCOR:UNDH:4180:BACT', 'XCOR:UNDH:4280:BACT', 'XCOR:UNDH:4380:BACT', 'XCOR:UNDH:4480:BACT',
        'XCOR:UNDH:4580:BACT', 'XCOR:UNDH:4680:BACT', 'XCOR:UNDH:4780:BACT'
    ]
    
    undh_corr_y = [
        'YCOR:UNDH:1380:BACT', 'YCOR:UNDH:1480:BACT', 'YCOR:UNDH:1580:BACT', 'YCOR:UNDH:1680:BACT',
        'YCOR:UNDH:1780:BACT', 'YCOR:UNDH:1880:BACT', 'YCOR:UNDH:1980:BACT', 'YCOR:UNDH:2080:BACT',
        'YCOR:UNDH:2180:BACT', 'YCOR:UNDH:2280:BACT', 'YCOR:UNDH:2380:BACT', 'YCOR:UNDH:2480:BACT',
        'YCOR:UNDH:2580:BACT', 'YCOR:UNDH:2680:BACT', 'YCOR:UNDH:2780:BACT', 'YCOR:UNDH:2880:BACT',
        'YCOR:UNDH:2980:BACT', 'YCOR:UNDH:3080:BACT', 'YCOR:UNDH:3180:BACT', 'YCOR:UNDH:3280:BACT',
        'YCOR:UNDH:3380:BACT', 'YCOR:UNDH:3480:BACT', 'YCOR:UNDH:3580:BACT', 'YCOR:UNDH:3680:BACT',
        'YCOR:UNDH:3780:BACT', 'YCOR:UNDH:3880:BACT', 'YCOR:UNDH:3980:BACT', 'YCOR:UNDH:4080:BACT',
        'YCOR:UNDH:4180:BACT', 'YCOR:UNDH:4280:BACT', 'YCOR:UNDH:4380:BACT', 'YCOR:UNDH:4480:BACT',
        'YCOR:UNDH:4580:BACT', 'YCOR:UNDH:4680:BACT', 'YCOR:UNDH:4780:BACT'
    ]
    
    undh_shifter = [
        'PHAS:UNDH:1495:GapAct', 'PHAS:UNDH:1595:GapAct', 'PHAS:UNDH:1695:GapAct', 'PHAS:UNDH:1795:GapAct',
        'PHAS:UNDH:1895:GapAct', 'PHAS:UNDH:1995:GapAct', 'PHAS:UNDH:2095:GapAct', 'PHAS:UNDH:2295:GapAct',
        'PHAS:UNDH:2395:GapAct', 'PHAS:UNDH:2495:GapAct', 'PHAS:UNDH:2595:GapAct', 'PHAS:UNDH:2695:GapAct',
        'PHAS:UNDH:2795:GapAct', 'PHAS:UNDH:2995:GapAct', 'PHAS:UNDH:3095:GapAct', 'PHAS:UNDH:3195:GapAct',
        'PHAS:UNDH:3295:GapAct', 'PHAS:UNDH:3395:GapAct', 'PHAS:UNDH:3495:GapAct', 'PHAS:UNDH:3595:GapAct',
        'PHAS:UNDH:3695:GapAct', 'PHAS:UNDH:3795:GapAct', 'PHAS:UNDH:3895:GapAct', 'PHAS:UNDH:3995:GapAct',
        'PHAS:UNDH:4095:GapAct', 'PHAS:UNDH:4195:GapAct', 'PHAS:UNDH:4295:GapAct', 'PHAS:UNDH:4395:GapAct',
        'PHAS:UNDH:4495:GapAct', 'PHAS:UNDH:4595:GapAct', 'PHAS:UNDH:4695:GapAct'
    ]
    
    undh_gap = [
        'USEG:UNDH:1450:GapAct', 'USEG:UNDH:1550:GapAct', 'USEG:UNDH:1650:GapAct', 'USEG:UNDH:1750:GapAct',
        'USEG:UNDH:1850:GapAct', 'USEG:UNDH:1950:GapAct', 'USEG:UNDH:2050:GapAct', 'USEG:UNDH:2250:GapAct',
        'USEG:UNDH:2350:GapAct', 'USEG:UNDH:2450:GapAct', 'USEG:UNDH:2550:GapAct', 'USEG:UNDH:2650:GapAct',
        'USEG:UNDH:2750:GapAct', 'USEG:UNDH:2950:GapAct', 'USEG:UNDH:3050:GapAct', 'USEG:UNDH:3150:GapAct',
        'USEG:UNDH:3250:GapAct', 'USEG:UNDH:3350:GapAct', 'USEG:UNDH:3450:GapAct', 'USEG:UNDH:3550:GapAct',
        'USEG:UNDH:3650:GapAct', 'USEG:UNDH:3750:GapAct', 'USEG:UNDH:3850:GapAct', 'USEG:UNDH:3950:GapAct',
        'USEG:UNDH:4050:GapAct', 'USEG:UNDH:4150:GapAct', 'USEG:UNDH:4250:GapAct', 'USEG:UNDH:4350:GapAct',
        'USEG:UNDH:4450:GapAct', 'USEG:UNDH:4550:GapAct', 'USEG:UNDH:4650:GapAct', 'USEG:UNDH:4750:GapAct'
    ]
    
    # Load quadrupoles from CSV
    try:
        quads = pd.read_csv('quad_mapping.csv')
        quads_list = quads['device_name'].tolist()
        quads_list = [quad + ':BACT' for quad in quads_list]
        logger.info(f"Loaded {len(quads_list)} quads from quad_mapping.csv")
    except FileNotFoundError:
        logger.warning("quad_mapping.csv not found, using only additional quads")
        quads_list = []
    
    # Add additional quads
    quads_list.extend([
        'SOLN:IN20:121:BACT', 'SOLN:IN20:311:BACT', 'QUAD:IN20:121:BACT',
        'QUAD:IN20:122:BACT', 'QUAD:IN20:361:BACT', 'QUAD:IN20:371:BACT',
        'QUAD:IN20:425:BACT', 'QUAD:IN20:441:BACT', 'QUAD:IN20:511:BACT',
        'QUAD:IN20:525:BACT'
    ])
    # Combine all desired input columns
    desired_input_cols = (quads_list + RF_ampls + RF_phases + vcc_profile + blen + bcharge + 
                         undh_corr_x + undh_corr_y + undh_shifter + undh_gap)
    
    logger.info(f"Desired input features (from hardcoded lists): {len(desired_input_cols)}")
    
    # ---- Step 2: Check which desired columns actually exist ----
    available_cols = [c for c in desired_input_cols if c in final_df.columns]
    missing_cols = [c for c in desired_input_cols if c not in final_df.columns]
    
    logger.info(f"Available in data: {len(available_cols)}")
    if missing_cols:
        logger.warning(f"Missing from data: {len(missing_cols)} PVs")
        logger.warning(f"First 10 missing PVs: {missing_cols[: 10]}")
    
    
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
    output_cols = ['GDET:FEE1:241:ENRC']
    
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
    input_cols: List[str] = None,  # ⭐ ADD THIS
):
    """Save training checkpoint with atomic writes."""
    checkpoint = {
        "epoch": epoch,
        "model":   model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "input_features": input_cols,  # ⭐ ADD THIS
        "n_features": len(input_cols) if input_cols else None,  # ⭐ ADD THIS
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
    input_cols: List[str] = None,  # ⭐ ADD THIS
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
        epoch_start = time.time()  # ⭐ ADD
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
        epoch_time = time.time() - epoch_start  # ⭐ ADD
        # Log message
        log_msg = (
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Time: {epoch_time:.1f}s | "  # ⭐ ADD
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
            save_checkpoint(model, optimizer, scheduler, epoch + 1, last_ckpt_path, input_cols)  # ⭐ ADD input_cols
            
            # Save snapshots
            if (epoch + 1) % save_every == 0:
                snap_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch + 1, snap_path, input_cols)  # ⭐ ADD input_cols

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
    artifact_dir: str,
    pretrained_config_path: Optional[str] = None  # ⭐ ADD THIS
):
    """
    Save input/output variable specifications for model deployment.
    
    When retraining (pretrained_config_path provided), copies the ORIGINAL config.
    When training fresh, generates config from training data.
    
    Args:
        input_cols: List of input feature names
        output_cols: List of output feature names
        train_df_unscaled: UNSCALED training dataframe (original physical units)
        artifact_dir: Directory to save the configuration
        pretrained_config_path: Path to original feature_config.yml (if retraining)
    """
    logger.info("Saving variable configuration...")
    
    config_path = os.path.join(artifact_dir, 'feature_config.yml')
    
    # ⭐ CASE 1: RETRAINING - Copy original config exactly
    if pretrained_config_path is not None:
        if not os.path.exists(pretrained_config_path):
            logger.error(f"❌ Pretrained config not found: {pretrained_config_path}")
            logger.error("   Cannot save variable config")
            return False
        
        try:
            import shutil
            shutil.copy2(pretrained_config_path, config_path)
            logger.info(f"  ✓ Copied original feature_config.yml from pre-trained model")
            logger.info(f"    Source: {pretrained_config_path}")
            logger.info(f"    Dest:   {config_path}")
            
            # ⭐ OPTIONAL: Log a warning if new data ranges differ significantly
            try:
                orig_input_vars, _ = variables_from_yaml(pretrained_config_path)
                
                logger.info("\n  📊 Comparing original vs new data ranges:")
                mismatches = []
                
                for var in orig_input_vars[:10]:  # Check first 10 for brevity
                    col = var.name
                    if col in train_df_unscaled.columns:
                        orig_min, orig_max = var.value_range
                        new_min = train_df_unscaled[col].min()
                        new_max = train_df_unscaled[col].max()
                        
                        # Check if new data exceeds original range by >10%
                        orig_range = orig_max - orig_min
                        if orig_range > 1e-6:
                            if new_min < orig_min * 0.9 or new_max > orig_max * 1.1:
                                mismatches.append({
                                    'col': col,
                                    'orig_range': (orig_min, orig_max),
                                    'new_range': (new_min, new_max)
                                })
                
                if mismatches:
                    logger.warning(f"\n  ⚠️  {len(mismatches)} features have new data outside original range:")
                    for m in mismatches[:5]:
                        logger.warning(
                            f"     - {m['col']}: "
                            f"orig [{m['orig_range'][0]:.4f}, {m['orig_range'][1]:.4f}] "
                            f"→ new [{m['new_range'][0]:.4f}, {m['new_range'][1]:.4f}]"
                        )
                    if len(mismatches) > 5:
                        logger.warning(f"     ... and {len(mismatches)-5} more")
                    logger.warning("\n  ⚠️  Model may extrapolate outside training range!")
                else:
                    logger.info("     ✓ New data ranges are within original training ranges")
                    
            except Exception as e:
                logger.debug(f"Could not compare ranges: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy feature config: {e}")
            return False
    
    # ⭐ CASE 2: FRESH TRAINING - Generate config from data
    else:
        try:
            input_variables = []
            for col in input_cols:
                # Use unscaled data to get physical ranges
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
            
            variables_as_yaml(input_variables, output_variables, config_path)
            logger.info(f"  ✓ Generated new feature_config.yml from training data")
            logger.info(f"    Path: {config_path}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to generate variable config: {e}")
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

def load_model_metadata(model_path: str) -> Dict:
    """
    Load and validate pre-trained model metadata.
    
    Returns:
        dict with keys:
            - input_features: List[str]
            - architecture: List[int]
            - scalers: dict with 'input' and 'output'
            - config_path: str
    """
    logger.info("=" * 70)
    logger.info("LOADING PRE-TRAINED MODEL METADATA")
    logger.info("=" * 70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"Model directory: {model_path}")
    
    metadata = {}
    
    # 1. Load feature config
    config_path = os.path.join(model_path, 'feature_config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ No feature_config.yml found in {model_path}")
    
    try:
        input_variables, output_variables = variables_from_yaml(config_path)
        metadata['input_features'] = [v.name for v in input_variables]
        metadata['output_features'] = [v.name for v in output_variables]
        logger.info(f"✓ Loaded {len(metadata['input_features'])} input features")
        logger.info(f"✓ Loaded {len(metadata['output_features'])} output features")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load feature config: {e}")
    
    # 2. Load scalers
    input_scaler_path = os.path.join(model_path, 'input_scaler.pt')
    output_scaler_path = os.path.join(model_path, 'output_scaler.pt')
    
    if not os.path.exists(input_scaler_path) or not os.path.exists(output_scaler_path):
        raise FileNotFoundError(f"❌ Scalers not found in {model_path}")
    
    try:
        metadata['scalers'] = {
            'input': torch.load(input_scaler_path, weights_only=False),
            'output': torch.load(output_scaler_path, weights_only=False)
        }
        logger.info(f"✓ Loaded input scaler ({metadata['scalers']['input'].coefficient.shape[0]} features)")
        logger.info(f"✓ Loaded output scaler ({metadata['scalers']['output'].coefficient.shape[0]} features)")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load scalers: {e}")
    
    # 3. Detect model architecture
    model_file = os.path.join(model_path, 'final_model.pt')
    if not os.path.exists(model_file):
        model_file = os.path.join(model_path, 'best_model.pt')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ No model file found in {model_path}")
    
    try:
        pretrained_model = torch.load(model_file, map_location='cpu', weights_only=False)
        
        # Get state dict
        if isinstance(pretrained_model, nn.Sequential):
            state_dict = pretrained_model.state_dict()
        elif isinstance(pretrained_model, dict):
            state_dict = pretrained_model
        else:
            state_dict = pretrained_model.state_dict()
        
        # Infer hidden dimensions
        hidden_dims = []
        layer_idx = 0
        while f'{layer_idx}.weight' in state_dict:
            out_features = state_dict[f'{layer_idx}.weight'].shape[0]
            hidden_dims.append(out_features)
            
            # Skip to next Linear layer (skip ELU and optional Dropout)
            layer_idx += 2
            if f'{layer_idx}.weight' not in state_dict and f'{layer_idx+1}.weight' in state_dict:
                layer_idx += 1
        
        # Remove output layer
        hidden_dims = hidden_dims[:-1]
        
        metadata['architecture'] = hidden_dims
        metadata['model_state_dict'] = state_dict
        logger.info(f"✓ Detected architecture: {hidden_dims}")
        
        # Verify dimensions
        expected_input_size = state_dict['0.weight'].shape[1]
        if expected_input_size != len(metadata['input_features']):
            raise ValueError(
                f"❌ Model/config mismatch!\n"
                f"   Model expects: {expected_input_size} features\n"
                f"   Config has: {len(metadata['input_features'])} features"
            )
        
        logger.info(f"✓ Model dimensions validated")
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model architecture: {e}")
    
    # 4. Load training config (optional, for reference)
    training_config_path = os.path.join(model_path, 'training_config.json')
    if os.path.exists(training_config_path):
        try:
            with open(training_config_path, 'r') as f:
                metadata['training_config'] = json.load(f)
            logger.info(f"✓ Loaded training config")
        except Exception as e:
            logger.warning(f"⚠️  Could not load training config: {e}")
    
    logger.info("=" * 70)
    return metadata


# ============================================================================
# UPDATED: Main function with fixed retraining logic
# ============================================================================

def main():
    """Main function with corrected retraining logic."""
    args = parse_arguments()
    
    logger.info("=" * 70)
    logger.info("FEL NEURAL NETWORK MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # Setup checkpoint directory
    ckpt_dir = setup_checkpoint_dir(args.checkpoint_dir)
    
    # ============================================================================
    # ⭐ STEP 1: Determine training mode and load metadata FIRST
    # ============================================================================
    
    training_mode = None
    model_metadata = None
    input_cols_override = None
    pretrained_scalers = None
    pretrained_architecture = None
    
    if args.resume_from:
        training_mode = "RESUME"
        logger.info("🔄 RESUME MODE: Continuing from checkpoint")
        logger.info(f"   Checkpoint: {args.resume_from}")
        logger.info("   Will detect features from data (same as original run)")
        
    elif args.model_path:
        training_mode = "RETRAIN"
        logger.info("🔄 RETRAIN MODE: Fine-tuning pre-trained model")
        logger.info(f"   Model path: {args.model_path}")
        
        # Load ALL model metadata BEFORE data processing
        try:
            model_metadata = load_model_metadata(args.model_path)
            input_cols_override = model_metadata['input_features']
            pretrained_scalers = model_metadata['scalers']
            pretrained_architecture = model_metadata['architecture']
            
            logger.info("✓ Model metadata loaded successfully")
            logger.info(f"   Features: {len(input_cols_override)}")
            logger.info(f"   Architecture: {pretrained_architecture}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model metadata: {e}")
            logger.error("\n💡 Cannot proceed with retraining. Options:")
            logger.error("   1. Check model path is correct")
            logger.error("   2. Ensure all required files exist (model, scalers, config)")
            logger.error("   3. Train from scratch (remove --model_path)")
            exit(1)
    
    else:
        training_mode = "FRESH"
        logger.info("🆕 FRESH TRAINING: Starting from scratch")
    
    logger.info("=" * 70 + "\n")
    
    # ============================================================================
    # ⭐ STEP 2: Load and preprocess data (with feature override if retraining)
    # ============================================================================
    
    final_df, val_df, input_cols, output_cols = load_and_preprocess_data(
        args, 
        input_cols_override=input_cols_override  # None for fresh/resume, list for retrain
    )
    
    # ============================================================================
    # ⭐ STEP 3: Validate data compatibility (RETRAIN mode only)
    # ============================================================================
    
    if training_mode == "RETRAIN":
        logger.info("=" * 70)
        logger.info("VALIDATING DATA COMPATIBILITY")
        logger.info("=" * 70)
        
        # Check 1: All required features present
        missing_features = [f for f in input_cols_override if f not in final_df.columns]
        if missing_features:
            logger.error(f"❌ Missing {len(missing_features)} required features!")
            for feat in missing_features[:10]:
                logger.error(f"     - {feat}")
            if len(missing_features) > 10:
                logger.error(f"     ... and {len(missing_features)-10} more")
            
            logger.error("\n💡 SOLUTIONS:")
            logger.error("   1. Use data from same time period as original training")
            logger.error("   2. Check if PV names changed")
            logger.error("   3. Train from scratch on new data")
            exit(1)
        
        logger.info(f"✓ All {len(input_cols)} required features present")
        
        # Check 2: Warn about constant features (but don't drop them)
        feature_ranges = final_df[input_cols].max() - final_df[input_cols].min()
        constant_features = feature_ranges[feature_ranges == 0].index.tolist()
        
        if constant_features:
            logger.warning(f"⚠️  {len(constant_features)} features are constant in new data:")
            for feat in constant_features[:10]:
                logger.warning(f"     - {feat}")
            if len(constant_features) > 10:
                logger.warning(f"     ... and {len(constant_features)-10} more")
            logger.warning("\n   ⚠️  These will be KEPT to match pre-trained model architecture")
            logger.warning("   ⚠️  This may indicate data distribution shift!")
        else:
            logger.info("✓ No constant features detected")
        
        # Check 3: Compare data ranges (warning only)
        logger.info("\nComparing data ranges with original training data...")
        original_mins = pretrained_scalers['input'].offset
        original_maxs = original_mins + pretrained_scalers['input'].coefficient
        
        current_mins = final_df[input_cols].min().values
        current_maxs = final_df[input_cols].max().values
        
        # Features with significant range changes (>50% change)
        range_changes = []
        for i, col in enumerate(input_cols):
            orig_range = (original_maxs[i] - original_mins[i]).item()
            curr_range = current_maxs[i] - current_mins[i]
            
            if orig_range > 1e-6:  # Skip constant features
                range_ratio = abs(curr_range - orig_range) / orig_range
                if range_ratio > 0.5:  # >50% change
                    range_changes.append((col, range_ratio, orig_range, curr_range))
        
        if range_changes:
            logger.warning(f"⚠️  {len(range_changes)} features have significant range changes:")
            for col, ratio, orig, curr in sorted(range_changes, key=lambda x: x[1], reverse=True)[:10]:
                logger.warning(f"     - {col}: {ratio*100:.0f}% change (orig: {orig:.4f}, new: {curr:.4f})")
            if len(range_changes) > 10:
                logger.warning(f"     ... and {len(range_changes)-10} more")
            logger.warning("\n   ⚠️  Large range changes may hurt model performance")
            logger.warning("   ⚠️  Consider retraining from scratch if performance degrades")
        else:
            logger.info("✓ Data ranges are similar to original training")
        
        logger.info("=" * 70)
    
    # ============================================================================
    # ⭐ STEP 4: Train/test split
    # ============================================================================
    
    logger.info("\nPerforming train/test split (80/20)...")
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=39)
    
    # Save unscaled copy (for variable config)
    train_df_unscaled = train_df[input_cols + output_cols].copy()
    
    # ============================================================================
    # ⭐ STEP 5: Handle scalers (use pretrained or create new)
    # ============================================================================
    
    if training_mode == "RETRAIN":
        # Use pretrained scalers
        logger.info("=" * 70)
        logger.info("USING PRE-TRAINED SCALERS")
        logger.info("=" * 70)
        
        input_scaler = pretrained_scalers['input']
        output_scaler = pretrained_scalers['output']
        
        logger.info(f"✓ Input scaler:  {input_scaler.coefficient.shape[0]} features")
        logger.info(f"✓ Output scaler: {output_scaler.coefficient.shape[0]} features")
        
        # Verify dimensions
        if input_scaler.coefficient.shape[0] != len(input_cols):
            logger.error(f"❌ Scaler dimension mismatch!")
            logger.error(f"   Expected: {len(input_cols)}")
            logger.error(f"   Got: {input_scaler.coefficient.shape[0]}")
            exit(1)
        
        logger.info("=" * 70)
    
    else:
        # Create new scalers (FRESH or RESUME mode)
        logger.info("=" * 70)
        logger.info("CREATING NEW SCALERS FROM TRAINING DATA")
        logger.info("=" * 70)
        
        input_mins = train_df[input_cols].min()
        input_maxs = train_df[input_cols].max()
        output_mins = train_df[output_cols].min()
        output_maxs = train_df[output_cols].max()
        
        # Handle constant features
        input_ranges = input_maxs - input_mins
        constant_mask = input_ranges == 0
        
        if constant_mask.any():
            n_constant = constant_mask.sum()
            logger.warning(f"⚠️  {n_constant} features have zero range - setting range to 1.0")
            input_ranges[constant_mask] = 1.0
        
        input_scaler = AffineInputTransform(
            d=len(input_cols),
            coefficient=torch.tensor(input_ranges.values, dtype=torch.float32),
            offset=torch.tensor(input_mins.values, dtype=torch.float32),
        )
        output_scaler = AffineInputTransform(
            d=len(output_cols),
            coefficient=torch.tensor((output_maxs - output_mins).values, dtype=torch.float32),
            offset=torch.tensor(output_mins.values, dtype=torch.float32),
        )
        
        logger.info(f"✓ Created scalers")
        logger.info("=" * 70)
    
    # ============================================================================
    # ⭐ STEP 6: Apply scaling
    # ============================================================================
    
    logger.info("\nApplying scaling to all datasets...")
    train_df.loc[:, input_cols] = input_scaler.transform(
        torch.tensor(train_df[input_cols].to_numpy(dtype=np.float32))
    ).numpy()
    test_df.loc[:, input_cols] = input_scaler.transform(
        torch.tensor(test_df[input_cols].to_numpy(dtype=np.float32))
    ).numpy()
    train_df.loc[:, output_cols] = output_scaler.transform(
        torch.tensor(train_df[output_cols].to_numpy(dtype=np.float32))
    ).numpy()
    test_df.loc[:, output_cols] = output_scaler.transform(
        torch.tensor(test_df[output_cols].to_numpy(dtype=np.float32))
    ).numpy()
    
    if len(val_df) > 0:
        val_df.loc[:, input_cols] = input_scaler.transform(
            torch.tensor(val_df[input_cols].to_numpy(dtype=np.float32))
        ).numpy()
        val_df.loc[:, output_cols] = output_scaler.transform(
            torch.tensor(val_df[output_cols].to_numpy(dtype=np.float32))
        ).numpy()
    
    # Validate scaled data
    logger.info("\nValidating scaled data...")
    train_nan_count = train_df[input_cols].isna().sum().sum()
    train_inf_count = np.isinf(train_df[input_cols].values).sum()
    
    if train_nan_count > 0 or train_inf_count > 0:
        logger.error(f"❌ Invalid values in training data after scaling!")
        logger.error(f"   NaN: {train_nan_count}, Inf: {train_inf_count}")
        exit(1)
    
    logger.info("✓ Scaled data validated")
    logger.info(f"  Range: [{train_df[input_cols].min().min():.4f}, {train_df[input_cols].max().max():.4f}]")
    
    # ============================================================================
    # ⭐ STEP 7: Create model (use pretrained architecture if retraining)
    # ============================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("MODEL CREATION")
    logger.info("=" * 70)
    
    if training_mode == "RETRAIN":
        hidden_dims = pretrained_architecture
        logger.info(f"Using pre-trained architecture: {hidden_dims}")
    else:
        hidden_dims = [1024, 512, 256, 128, 64, 32, 16]
        logger.info(f"Fresh architecture: {hidden_dims}")
    
    model = FELNeuralNetwork(
        input_size=len(input_cols),
        output_size=len(output_cols),
        hidden_dims=hidden_dims,
        dropout=0.05,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Load pretrained weights if retraining
    if training_mode == "RETRAIN":
        logger.info("Loading pre-trained weights...")
        try:
            if isinstance(model_metadata['model_state_dict'], dict):
                # Check if it's a state dict for Sequential (with 'net.' prefix)
                if any(k.startswith('net.') for k in model_metadata['model_state_dict'].keys()):
                    model.load_state_dict(model_metadata['model_state_dict'])
                else:
                    # It's a state dict for the Sequential module only
                    model.net.load_state_dict(model_metadata['model_state_dict'])
            logger.info("✓ Pre-trained weights loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load weights: {e}")
            logger.error("   Training from scratch instead")
    
    # ============================================================================
    # ⭐ STEP 8: Create optimizer, scheduler, dataloaders
    # ============================================================================
    
    train_loader, test_loader = create_dataloaders(
        train_df, test_df, input_cols, output_cols, batch_size=args.batch_size
    )
    
    lr = 3e-6 # default: 1e-5
    weight_decay = 1e-4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=4)
    
    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 70)
    logger.info(f"Training mode:     {training_mode}")
    logger.info(f"Learning rate:     {lr}")
    logger.info(f"Weight decay:      {weight_decay}")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info(f"Epochs:            {args.epochs}")
    logger.info("=" * 70)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
    
    # ============================================================================
    # ⭐ STEP 9: Train
    # ============================================================================
    
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
    
    # ============================================================================
    # ⭐ STEP 10: Save artifacts
    # ============================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 70)
    
    model_dir = '/sdf/data/ad/ard/u/zihanzhu/ml/lcls_fel_tuning/model/'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add training mode to directory name
    mode_suffix = {
        "FRESH": "fresh",
        "RETRAIN": "retrain",
        "RESUME": "resume"
    }.get(training_mode, "unknown")
    
    artifact_dir = os.path.join(model_dir, f"{timestamp}_nn_{mode_suffix}")
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save all artifacts
    torch.save(best_model_state, os.path.join(artifact_dir, 'best_model.pt'))
    torch.save(model.net, os.path.join(artifact_dir, 'final_model.pt'))
    torch.save(input_scaler, os.path.join(artifact_dir, 'input_scaler.pt'))
    torch.save(output_scaler, os.path.join(artifact_dir, 'output_scaler.pt'))
    np.save(os.path.join(artifact_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(artifact_dir, 'test_losses.npy'), np.array(test_losses))
    
    logger.info(f"\nArtifacts saved to: {artifact_dir}")
    
    # ⭐ Save variable config (use original if retraining)
    pretrained_config_path = None
    if training_mode == "RETRAIN" and model_metadata is not None:
        pretrained_config_path = os.path.join(args.model_path, 'feature_config.yml')
    
    save_variable_config(
        input_cols, 
        output_cols, 
        train_df_unscaled, 
        artifact_dir,
        pretrained_config_path=pretrained_config_path  # ⭐ ADD THIS
    )
    
    # Save training config
    additional_info = {
        'training_mode': training_mode,
        'best_test_loss': float(min(test_losses)),
        'final_test_loss': float(test_losses[-1]),
        'n_test_samples': len(test_df),
        'n_validation_samples': len(val_df),
    }
    
    if training_mode == "RETRAIN":
        additional_info['pretrained_model_path'] = args.model_path
        additional_info['original_feature_config'] = pretrained_config_path
    
    save_training_config(args, input_cols, output_cols, train_df, artifact_dir, additional_info)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
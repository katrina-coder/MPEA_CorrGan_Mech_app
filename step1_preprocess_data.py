"""
step1_preprocess_data.py  —  Preprocess GAN-augmented MPEA corrosion dataset
─────────────────────────────────────────────────────────────────────────────
Downloads full 202k-row file from GitHub, preprocesses, and saves a clean
Excel file ready for model training.

What this script does:
  1. Downloads MPEAs_Mech_CorrGAN_DB.xlsx from GitHub (202269 rows x 77 cols)
  2. Drops Be and La columns (only 1 row each in original data → excluded)
  3. Splits into mechanical (1704) and corrosion (565 real + ~200k GAN) rows
  4. For corrosion rows:
     - Drops PBS and Hanks rows if any remain (n < 15, excluded)
     - Validates electrolyte one-hot encoding readiness
     - Log₁₀-scales icorr (Corrosion current density) — handles zeros safely
     - Adds icorr_log10 column (original icorr preserved for reference)
  5. Saves MPEAs_Mech_CorrGAN_DB_processed.xlsx with two sheets:
       - 'mechanical'  : 1704 rows, ready for mechanical model training
       - 'corrosion'   : all corrosion rows with icorr_log10 added
     And one combined sheet:
       - 'all'         : full dataset (mechanical + corrosion)

Usage:
  python step1_preprocess_data.py

Output:
  MPEAs_Mech_CorrGAN_DB_processed.xlsx
"""

import os
import urllib.request
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_RAW_URL = (
    "https://media.githubusercontent.com/media/katrina-coder/"
    "MPEA_CorrGan_Mech_app/main/MPEAs_Mech_CorrGAN_DB.xlsx"
)
LOCAL_FILENAME  = "MPEAs_Mech_CorrGAN_DB.xlsx"
OUTPUT_FILENAME = "Updated_MPEAs_Mech_CorrGAN_DB_processed.xlsx"

ELEMENTS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
            'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
            'Sn','Ta','Ti','V','W','Y','Zn','Zr']

ELECTROLYTES_KEEP = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']
ELECTROLYTES_DROP = ['PBS','Hanks']   # n < 15, excluded

EMP_COLS = ['a','delta','Tm','std of Tm','entropy','enthalpy',
            'std of enthalpy','omega','X','std of X',
            'VEC','std of vec','K','std of K','density']

# ── Step 1: Download ───────────────────────────────────────────────────────────
if os.path.exists(LOCAL_FILENAME):
    print(f"✓ Found local file: {LOCAL_FILENAME}")
else:
    print(f"Downloading from GitHub LFS...")
    print(f"  URL: {GITHUB_RAW_URL}")
    urllib.request.urlretrieve(GITHUB_RAW_URL, LOCAL_FILENAME)
    size_mb = os.path.getsize(LOCAL_FILENAME) / 1024 / 1024
    print(f"✓ Downloaded: {size_mb:.1f} MB")

# ── Step 2: Load ───────────────────────────────────────────────────────────────
print("\nLoading dataset (202k rows — may take ~30 seconds)...")
df = pd.read_excel(LOCAL_FILENAME)
print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Step 3: Basic validation ───────────────────────────────────────────────────
print("\n=== Dataset composition ===")
print(df['OG property'].value_counts())

# ── Step 4: Drop Be and La ─────────────────────────────────────────────────────
for col in ['Be', 'La']:
    if col in df.columns:
        n_nonzero = (df[col] != 0).sum()
        print(f"\nDropping '{col}' column — {n_nonzero} non-zero rows (negligible, excluded)")
        df = df.drop(columns=[col])

print(f"\n✓ Elements after dropping Be/La: {len([c for c in ELEMENTS if c in df.columns])}/32")

# ── Step 5: Drop unused columns not needed for training ───────────────────────
DROP_COLS = ['Calculated passive window', 'Reference', 'Test environment']
dropped = [c for c in DROP_COLS if c in df.columns]
if dropped:
    df = df.drop(columns=dropped)
    print(f"✓ Dropped unused columns: {dropped}")

# ── Step 6: Split mechanical vs corrosion ─────────────────────────────────────
mech = df[df['OG property'] == 'mechanical'].copy().reset_index(drop=True)
corr = df[df['OG property'] != 'mechanical'].copy().reset_index(drop=True)

print(f"\n=== Split ===")
print(f"  Mechanical rows : {len(mech):,}")
print(f"  Corrosion rows  : {len(corr):,}")

# ── Step 7: Corrosion — inspect and clean ─────────────────────────────────────
print(f"\n=== Corrosion electrolyte distribution ===")
print(corr['Electrolyte'].value_counts(dropna=False))

# Drop PBS and Hanks if present
for elec in ELECTROLYTES_DROP:
    n = (corr['Electrolyte'] == elec).sum()
    if n > 0:
        print(f"\nDropping {n} '{elec}' rows (n < 15 threshold)")
        corr = corr[corr['Electrolyte'] != elec].reset_index(drop=True)

print(f"\n✓ Corrosion rows after electrolyte filtering: {len(corr):,}")

# ── Step 8: Corrosion targets — report stats ──────────────────────────────────
print(f"\n=== Corrosion target statistics ===")
target_cols = {
    'Ecorr' : 'Corrosion potential (mV vs SCE)',
    'Epit'  : 'Pitting potential (mV vs SCE)',
    'icorr' : 'Corrosion current density (microA/cm2)',
}
for name, col in target_cols.items():
    vals = corr[col].replace(0, np.nan).dropna()
    print(f"  {name} ({col}):")
    print(f"    n={len(vals):,}  |  min={vals.min():.4f}  max={vals.max():.2f}  median={vals.median():.4f}")

# ── Step 9: Log₁₀ scale icorr ────────────────────────────────────────────────
print(f"\n=== Log₁₀ scaling icorr ===")
icorr_raw = corr['Corrosion current density (microA/cm2)'].copy()

# Identify problematic values
n_zero = (icorr_raw == 0).sum()
n_neg  = (icorr_raw < 0).sum()
n_pos  = (icorr_raw > 0).sum()
print(f"  Zero values  : {n_zero:,}  (treated as missing — set to NaN in log column)")
print(f"  Negative vals: {n_neg:,}  (treated as missing — set to NaN in log column)")
print(f"  Positive vals: {n_pos:,}  (these get log₁₀ transformed)")

# Create log10 column — NaN where original is 0 or negative
icorr_log = icorr_raw.copy().astype(float)
icorr_log[icorr_raw <= 0] = np.nan
icorr_log[icorr_raw > 0]  = np.log10(icorr_raw[icorr_raw > 0])

corr['icorr_log10'] = icorr_log

valid_log = icorr_log.dropna()
print(f"\n  log₁₀(icorr) statistics:")
print(f"    n={len(valid_log):,}  |  min={valid_log.min():.3f}  max={valid_log.max():.3f}  "
      f"mean={valid_log.mean():.3f}  std={valid_log.std():.3f}")

# ── Step 10: Validate empirical params ────────────────────────────────────────
print(f"\n=== Empirical parameter coverage ===")
all_filled = True
for col in EMP_COLS:
    if col not in corr.columns:
        print(f"  MISSING column: {col}")
        all_filled = False
    else:
        n_nonzero = (corr[col].replace(0, np.nan).dropna()).shape[0]
        pct = 100 * n_nonzero / len(corr)
        status = "✓" if pct > 95 else "⚠"
        print(f"  {status} {col}: {n_nonzero:,}/{len(corr):,} filled ({pct:.1f}%)")

if all_filled:
    print("\n✓ All empirical parameters pre-filled — no recalculation needed")

# ── Step 11: Validate element columns ────────────────────────────────────────
missing_elements = [e for e in ELEMENTS if e not in corr.columns]
if missing_elements:
    print(f"\n⚠ Missing element columns: {missing_elements}")
else:
    print(f"\n✓ All 32 element columns present")

# ── Step 12: Final summary ────────────────────────────────────────────────────
print(f"\n=== Final dataset summary ===")
print(f"  Mechanical rows : {len(mech):,}")
print(f"  Corrosion rows  : {len(corr):,}")
print(f"    of which Ecorr valid : {corr['Corrosion potential (mV vs SCE)'].replace(0,np.nan).dropna().shape[0]:,}")
print(f"    of which Epit  valid : {corr['Pitting potential (mV vs SCE)'].replace(0,np.nan).dropna().shape[0]:,}")
print(f"    of which icorr valid : {corr['icorr_log10'].dropna().shape[0]:,}")
print(f"\n  Columns in output: {df.shape[1] - len(['Be','La']) + 1} (added icorr_log10)")

# ── Step 13: Save ─────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILENAME}...")

# Recombine for the 'all' sheet
all_df = pd.concat([mech, corr], ignore_index=True)

with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
    all_df.to_excel(writer, sheet_name='all',        index=False)
    mech.to_excel(  writer, sheet_name='mechanical', index=False)
    corr.to_excel(  writer, sheet_name='corrosion',  index=False)

size_mb = os.path.getsize(OUTPUT_FILENAME) / 1024 / 1024
print(f"✓ Saved: {OUTPUT_FILENAME} ({size_mb:.1f} MB)")
print(f"\n  Sheets:")
print(f"    'all'         : {len(all_df):,} rows — full dataset")
print(f"    'mechanical'  : {len(mech):,} rows — for mechanical model training")
print(f"    'corrosion'   : {len(corr):,} rows — for corrosion model training")
print(f"\n✅ Preprocessing complete — ready for step2_retrain_corr_models.py")

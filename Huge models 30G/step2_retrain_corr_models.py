"""
step2_retrain_corr_models.py  —  Retrain corrosion models with GAN-augmented data
──────────────────────────────────────────────────────────────────────────────────
Trains new Ecorr, Epit, and icorr regressors on the GAN-augmented corrosion
dataset (~200k rows). Mechanical models and phase classifiers are NOT retrained —
they are downloaded at app runtime from the existing GitHub repo.

Pipeline A: Trains on corrosion rows only (565 real + 200k GAN).
            Uses only rows where each target is non-zero/non-NaN.
            Strict: no imputation — mirrors the original Pipeline A philosophy.

Pipeline B: Trains on all rows (1704 mech + 200,565 corr) with MissForest-style
            imputation (IterativeImputer + RandomForest, max_iter=5).
            Same approach as the existing Pipeline B mechanical models.

Feature vector (66-dim):
  32 element fractions
  + 7 processing one-hot (process_1..7)
  + 15 empirical params (a, delta, Tm, std Tm, entropy, enthalpy, std enthalpy,
                          omega, X, std X, VEC, std VEC, K, std K, density)
  + 4 phase flags (FCC, BCC, HCP, IM)
  = 58 mechanical features
  + 7 electrolyte one-hot (NaCl, H2SO4, Seawater, HNO3, NaOH, HCl, KOH)
  + 1 concentration (normalised by 6 M)
  = 66 corrosion features

Model: RandomForestRegressor (n_estimators=300, n_jobs=-1)
       Same architecture as existing mechanical models for consistency.

Evaluation: 10-run average R² with random train/test splits (80/20).

Usage:
  python step2_retrain_corr_models.py

Output:
  models_corr_A/  ─ ecorr_regressor.joblib, epit_regressor.joblib, icorr_regressor.joblib
  models_corr_B/  ─ ecorr_regressor.joblib, epit_regressor.joblib, icorr_regressor.joblib
  corr_model_r2_results.csv  ─ R² summary for all models and pipelines
"""

import os
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_FILE = "Updated_MPEAs_Mech_CorrGAN_DB_processed.xlsx"
OUT_DIR_A      = "models_corr_A"
OUT_DIR_B      = "models_corr_B"
N_ESTIMATORS   = 300
N_RUNS         = 10      # repeated R² evaluation runs
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
MAX_ITER_IMP   = 5       # IterativeImputer max iterations (Pipeline B)
CONC_MAX       = 6.0     # normalisation denominator for concentration

ELEMENTS    = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
               'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
               'Sn','Ta','Ti','V','W','Y','Zn','Zr']
PROC_COLS   = ['process_1','process_2','process_3','process_4',
               'process_5','process_6','process_7']
EMP_COLS    = ['a','delta','Tm','std of Tm','entropy','enthalpy',
               'std of enthalpy','omega','X','std of X',
               'VEC','std of vec','K','std of K','density']
PHASE_COLS  = ['FCC','BCC','HCP','IM']
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']

MECH_FEAT_COLS = ELEMENTS + PROC_COLS + EMP_COLS + PHASE_COLS   # 58-dim

TARGETS = {
    'Ecorr' : 'Corrosion potential (mV vs SCE)',
    'Epit'  : 'Pitting potential (mV vs SCE)',
    'icorr' : 'icorr_log10',   # log₁₀-scaled, added by step1
}

os.makedirs(OUT_DIR_A, exist_ok=True)
os.makedirs(OUT_DIR_B, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_corr_features(df):
    """Build 66-dim corrosion feature matrix from a DataFrame.
    Rows with missing electrolyte or concentration are filled with zeros."""
    # 58 mechanical features
    X58 = df[MECH_FEAT_COLS].to_numpy(dtype=float)

    # 7 electrolyte one-hot
    elec_onehot = np.zeros((len(df), len(ELECTROLYTES)))
    for j, elec in enumerate(ELECTROLYTES):
        elec_onehot[:, j] = (df['Electrolyte'] == elec).astype(float).values

    # 1 concentration (normalised)
    conc = df['Concentration in M'].fillna(0).to_numpy(dtype=float) / CONC_MAX

    return np.column_stack([X58, elec_onehot, conc.reshape(-1, 1)])


def evaluate_model(model_cls, X, y, n_runs, test_size, label):
    """Train and evaluate a model n_runs times, return mean ± std R²."""
    r2_scores = []
    for run in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=run)
        m = model_cls()
        m.fit(X_tr, y_tr)
        r2_scores.append(r2_score(y_te, m.predict(X_te)))
    mean_r2 = np.mean(r2_scores)
    std_r2  = np.std(r2_scores)
    print(f"    {label}: R² = {mean_r2:.3f} ± {std_r2:.3f}  (n={len(y):,})")
    return mean_r2, std_r2


def train_final_model(model_cls, X, y):
    """Train final model on ALL available data for deployment."""
    m = model_cls()
    m.fit(X, y)
    return m


def rf_factory():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )


# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  MPEA Corrosion Model Retraining — GAN-Augmented Dataset")
print("=" * 65)
print(f"\nLoading {PROCESSED_FILE}...")
print("(Reading 200k rows — may take ~30 seconds)")

corr = pd.read_excel(PROCESSED_FILE, sheet_name='corrosion')
mech = pd.read_excel(PROCESSED_FILE, sheet_name='mechanical')

print(f"✓ Corrosion sheet: {len(corr):,} rows")
print(f"✓ Mechanical sheet: {len(mech):,} rows")

print(f"\nCorrosion data breakdown:")
print(f"  {corr['OG property'].value_counts().to_dict()}")

print(f"\nElectrolyte distribution (corrosion):")
print(corr['Electrolyte'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE A  —  Corrosion rows only, no imputation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE A — Corrosion rows only (565 real + 200k GAN)")
print("=" * 65)

X_corr_A = build_corr_features(corr)
results_A = {}

for name, col in TARGETS.items():
    print(f"\n── {name} ──")

    # Use only rows where this target is non-NaN
    valid_mask = corr[col].notna() & (corr[col] != 0)
    X_valid    = X_corr_A[valid_mask.values]
    y_valid    = corr.loc[valid_mask, col].to_numpy(dtype=float)

    print(f"  Training rows: {len(y_valid):,}")
    if name == 'icorr':
        print(f"  Target: icorr_log10  (range [{y_valid.min():.3f}, {y_valid.max():.3f}])")
    else:
        print(f"  Target range: [{y_valid.min():.1f}, {y_valid.max():.1f}]")

    # Evaluate
    mean_r2, std_r2 = evaluate_model(rf_factory, X_valid, y_valid,
                                     N_RUNS, TEST_SIZE, "Pipeline A")
    results_A[name] = {'mean_r2': mean_r2, 'std_r2': std_r2, 'n': len(y_valid)}

    # Train final model on all available data
    print(f"  Training final model on all {len(y_valid):,} rows...")
    model = train_final_model(rf_factory, X_valid, y_valid)
    fname = f"{name.lower()}_regressor.joblib"
    dump(model, os.path.join(OUT_DIR_A, fname))
    print(f"  ✓ Saved → {OUT_DIR_A}/{fname}")

print(f"\nPipeline A summary:")
for name, r in results_A.items():
    print(f"  {name}: R² = {r['mean_r2']:.3f} ± {r['std_r2']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE B  —  All rows, MissForest-style imputation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE B — All rows with MissForest imputation")
print("=" * 65)

# Combine mechanical + corrosion for Pipeline B
# Mechanical rows get NaN for corrosion targets and electrolyte features
print(f"\nBuilding unified feature matrix (mech + corrosion)...")

# Add missing columns to mech so it can be stacked
mech_aug = mech.copy()
mech_aug['Electrolyte']       = np.nan
mech_aug['Concentration in M'] = np.nan
for col in ['Corrosion potential (mV vs SCE)',
            'Pitting potential (mV vs SCE)',
            'icorr_log10']:
    if col not in mech_aug.columns:
        mech_aug[col] = np.nan

# Build feature matrices
X_corr_B = build_corr_features(corr)
X_mech_B = build_corr_features(mech_aug)   # elec/conc cols will be 0 (NaN→0 in fillna)
X_all_B  = np.vstack([X_mech_B, X_corr_B])

print(f"  Combined feature matrix: {X_all_B.shape[0]:,} rows × {X_all_B.shape[1]} features")

# Target arrays — NaN for mechanical rows
y_ecorr_all = np.concatenate([
    np.full(len(mech), np.nan),
    corr['Corrosion potential (mV vs SCE)'].to_numpy(dtype=float)
])
y_epit_all  = np.concatenate([
    np.full(len(mech), np.nan),
    corr['Pitting potential (mV vs SCE)'].to_numpy(dtype=float)
])
y_icorr_all = np.concatenate([
    np.full(len(mech), np.nan),
    corr['icorr_log10'].to_numpy(dtype=float)
])

# Replace 0 with NaN in targets (0 = missing in original data)
y_ecorr_all[y_ecorr_all == 0] = np.nan
y_epit_all[ y_epit_all  == 0] = np.nan

print(f"\n  Target coverage in combined dataset:")
print(f"    Ecorr valid: {np.sum(~np.isnan(y_ecorr_all)):,} / {len(y_ecorr_all):,}")
print(f"    Epit  valid: {np.sum(~np.isnan(y_epit_all)):,}  / {len(y_epit_all):,}")
print(f"    icorr valid: {np.sum(~np.isnan(y_icorr_all)):,} / {len(y_icorr_all):,}")

# MissForest imputation on feature matrix
print(f"\n  Running MissForest imputation on feature matrix...")
print(f"  (IterativeImputer, RandomForest estimator, max_iter={MAX_ITER_IMP})")
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, n_jobs=-1,
                                    random_state=RANDOM_STATE),
    max_iter=MAX_ITER_IMP,
    random_state=RANDOM_STATE
)
X_all_B_imp = imputer.fit_transform(X_all_B)
print(f"  ✓ Imputation complete")

results_B = {}

for name, y_all in [('Ecorr', y_ecorr_all),
                    ('Epit',  y_epit_all),
                    ('icorr', y_icorr_all)]:
    print(f"\n── {name} ──")

    valid_mask = ~np.isnan(y_all)
    X_valid    = X_all_B_imp[valid_mask]
    y_valid    = y_all[valid_mask]

    print(f"  Training rows: {len(y_valid):,}")

    mean_r2, std_r2 = evaluate_model(rf_factory, X_valid, y_valid,
                                     N_RUNS, TEST_SIZE, "Pipeline B")
    results_B[name] = {'mean_r2': mean_r2, 'std_r2': std_r2, 'n': len(y_valid)}

    print(f"  Training final model on all {len(y_valid):,} rows...")
    model = train_final_model(rf_factory, X_valid, y_valid)
    fname = f"{name.lower()}_regressor.joblib"
    dump(model, os.path.join(OUT_DIR_B, fname))
    print(f"  ✓ Saved → {OUT_DIR_B}/{fname}")

print(f"\nPipeline B summary:")
for name, r in results_B.items():
    print(f"  {name}: R² = {r['mean_r2']:.3f} ± {r['std_r2']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  R² COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  R² COMPARISON — Original vs GAN-Augmented")
print("=" * 65)

# Original R² values from existing app (for reference)
orig_A = {'Ecorr': 0.646, 'Epit': 0.775, 'icorr': 0.451}
orig_B = {'Ecorr': 0.742, 'Epit': 0.856, 'icorr': 0.598}

rows = []
print(f"\n{'Model':<8} {'Orig A':>8} {'New A':>8} {'ΔA':>7}  {'Orig B':>8} {'New B':>8} {'ΔB':>7}")
print("-" * 65)
for name in ['Ecorr', 'Epit', 'icorr']:
    new_a  = results_A[name]['mean_r2']
    new_b  = results_B[name]['mean_r2']
    delta_a = new_a - orig_A[name]
    delta_b = new_b - orig_B[name]
    print(f"{name:<8} {orig_A[name]:>8.3f} {new_a:>8.3f} {delta_a:>+7.3f}  "
          f"{orig_B[name]:>8.3f} {new_b:>8.3f} {delta_b:>+7.3f}")
    rows.append({
        'Property'         : name,
        'Original A R²'    : orig_A[name],
        'GAN-augmented A R²': round(new_a, 3),
        'Δ A'              : round(delta_a, 3),
        'n (A)'            : results_A[name]['n'],
        'std A'            : round(results_A[name]['std_r2'], 3),
        'Original B R²'    : orig_B[name],
        'GAN-augmented B R²': round(new_b, 3),
        'Δ B'              : round(delta_b, 3),
        'n (B)'            : results_B[name]['n'],
        'std B'            : round(results_B[name]['std_r2'], 3),
    })

# Save R² summary
results_df = pd.DataFrame(rows)
results_df.to_csv('corr_model_r2_results.csv', index=False)
print(f"\n✓ R² results saved → corr_model_r2_results.csv")

print(f"\n{'=' * 65}")
print(f"  OUTPUT FILES")
print(f"{'=' * 65}")
for d in [OUT_DIR_A, OUT_DIR_B]:
    files = os.listdir(d)
    total_mb = sum(os.path.getsize(os.path.join(d, f)) for f in files) / 1024 / 1024
    print(f"\n  {d}/  ({total_mb:.0f} MB total)")
    for f in sorted(files):
        sz = os.path.getsize(os.path.join(d, f)) / 1024 / 1024
        print(f"    {f}  ({sz:.1f} MB)")

print(f"\n✅ Training complete!")
print(f"   Next: push models_corr_A/ and models_corr_B/ to GitHub,")
print(f"   then run step3_build_app.py to build the Streamlit app.")

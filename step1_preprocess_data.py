"""
step1_preprocess_data.py  —  Preprocess GAN-augmented MPEA corrosion dataset
─────────────────────────────────────────────────────────────────────────────
Downloads full 202k-row file from GitHub, preprocesses, and saves a clean
Excel file ready for model training.

What this script does:
  1. Downloads MPEAs_Mech_CorrGAN_DB.xlsx from GitHub (202269 rows x 78 cols)
  2. Drops Be and La columns → keeps 32-element feature space
  3. Splits into mechanical (1704) and corrosion (565 real + ~200k GAN) rows
  4. For corrosion rows:
     - Drops PBS and Hanks rows if any remain (n < 15, excluded)
     - Calculates all 15 empirical parameters for GAN rows where they are zero
     - Log₁₀-scales icorr — adds icorr_log10 column (original preserved)
  5. Saves Updated_MPEAs_Mech_CorrGAN_DB_processed.xlsx with 3 sheets:
       'all', 'mechanical', 'corrosion'

Usage:
  python step1_preprocess_data.py

Output:
  Updated_MPEAs_Mech_CorrGAN_DB_processed.xlsx
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

ELECTROLYTES_DROP = ['PBS','Hanks']

EMP_COLS = ['a','delta','Tm','std of Tm','entropy','enthalpy',
            'std of enthalpy','omega','X','std of X',
            'VEC','std of vec','K','std of K','density']

# ── Empirical parameter lookup tables (identical to app.py) ──────────────────
ATOMIC_RADII   = {'Ag':1.44,'Al':1.43,'B':0.87,'C':0.77,'Ca':1.97,'Co':1.25,'Cr':1.28,
                  'Cu':1.28,'Fe':1.26,'Ga':1.22,'Ge':1.22,'Hf':1.59,'Li':1.52,'Mg':1.60,
                  'Mn':1.26,'Mo':1.36,'N':0.75,'Nb':1.43,'Nd':1.82,'Ni':1.24,'Pd':1.37,
                  'Re':1.37,'Sc':1.62,'Si':1.18,'Sn':1.40,'Ta':1.43,'Ti':1.47,'V':1.34,
                  'W':1.37,'Y':1.80,'Zn':1.33,'Zr':1.60}
MELTING_TEMPS  = {'Ag':1235,'Al':933,'B':2349,'C':3823,'Ca':1115,'Co':1768,'Cr':2180,
                  'Cu':1358,'Fe':1811,'Ga':303,'Ge':1211,'Hf':2506,'Li':454,'Mg':923,
                  'Mn':1519,'Mo':2896,'N':63,'Nb':2750,'Nd':1297,'Ni':1728,'Pd':1828,
                  'Re':3459,'Sc':1814,'Si':1687,'Sn':505,'Ta':3290,'Ti':1941,'V':2183,
                  'W':3695,'Y':1799,'Zn':693,'Zr':2128}
ELECTRONEG_D   = {'Ag':1.93,'Al':1.61,'B':2.04,'C':2.55,'Ca':1.00,'Co':1.88,'Cr':1.66,
                  'Cu':1.90,'Fe':1.83,'Ga':1.81,'Ge':2.01,'Hf':1.30,'Li':0.98,'Mg':1.31,
                  'Mn':1.55,'Mo':2.16,'N':3.04,'Nb':1.60,'Nd':1.14,'Ni':1.91,'Pd':2.20,
                  'Re':1.90,'Sc':1.36,'Si':1.90,'Sn':1.96,'Ta':1.50,'Ti':1.54,'V':1.63,
                  'W':2.36,'Y':1.22,'Zn':1.65,'Zr':1.33}
VEC_D          = {'Ag':11,'Al':3,'B':3,'C':4,'Ca':2,'Co':9,'Cr':6,'Cu':11,'Fe':8,'Ga':3,
                  'Ge':4,'Hf':4,'Li':1,'Mg':2,'Mn':7,'Mo':6,'N':5,'Nb':5,'Nd':4,'Ni':10,
                  'Pd':10,'Re':7,'Sc':3,'Si':4,'Sn':4,'Ta':5,'Ti':4,'V':5,'W':6,'Y':3,
                  'Zn':12,'Zr':4}
MOLAR_MASSES_D = {'Ag':107.87,'Al':26.98,'B':10.81,'C':12.01,'Ca':40.08,'Co':58.93,
                  'Cr':52.00,'Cu':63.55,'Fe':55.85,'Ga':69.72,'Ge':72.63,'Hf':178.49,
                  'Li':6.94,'Mg':24.31,'Mn':54.94,'Mo':95.96,'N':14.01,'Nb':92.91,
                  'Nd':144.24,'Ni':58.69,'Pd':106.42,'Re':186.21,'Sc':44.96,'Si':28.09,
                  'Sn':118.71,'Ta':180.95,'Ti':47.87,'V':50.94,'W':183.84,'Y':88.91,
                  'Zn':65.38,'Zr':91.22}
MOLAR_VOLS_D   = {'Ag':10.27,'Al':10.00,'B':4.39,'C':5.29,'Ca':26.20,'Co':6.67,'Cr':7.23,
                  'Cu':7.11,'Fe':7.09,'Ga':11.80,'Ge':13.63,'Hf':13.44,'Li':13.02,'Mg':14.00,
                  'Mn':7.35,'Mo':9.38,'N':13.54,'Nb':10.83,'Nd':20.59,'Ni':6.59,'Pd':8.56,
                  'Re':8.86,'Sc':15.00,'Si':12.06,'Sn':16.29,'Ta':10.85,'Ti':10.64,'V':8.32,
                  'W':9.47,'Y':19.88,'Zn':9.16,'Zr':14.02}
LATTICE_D      = {'Ag':4.09,'Al':4.05,'B':5.06,'C':3.57,'Ca':5.58,'Co':2.51,'Cr':2.88,
                  'Cu':3.62,'Fe':2.87,'Ga':4.52,'Ge':5.66,'Hf':3.20,'Li':3.51,'Mg':3.21,
                  'Mn':8.91,'Mo':3.15,'N':4.04,'Nb':3.30,'Nd':3.66,'Ni':3.52,'Pd':3.89,
                  'Re':2.76,'Sc':3.31,'Si':5.43,'Sn':5.83,'Ta':3.31,'Ti':2.95,'V':3.02,
                  'W':3.16,'Y':3.65,'Zn':2.66,'Zr':3.23}
BULK_MODULI_D  = {'Ag':100,'Al':76,'B':320,'C':443,'Ca':17,'Co':180,'Cr':160,'Cu':140,
                  'Fe':170,'Ga':59,'Ge':75,'Hf':110,'Li':11,'Mg':45,'Mn':120,'Mo':230,
                  'N':0,'Nb':170,'Nd':32,'Ni':180,'Pd':180,'Re':370,'Sc':57,'Si':98,
                  'Sn':58,'Ta':200,'Ti':110,'V':160,'W':310,'Y':41,'Zn':70,'Zr':94}
ENTHALPY_D     = {('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1,('Al','Fe'):-11,
                  ('Al','Hf'):-45,('Al','Mg'):-2,('Al','Mn'):-19,('Al','Mo'):-22,
                  ('Al','Nb'):-18,('Al','Ni'):-22,('Al','Si'):-19,('Al','Ta'):-19,
                  ('Al','Ti'):-30,('Al','V'):-16,('Al','W'):-16,('Al','Zr'):-44,
                  ('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):0,('Co','Mn'):0,('Co','Mo'):-5,
                  ('Co','Nb'):-25,('Co','Ni'):0,('Co','Ti'):-28,('Co','V'):-14,('Co','W'):-1,
                  ('Co','Zr'):-41,('Cr','Cu'):12,('Cr','Fe'):-1,('Cr','Mn'):2,('Cr','Mo'):0,
                  ('Cr','Nb'):-7,('Cr','Ni'):-7,('Cr','Si'):-37,('Cr','Ta'):-7,('Cr','Ti'):-7,
                  ('Cr','V'):-2,('Cr','W'):0,('Cr','Zr'):-12,('Cu','Fe'):13,('Cu','Mn'):4,
                  ('Cu','Mo'):19,('Cu','Ni'):4,('Cu','Ti'):-9,('Cu','Zr'):-23,('Fe','Mn'):0,
                  ('Fe','Mo'):-2,('Fe','Nb'):-16,('Fe','Ni'):-2,('Fe','Si'):-35,('Fe','Ta'):-15,
                  ('Fe','Ti'):-17,('Fe','V'):-7,('Fe','W'):-6,('Fe','Zr'):-25,('Mn','Mo'):0,
                  ('Mn','Ni'):-8,('Mn','Ti'):-8,('Mn','V'):-1,('Mo','Nb'):-6,('Mo','Ni'):-7,
                  ('Mo','Si'):-38,('Mo','Ta'):-5,('Mo','Ti'):-4,('Mo','V'):-5,('Mo','W'):0,
                  ('Mo','Zr'):-6,('Nb','Ni'):-30,('Nb','Si'):-56,('Nb','Ta'):0,('Nb','Ti'):-2,
                  ('Nb','V'):-2,('Nb','W'):-8,('Nb','Zr'):4,('Ni','Si'):-40,('Ni','Ta'):-24,
                  ('Ni','Ti'):-35,('Ni','V'):-18,('Ni','W'):-3,('Ni','Zr'):-49,('Si','Ta'):-45,
                  ('Si','Ti'):-66,('Si','V'):-48,('Si','W'):-37,('Si','Zr'):-84,('Ta','Ti'):-4,
                  ('Ta','V'):-1,('Ta','W'):-7,('Ti','V'):-2,('Ti','W'):-27,('Ti','Zr'):0,
                  ('V','W'):-8,('V','Zr'):-4,('W','Zr'):-27}
R_GAS = 8.314


def calc_empirical_vector(comp32):
    """Calculate 15 empirical parameters from a 32-element composition array."""
    x = {ELEMENTS[i]: comp32[i] for i in range(32) if comp32[i] > 1e-6}
    if not x:
        return np.zeros(15)
    total = sum(x.values())
    x = {e: v / total for e, v in x.items()}
    elems = list(x.keys())

    a_mean   = sum(x[e] * LATTICE_D[e]     for e in elems)
    r_mean   = sum(x[e] * ATOMIC_RADII[e]  for e in elems)
    delta    = 100 * np.sqrt(sum(x[e] * (1 - ATOMIC_RADII[e] / r_mean) ** 2 for e in elems))
    tm_mean  = sum(x[e] * MELTING_TEMPS[e] for e in elems)
    tm_std   = np.sqrt(sum(x[e] * (MELTING_TEMPS[e] - tm_mean) ** 2 for e in elems))
    entropy  = -R_GAS * sum(xi * np.log(xi) for xi in x.values())
    enthalpy = sum(4 * ENTHALPY_D.get((e1, e2), ENTHALPY_D.get((e2, e1), 0)) * x[e1] * x[e2]
                   for i, e1 in enumerate(elems) for e2 in elems[i + 1:])
    enth_sq  = sum((4 * ENTHALPY_D.get((e1, e2), ENTHALPY_D.get((e2, e1), 0)) * x[e1] * x[e2]) ** 2
                   for i, e1 in enumerate(elems) for e2 in elems[i + 1:])
    enth_std = np.sqrt(enth_sq) if enth_sq > 0 else 0.0
    omega    = (tm_mean * entropy / (abs(enthalpy) * 1000)) if enthalpy != 0 else 0.0
    xm       = sum(x[e] * ELECTRONEG_D[e]   for e in elems)
    xs       = np.sqrt(sum(x[e] * (ELECTRONEG_D[e] - xm) ** 2 for e in elems))
    vm       = sum(x[e] * VEC_D[e]          for e in elems)
    vs       = np.sqrt(sum(x[e] * (VEC_D[e] - vm) ** 2 for e in elems))
    km       = sum(x[e] * BULK_MODULI_D[e]  for e in elems)
    ks       = np.sqrt(sum(x[e] * (BULK_MODULI_D[e] - km) ** 2 for e in elems))
    mm       = sum(x[e] * MOLAR_MASSES_D[e] for e in elems)
    vol      = sum(x[e] * MOLAR_VOLS_D[e]   for e in elems)
    dens     = mm / vol if vol > 0 else 0.0

    return np.array([a_mean, delta, tm_mean, tm_std, entropy, enthalpy,
                     enth_std, omega, xm, xs, vm, vs, km, ks, dens])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1: Download ──────────────────────────────────────────────────────────
if os.path.exists(LOCAL_FILENAME):
    print(f"✓ Found local file: {LOCAL_FILENAME}")
else:
    print("Downloading from GitHub LFS...")
    urllib.request.urlretrieve(GITHUB_RAW_URL, LOCAL_FILENAME)
    print(f"✓ Downloaded: {os.path.getsize(LOCAL_FILENAME)/1024/1024:.1f} MB")

# ── Step 2: Load ──────────────────────────────────────────────────────────────
print("\nLoading dataset (202k rows — may take ~30 seconds)...")
df = pd.read_excel(LOCAL_FILENAME)
print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\n=== Dataset composition ===")
print(df['OG property'].value_counts())

# ── Step 3: Drop Be and La ────────────────────────────────────────────────────
for col in ['Be', 'La']:
    if col in df.columns:
        n_nonzero = (df[col] != 0).sum()
        print(f"\nDropping '{col}' — {n_nonzero:,} non-zero rows (excluded to keep 32-element space)")
        df = df.drop(columns=[col])

print(f"\n✓ Element columns: {len([c for c in ELEMENTS if c in df.columns])}/32")

# ── Step 4: Drop unused columns ───────────────────────────────────────────────
DROP_COLS = ['Calculated passive window', 'Reference', 'Test environment']
dropped = [c for c in DROP_COLS if c in df.columns]
if dropped:
    df = df.drop(columns=dropped)
    print(f"✓ Dropped unused columns: {dropped}")

# ── Step 5: Split mechanical vs corrosion ─────────────────────────────────────
mech = df[df['OG property'] == 'mechanical'].copy().reset_index(drop=True)
corr = df[df['OG property'] != 'mechanical'].copy().reset_index(drop=True)

print(f"\n=== Split ===")
print(f"  Mechanical rows : {len(mech):,}")
print(f"  Corrosion rows  : {len(corr):,}")
print(f"    Real (existing data) : {(corr['OG property']=='existing data').sum():,}")
print(f"    GAN synthetic        : {(corr['OG property']=='GAN').sum():,}")

# ── Step 6: Drop PBS / Hanks if present ───────────────────────────────────────
print(f"\n=== Corrosion electrolyte distribution ===")
print(corr['Electrolyte'].value_counts(dropna=False))

for elec in ELECTROLYTES_DROP:
    n = (corr['Electrolyte'] == elec).sum()
    if n > 0:
        print(f"Dropping {n} '{elec}' rows (n < 15)")
        corr = corr[corr['Electrolyte'] != elec].reset_index(drop=True)

print(f"\n✓ Corrosion rows after filtering: {len(corr):,}")

# ── Step 7: Corrosion target statistics ───────────────────────────────────────
print(f"\n=== Corrosion target statistics ===")
for name, col in [('Ecorr','Corrosion potential (mV vs SCE)'),
                  ('Epit', 'Pitting potential (mV vs SCE)'),
                  ('icorr','Corrosion current density (microA/cm2)')]:
    vals = corr[col].replace(0, np.nan).dropna()
    print(f"  {name}: n={len(vals):,}  min={vals.min():.4f}  "
          f"max={vals.max():.2f}  median={vals.median():.4f}")

# ── Step 8: Log₁₀ scale icorr ────────────────────────────────────────────────
print(f"\n=== Log₁₀ scaling icorr ===")
icorr_raw = corr['Corrosion current density (microA/cm2)'].copy()
print(f"  Zeros   : {(icorr_raw==0).sum():,} → NaN")
print(f"  Negative: {(icorr_raw<0).sum():,} → NaN")
print(f"  Positive: {(icorr_raw>0).sum():,} → log₁₀ transformed")

icorr_log = icorr_raw.copy().astype(float)
icorr_log[icorr_raw <= 0] = np.nan
icorr_log[icorr_raw  > 0] = np.log10(icorr_raw[icorr_raw > 0])
corr['icorr_log10'] = icorr_log

valid_log = icorr_log.dropna()
print(f"\n  log₁₀ stats: n={len(valid_log):,}  "
      f"min={valid_log.min():.3f}  max={valid_log.max():.3f}  "
      f"mean={valid_log.mean():.3f}  std={valid_log.std():.3f}")

# ── Step 9: Calculate empirical parameters for GAN rows ───────────────────────
print(f"\n=== Empirical parameter calculation ===")

# Rows needing calculation: those where ALL empirical cols are zero
emp_sum    = corr[EMP_COLS].abs().sum(axis=1)
needs_calc = corr.index[emp_sum == 0].tolist()
filled     = corr.index[emp_sum >  0].tolist()

print(f"  Already filled  : {len(filled):,} rows (real existing data)")
print(f"  Need calculation: {len(needs_calc):,} rows (GAN synthetic)")

if len(needs_calc) > 0:
    print(f"\n  Calculating empirical parameters in chunks of 10,000...")
    print(f"  (May take a few minutes for 200k rows)")

    chunk_size = 10000
    n_chunks   = (len(needs_calc) + chunk_size - 1) // chunk_size
    all_results = []

    for i in range(n_chunks):
        chunk_idx = needs_calc[i * chunk_size : (i + 1) * chunk_size]
        comp32    = corr.loc[chunk_idx, ELEMENTS].to_numpy(dtype=float)
        emp_arr   = np.array([calc_empirical_vector(row) for row in comp32])
        all_results.append(
            pd.DataFrame(emp_arr, index=chunk_idx, columns=EMP_COLS)
        )
        done = min((i + 1) * chunk_size, len(needs_calc))
        if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
            print(f"    {done:,}/{len(needs_calc):,} ({100*done/len(needs_calc):.0f}%)")

    emp_df = pd.concat(all_results)
    for col in EMP_COLS:
        corr.loc[needs_calc, col] = emp_df[col].values

    print(f"  ✓ Done")

# Verify
still_zero = (corr[EMP_COLS].abs().sum(axis=1) == 0).sum()
print(f"\n  Rows still missing empirical params after calculation: {still_zero}")
if still_zero == 0:
    print(f"  ✓ All {len(corr):,} corrosion rows now have empirical parameters")

# ── Step 10: Final summary ────────────────────────────────────────────────────
print(f"\n=== Final dataset summary ===")
print(f"  Mechanical rows  : {len(mech):,}")
print(f"  Corrosion rows   : {len(corr):,}")
print(f"    Ecorr valid    : {corr['Corrosion potential (mV vs SCE)'].replace(0,np.nan).dropna().shape[0]:,}")
print(f"    Epit  valid    : {corr['Pitting potential (mV vs SCE)'].replace(0,np.nan).dropna().shape[0]:,}")
print(f"    icorr valid    : {corr['icorr_log10'].dropna().shape[0]:,}")
print(f"  Output columns   : {corr.shape[1]}")

# ── Step 11: Save ─────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILENAME}...")
print("(Writing 200k rows to Excel — may take a few minutes...)")

all_df = pd.concat([mech, corr], ignore_index=True)

with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
    all_df.to_excel(writer, sheet_name='all',        index=False)
    mech.to_excel(  writer, sheet_name='mechanical', index=False)
    corr.to_excel(  writer, sheet_name='corrosion',  index=False)

size_mb = os.path.getsize(OUTPUT_FILENAME) / 1024 / 1024
print(f"✓ Saved: {OUTPUT_FILENAME} ({size_mb:.1f} MB)")
print(f"\n  Sheets:")
print(f"    'all'        : {len(all_df):,} rows")
print(f"    'mechanical' : {len(mech):,} rows")
print(f"    'corrosion'  : {len(corr):,} rows  ← use this for corrosion model training")
print(f"\n✅ Preprocessing complete — ready for step2_retrain_corr_models.py")

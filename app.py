"""
app.py  —  MPEA GAN-Augmented Corrosion + Mechanical Generative Design Tool
─────────────────────────────────────────────────────────────────────────────
Same NSGAN framework as the original app, but corrosion models (Ecorr, Epit,
icorr) are retrained on GAN-augmented data (~200k rows) giving dramatically
higher R²:
  Ecorr: 0.646 → 0.928  |  Epit: 0.775 → 0.983  |  icorr: 0.451 → 0.914

Mechanical models + phase classifiers + generator are loaded from the original
GitHub repo (no retraining needed — they are already excellent).
New corrosion models are loaded from this repo (models_corr_A / models_corr_B).

Model loading strategy:
  - Mechanical/phase/generator: downloaded from MPEA_corr_mech_app repo at startup
  - Corrosion regressors: loaded locally from models_corr_A / models_corr_B
"""

import io, os, warnings, json, urllib.request, tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import load
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="MPEA GAN-Augmented Corrosion Design",
                   page_icon="⚗️", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────
ELEMENTS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
            'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
            'Sn','Ta','Ti','V','W','Y','Zn','Zr']
MASSES  = [107.87,26.98,10.81,12.01,40.08,58.93,52.00,63.55,55.85,69.72,
           72.63,178.49,6.94,24.31,54.94,95.96,14.01,92.91,144.24,58.69,
           106.42,186.21,44.96,28.09,118.71,180.95,47.87,50.94,183.84,
           88.91,65.38,91.22]
VOLUMES = [10.27,10.00,4.39,5.29,26.20,6.67,7.23,7.11,7.09,11.80,
           13.63,13.44,13.02,14.00,7.35,9.38,13.54,10.83,20.59,6.59,
           8.56,8.86,15.00,12.06,16.29,10.85,10.64,8.32,9.47,19.88,
           9.16,14.02]
PROCESS_MAP = {
    'process_1': "As-cast / arc-melted",
    'process_2': "Arc-melted + artificial aging",
    'process_3': "Arc-melted + annealing",
    'process_4': "Powder metallurgy",
    'process_5': "Novel synthesis (ball milling etc.)",
    'process_6': "Arc-melted + wrought processing",
    'process_7': "Cryogenic treatments",
}
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']
OBJECTIVE_INFO = {
    'Tensile Strength' : ('maximize','MPa'),
    'Yield Strength'   : ('maximize','MPa'),
    'Elongation'       : ('maximize','%'),
    'Hardness'         : ('maximize','HV'),
    'Ecorr'            : ('maximize','mV vs SCE'),
    'Epit'             : ('maximize','mV vs SCE'),
    'icorr'            : ('minimize','µA/cm²'),
    'Density'          : ('minimize','g/cm³'),
    'FCC'              : ('maximize','probability'),
    'BCC'              : ('maximize','probability'),
    'HCP'              : ('maximize','probability'),
    'IM'               : ('maximize','probability'),
    'Aluminum Content' : ('maximize','molar ratio'),
}
PROP_KEY = {
    'Tensile Strength' : 'Tensile Strength (MPa)',
    'Yield Strength'   : 'Yield Strength (MPa)',
    'Elongation'       : 'Elongation (%)',
    'Hardness'         : 'Hardness (HV)',
    'Ecorr'            : 'Ecorr (mV vs SCE)',
    'Epit'             : 'Epit (mV vs SCE)',
    'icorr'            : 'icorr (µA/cm²)',
    'Density'          : 'Density (g/cm³)',
    'FCC'              : 'FCC probability',
    'BCC'              : 'BCC probability',
    'HCP'              : 'HCP probability',
    'IM'               : 'IM probability',
    'Aluminum Content' : 'Al molar fraction',
}

# ── URLs for mechanical models from original repo ─────────────────────────────
ORIG_REPO_RAW = (
    "https://media.githubusercontent.com/media/katrina-coder/"
    "MPEA_corr_mech_app/main"
)
MECH_FILES = {
    'generator'          : 'generator_net_MPEA.pt',
    'hardness'           : 'hardness_regressor.joblib',
    'tensile'            : 'tensile_regressor.joblib',
    'yield'              : 'yield_regressor.joblib',
    'elongation'         : 'elongation_regressor.joblib',
    'FCC_clf'            : 'FCC_classifier.joblib',
    'BCC_clf'            : 'BCC_classifier.joblib',
    'HCP_clf'            : 'HCP_classifier.joblib',
    'IM_clf'             : 'IM_classifier.joblib',
}
DB_FILE = 'Updated_MPEAs_Mech_CorrGAN_DB_processed.xlsx'

# ── Empirical parameter lookup tables ─────────────────────────────────────────
ATOMIC_RADII   = {'Ag':1.44,'Al':1.43,'B':0.87,'C':0.77,'Ca':1.97,'Co':1.25,'Cr':1.28,'Cu':1.28,'Fe':1.26,'Ga':1.22,'Ge':1.22,'Hf':1.59,'Li':1.52,'Mg':1.60,'Mn':1.26,'Mo':1.36,'N':0.75,'Nb':1.43,'Nd':1.82,'Ni':1.24,'Pd':1.37,'Re':1.37,'Sc':1.62,'Si':1.18,'Sn':1.40,'Ta':1.43,'Ti':1.47,'V':1.34,'W':1.37,'Y':1.80,'Zn':1.33,'Zr':1.60}
MELTING_TEMPS  = {'Ag':1235,'Al':933,'B':2349,'C':3823,'Ca':1115,'Co':1768,'Cr':2180,'Cu':1358,'Fe':1811,'Ga':303,'Ge':1211,'Hf':2506,'Li':454,'Mg':923,'Mn':1519,'Mo':2896,'N':63,'Nb':2750,'Nd':1297,'Ni':1728,'Pd':1828,'Re':3459,'Sc':1814,'Si':1687,'Sn':505,'Ta':3290,'Ti':1941,'V':2183,'W':3695,'Y':1799,'Zn':693,'Zr':2128}
ELECTRONEG_D   = {'Ag':1.93,'Al':1.61,'B':2.04,'C':2.55,'Ca':1.00,'Co':1.88,'Cr':1.66,'Cu':1.90,'Fe':1.83,'Ga':1.81,'Ge':2.01,'Hf':1.30,'Li':0.98,'Mg':1.31,'Mn':1.55,'Mo':2.16,'N':3.04,'Nb':1.60,'Nd':1.14,'Ni':1.91,'Pd':2.20,'Re':1.90,'Sc':1.36,'Si':1.90,'Sn':1.96,'Ta':1.50,'Ti':1.54,'V':1.63,'W':2.36,'Y':1.22,'Zn':1.65,'Zr':1.33}
VEC_D          = {'Ag':11,'Al':3,'B':3,'C':4,'Ca':2,'Co':9,'Cr':6,'Cu':11,'Fe':8,'Ga':3,'Ge':4,'Hf':4,'Li':1,'Mg':2,'Mn':7,'Mo':6,'N':5,'Nb':5,'Nd':4,'Ni':10,'Pd':10,'Re':7,'Sc':3,'Si':4,'Sn':4,'Ta':5,'Ti':4,'V':5,'W':6,'Y':3,'Zn':12,'Zr':4}
MOLAR_MASSES_D = {'Ag':107.87,'Al':26.98,'B':10.81,'C':12.01,'Ca':40.08,'Co':58.93,'Cr':52.00,'Cu':63.55,'Fe':55.85,'Ga':69.72,'Ge':72.63,'Hf':178.49,'Li':6.94,'Mg':24.31,'Mn':54.94,'Mo':95.96,'N':14.01,'Nb':92.91,'Nd':144.24,'Ni':58.69,'Pd':106.42,'Re':186.21,'Sc':44.96,'Si':28.09,'Sn':118.71,'Ta':180.95,'Ti':47.87,'V':50.94,'W':183.84,'Y':88.91,'Zn':65.38,'Zr':91.22}
MOLAR_VOLS_D   = {'Ag':10.27,'Al':10.00,'B':4.39,'C':5.29,'Ca':26.20,'Co':6.67,'Cr':7.23,'Cu':7.11,'Fe':7.09,'Ga':11.80,'Ge':13.63,'Hf':13.44,'Li':13.02,'Mg':14.00,'Mn':7.35,'Mo':9.38,'N':13.54,'Nb':10.83,'Nd':20.59,'Ni':6.59,'Pd':8.56,'Re':8.86,'Sc':15.00,'Si':12.06,'Sn':16.29,'Ta':10.85,'Ti':10.64,'V':8.32,'W':9.47,'Y':19.88,'Zn':9.16,'Zr':14.02}
LATTICE_D      = {'Ag':4.09,'Al':4.05,'B':5.06,'C':3.57,'Ca':5.58,'Co':2.51,'Cr':2.88,'Cu':3.62,'Fe':2.87,'Ga':4.52,'Ge':5.66,'Hf':3.20,'Li':3.51,'Mg':3.21,'Mn':8.91,'Mo':3.15,'N':4.04,'Nb':3.30,'Nd':3.66,'Ni':3.52,'Pd':3.89,'Re':2.76,'Sc':3.31,'Si':5.43,'Sn':5.83,'Ta':3.31,'Ti':2.95,'V':3.02,'W':3.16,'Y':3.65,'Zn':2.66,'Zr':3.23}
BULK_MODULI_D  = {'Ag':100,'Al':76,'B':320,'C':443,'Ca':17,'Co':180,'Cr':160,'Cu':140,'Fe':170,'Ga':59,'Ge':75,'Hf':110,'Li':11,'Mg':45,'Mn':120,'Mo':230,'N':0,'Nb':170,'Nd':32,'Ni':180,'Pd':180,'Re':370,'Sc':57,'Si':98,'Sn':58,'Ta':200,'Ti':110,'V':160,'W':310,'Y':41,'Zn':70,'Zr':94}
ENTHALPY_D     = {('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1,('Al','Fe'):-11,('Al','Hf'):-45,('Al','Mg'):-2,('Al','Mn'):-19,('Al','Mo'):-22,('Al','Nb'):-18,('Al','Ni'):-22,('Al','Si'):-19,('Al','Ta'):-19,('Al','Ti'):-30,('Al','V'):-16,('Al','W'):-16,('Al','Zr'):-44,('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):0,('Co','Mn'):0,('Co','Mo'):-5,('Co','Nb'):-25,('Co','Ni'):0,('Co','Ti'):-28,('Co','V'):-14,('Co','W'):-1,('Co','Zr'):-41,('Cr','Cu'):12,('Cr','Fe'):-1,('Cr','Mn'):2,('Cr','Mo'):0,('Cr','Nb'):-7,('Cr','Ni'):-7,('Cr','Si'):-37,('Cr','Ta'):-7,('Cr','Ti'):-7,('Cr','V'):-2,('Cr','W'):0,('Cr','Zr'):-12,('Cu','Fe'):13,('Cu','Mn'):4,('Cu','Mo'):19,('Cu','Ni'):4,('Cu','Ti'):-9,('Cu','Zr'):-23,('Fe','Mn'):0,('Fe','Mo'):-2,('Fe','Nb'):-16,('Fe','Ni'):-2,('Fe','Si'):-35,('Fe','Ta'):-15,('Fe','Ti'):-17,('Fe','V'):-7,('Fe','W'):-6,('Fe','Zr'):-25,('Mn','Mo'):0,('Mn','Ni'):-8,('Mn','Ti'):-8,('Mn','V'):-1,('Mo','Nb'):-6,('Mo','Ni'):-7,('Mo','Si'):-38,('Mo','Ta'):-5,('Mo','Ti'):-4,('Mo','V'):-5,('Mo','W'):0,('Mo','Zr'):-6,('Nb','Ni'):-30,('Nb','Si'):-56,('Nb','Ta'):0,('Nb','Ti'):-2,('Nb','V'):-2,('Nb','W'):-8,('Nb','Zr'):4,('Ni','Si'):-40,('Ni','Ta'):-24,('Ni','Ti'):-35,('Ni','V'):-18,('Ni','W'):-3,('Ni','Zr'):-49,('Si','Ta'):-45,('Si','Ti'):-66,('Si','V'):-48,('Si','W'):-37,('Si','Zr'):-84,('Ta','Ti'):-4,('Ta','V'):-1,('Ta','W'):-7,('Ti','V'):-2,('Ti','W'):-27,('Ti','Zr'):0,('V','W'):-8,('V','Zr'):-4,('W','Zr'):-27}
R_GAS = 8.314

def calc_empirical_vector(comp32):
    x = {ELEMENTS[i]: comp32[i] for i in range(32) if comp32[i] > 1e-6}
    if not x: return np.zeros(15)
    total = sum(x.values()); x = {e: v/total for e, v in x.items()}; elems = list(x.keys())
    a_mean  = sum(x[e]*LATTICE_D[e]    for e in elems)
    r_mean  = sum(x[e]*ATOMIC_RADII[e] for e in elems)
    delta   = 100*np.sqrt(sum(x[e]*(1-ATOMIC_RADII[e]/r_mean)**2 for e in elems))
    tm_mean = sum(x[e]*MELTING_TEMPS[e]  for e in elems)
    tm_std  = np.sqrt(sum(x[e]*(MELTING_TEMPS[e]-tm_mean)**2 for e in elems))
    entropy = -R_GAS*sum(xi*np.log(xi) for xi in x.values())
    enthalpy= sum(4*ENTHALPY_D.get((e1,e2),ENTHALPY_D.get((e2,e1),0))*x[e1]*x[e2]
                  for i,e1 in enumerate(elems) for e2 in elems[i+1:])
    enth_sq = sum((4*ENTHALPY_D.get((e1,e2),ENTHALPY_D.get((e2,e1),0))*x[e1]*x[e2])**2
                  for i,e1 in enumerate(elems) for e2 in elems[i+1:])
    enth_std= np.sqrt(enth_sq) if enth_sq > 0 else 0.0
    omega   = (tm_mean*entropy/(abs(enthalpy)*1000)) if enthalpy != 0 else 0.0
    xm      = sum(x[e]*ELECTRONEG_D[e] for e in elems)
    xs      = np.sqrt(sum(x[e]*(ELECTRONEG_D[e]-xm)**2 for e in elems))
    vm      = sum(x[e]*VEC_D[e] for e in elems)
    vs      = np.sqrt(sum(x[e]*(VEC_D[e]-vm)**2 for e in elems))
    km      = sum(x[e]*BULK_MODULI_D[e] for e in elems)
    ks      = np.sqrt(sum(x[e]*(BULK_MODULI_D[e]-km)**2 for e in elems))
    mm      = sum(x[e]*MOLAR_MASSES_D[e] for e in elems)
    vol     = sum(x[e]*MOLAR_VOLS_D[e]  for e in elems)
    dens    = mm/vol if vol > 0 else 0.0
    return np.array([a_mean,delta,tm_mean,tm_std,entropy,enthalpy,enth_std,omega,xm,xs,vm,vs,km,ks,dens])

def build_mech_features(alloy39, phase4):
    return np.concatenate([alloy39[:32], alloy39[32:], calc_empirical_vector(alloy39[:32]), phase4])

def build_corr_features(alloy39, phase4, elec_onehot_7, conc_norm):
    return np.concatenate([build_mech_features(alloy39, phase4), elec_onehot_7, [conc_norm]])

# ── Generator architecture ────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
        )
    def forward(self, z): return self.model(z)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_mech_models(pipeline):
    """Download mechanical models from original repo at first run, cache locally."""
    subdir  = f"models_{'A' if pipeline == 'A' else 'B'}"
    cache   = os.path.join(tempfile.gettempdir(), f"mpea_mech_{pipeline}")
    os.makedirs(cache, exist_ok=True)

    for key, fname in MECH_FILES.items():
        dest = os.path.join(cache, fname)
        if not os.path.exists(dest):
            url = f"{ORIG_REPO_RAW}/{subdir}/{fname}"
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                st.error(f"Failed to download {fname}: {e}")
                return None

    # Load generator
    gen = Generator()
    gen.load_state_dict(torch.load(os.path.join(cache, 'generator_net_MPEA.pt'),
                                   map_location='cpu'))
    gen.eval()

    # Load mechanical regressors
    regressors_mech = {
        'Hardness'        : load(os.path.join(cache, 'hardness_regressor.joblib')),
        'Tensile Strength': load(os.path.join(cache, 'tensile_regressor.joblib')),
        'Yield Strength'  : load(os.path.join(cache, 'yield_regressor.joblib')),
        'Elongation'      : load(os.path.join(cache, 'elongation_regressor.joblib')),
    }

    # Load phase classifiers
    classifiers = {p: load(os.path.join(cache, f'{p}_classifier.joblib'))
                   for p in ['FCC','BCC','HCP','IM']}

    return gen, regressors_mech, classifiers


@st.cache_resource
def load_corr_models(pipeline):
    """Load GAN-augmented corrosion models from local repo."""
    subdir = f"models_corr_{'A' if pipeline == 'A' else 'B'}"
    regressors_corr = {
        'Ecorr': load(os.path.join(subdir, 'ecorr_regressor.joblib')),
        'Epit' : load(os.path.join(subdir, 'epit_regressor.joblib')),
        'icorr': load(os.path.join(subdir, 'icorr_regressor.joblib')),
    }
    return regressors_corr


@st.cache_data
def load_dataset_bounds():
    df = pd.read_excel(DB_FILE, sheet_name='mechanical')
    ELEM_COLS    = ELEMENTS
    PROCESS_COLS = ['process_1','process_2','process_3','process_4',
                    'process_5','process_6','process_7']
    comp = df[ELEM_COLS].to_numpy(dtype=float)
    return np.min(comp, axis=0), np.max(comp, axis=0), PROCESS_COLS


# ── Optimisation ──────────────────────────────────────────────────────────────
class AlloyProblem(Problem):
    def __init__(self, objectives, generator, reg_mech, reg_corr, classifiers,
                 comp_min, comp_max, elec_onehot, conc_norm,
                 max_elements=10, banned_indices=None, required_indices=None):
        n_constr = 1
        if banned_indices:  n_constr += len(banned_indices)
        if required_indices: n_constr += 1
        super().__init__(n_var=10, n_obj=len(objectives),
                         n_ieq_constr=n_constr, xl=-3.0, xu=3.0)
        self.objectives       = objectives
        self.generator        = generator
        self.reg_mech         = reg_mech
        self.reg_corr         = reg_corr
        self.classifiers      = classifiers
        self.comp_min         = comp_min
        self.comp_max         = comp_max
        self.elec_onehot      = elec_onehot
        self.conc_norm        = conc_norm
        self.max_elements     = max_elements
        self.banned_indices   = banned_indices  or []
        self.required_indices = required_indices or []

    def _evaluate(self, x, out, *args, **kwargs):
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            raw = self.generator(x_t).numpy()
        alloys39 = raw.copy()
        alloys39[:, :32] = raw[:, :32] * self.comp_max + self.comp_min

        zeros4 = np.zeros(4)
        base58 = np.array([np.concatenate([a[:32], a[32:],
                           calc_empirical_vector(a[:32]), zeros4]) for a in alloys39])
        phase4 = np.column_stack([self.classifiers[p].predict(base58).astype(float)
                                   for p in ['FCC','BCC','HCP','IM']])
        mf = np.array([build_mech_features(alloys39[i], phase4[i])
                        for i in range(len(alloys39))])
        cf = np.array([build_corr_features(alloys39[i], phase4[i],
                        self.elec_onehot, self.conc_norm)
                        for i in range(len(alloys39))])

        masses_a = np.array(MASSES); volumes_a = np.array(VOLUMES)
        comp32_n = alloys39[:, :32].copy()
        rs = comp32_n.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        comp32_n /= rs
        densities = (comp32_n * masses_a).sum(1) / (comp32_n * volumes_a).sum(1)

        def get_obj(name):
            if name == 'Tensile Strength': return -self.reg_mech['Tensile Strength'].predict(mf)
            if name == 'Yield Strength':   return -self.reg_mech['Yield Strength'].predict(mf)
            if name == 'Elongation':       return -self.reg_mech['Elongation'].predict(mf)
            if name == 'Hardness':         return -self.reg_mech['Hardness'].predict(mf)
            if name == 'Ecorr':            return -self.reg_corr['Ecorr'].predict(cf)
            if name == 'Epit':             return -self.reg_corr['Epit'].predict(cf)
            if name == 'icorr':            return  self.reg_corr['icorr'].predict(cf)
            if name == 'Density':          return densities
            if name == 'Aluminum Content': return -alloys39[:, 1]
            if name in ('FCC','BCC','HCP','IM'):
                return -self.classifiers[name].predict(mf).astype(float)
            return np.zeros(len(alloys39))

        out['F'] = np.column_stack([get_obj(o) for o in self.objectives])

        # G1: max elements
        n_elements = (alloys39[:, :32] > 0.005).sum(axis=1).astype(float)
        G = [(n_elements - self.max_elements)]
        # G2: banned elements
        for idx in self.banned_indices:
            G.append(alloys39[:, idx] - 0.005)
        # G3: required elements
        if self.required_indices:
            max_req = alloys39[:, self.required_indices].max(axis=1)
            G.append(-max_req + 0.005)
        out['G'] = np.column_stack(G)


def decode_results(res_X, generator, comp_min, comp_max,
                   reg_mech, reg_corr, classifiers,
                   proc_names, elec_onehot, conc_norm):
    res_X = np.atleast_2d(res_X)
    x_t = torch.tensor(res_X, dtype=torch.float32)
    with torch.no_grad():
        raw = generator(x_t).numpy()
    alloys39 = raw.copy()
    alloys39[:, :32] = raw[:, :32] * comp_max + comp_min
    rs = alloys39[:, :32].sum(1, keepdims=True); rs[rs == 0] = 1
    alloys39[:, :32] /= rs

    zeros4 = np.zeros(4)
    base58 = np.array([np.concatenate([a[:32], a[32:],
                       calc_empirical_vector(a[:32]), zeros4]) for a in alloys39])
    phase4     = np.column_stack([classifiers[p].predict(base58).astype(float)
                                   for p in ['FCC','BCC','HCP','IM']])
    phase_proba = np.column_stack([classifiers[p].predict_proba(base58)[:, 1]
                                    for p in ['FCC','BCC','HCP','IM']])
    mf = np.array([build_mech_features(alloys39[i], phase4[i])
                    for i in range(len(alloys39))])
    cf = np.array([build_corr_features(alloys39[i], phase4[i], elec_onehot, conc_norm)
                    for i in range(len(alloys39))])

    names, n_els, al_fracs = [], [], []
    for comp in alloys39[:, :32]:
        parts = [f"{ELEMENTS[j]}{comp[j]:.3f}" for j in range(32) if comp[j] > 0.005]
        names.append("".join(parts)); n_els.append(len(parts))
        al_fracs.append(round(comp[1], 4))

    proc_idx = np.argmax(alloys39[:, 32:], axis=1)
    procs    = [PROCESS_MAP.get(proc_names[i], "Unknown") for i in proc_idx]

    ma = np.array(MASSES); va = np.array(VOLUMES)
    c32 = alloys39[:, :32].copy(); c32 /= c32.sum(1, keepdims=True).clip(1e-9)
    densities = (c32 * ma).sum(1) / (c32 * va).sum(1)

    icorr_vals = np.clip(10 ** reg_corr['icorr'].predict(cf), 0, 1e6)

    phase_lbls = ['FCC','BCC','HCP','IM']
    phases = []
    for i in range(len(alloys39)):
        present = [phase_lbls[j] for j in range(4) if phase4[i, j] > 0]
        phases.append("+".join(present) if present
                      else f"{phase_lbls[int(np.argmax(phase_proba[i]))]} (dominant)")

    return pd.DataFrame({
        'Alloy Composition':      names,
        'N Elements':             n_els,
        'Processing Method':      procs,
        'Predicted Phase':        phases,
        'Hardness (HV)':          np.round(reg_mech['Hardness'].predict(mf),         2),
        'Tensile Strength (MPa)': np.round(reg_mech['Tensile Strength'].predict(mf), 2),
        'Yield Strength (MPa)':   np.round(reg_mech['Yield Strength'].predict(mf),   2),
        'Elongation (%)':         np.round(reg_mech['Elongation'].predict(mf),       2),
        'Ecorr (mV vs SCE)':      np.round(reg_corr['Ecorr'].predict(cf),            2),
        'Epit (mV vs SCE)':       np.round(reg_corr['Epit'].predict(cf),             2),
        'icorr (µA/cm²)':         np.round(icorr_vals,                               4),
        'Density (g/cm³)':        np.round(densities,                                3),
        'FCC probability':        np.round(phase_proba[:, 0],                        3),
        'BCC probability':        np.round(phase_proba[:, 1],                        3),
        'HCP probability':        np.round(phase_proba[:, 2],                        3),
        'IM probability':         np.round(phase_proba[:, 3],                        3),
        'Al molar fraction':      al_fracs,
    })


def run_optimisation(objectives, pop_size, n_gen, seed,
                     generator, reg_mech, reg_corr, classifiers,
                     comp_min, comp_max, proc_names,
                     elec_onehot, conc_norm,
                     max_elements=10, banned_indices=None, required_indices=None):
    problem   = AlloyProblem(objectives, generator, reg_mech, reg_corr, classifiers,
                              comp_min, comp_max, elec_onehot, conc_norm,
                              max_elements, banned_indices, required_indices)
    algorithm = NSGA2(pop_size=pop_size, mutation=PM(prob=0.1, eta=20))
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   save_history=False, seed=int(seed), verbose=False)
    if res.X is None:
        return None
    return decode_results(res.X, generator, comp_min, comp_max,
                          reg_mech, reg_corr, classifiers,
                          proc_names, elec_onehot, conc_norm)


def post_filter(df, banned_indices):
    if df is None or not banned_indices:
        return df
    mask = []
    for name in df['Alloy Composition']:
        ok = True
        for idx in banned_indices:
            el = ELEMENTS[idx]
            if el in name:
                try:
                    pos = name.index(el) + len(el)
                    if float(name[pos:pos+5]) > 0.005:
                        ok = False; break
                except (ValueError, IndexError):
                    pass
        mask.append(ok)
    filtered = df[mask].reset_index(drop=True)
    return filtered if len(filtered) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚗️ MPEA GAN-Augmented Corrosion + Mechanical Design")
st.markdown("""
Generates novel MPEAs optimised for **mechanical and corrosion** properties using the NSGAN
framework. Corrosion models are trained on **GAN-augmented data (~200k rows)** following
Ghorbani et al. (2025), achieving dramatically improved accuracy:
**Ecorr R²=0.928 · Epit R²=0.983 · icorr R²=0.914** (vs 0.646 / 0.775 / 0.451 without GAN data).
""")

comp_min, comp_max, proc_names = load_dataset_bounds()

with st.sidebar:
    st.header("⚙️ Settings")

    pipeline_choice = st.radio("Pipeline",
        ["A — Separate models", "B — Imputed unified models", "A vs B — Compare both"],
        index=0)

    st.divider()
    st.subheader("🌊 Test Environment")
    selected_electrolyte = st.selectbox("Electrolyte", ELECTROLYTES, index=0)
    selected_conc = st.number_input("Concentration (M)", min_value=0.05,
        max_value=6.0, value=0.6, step=0.05)
    elec_onehot = np.array([1.0 if e == selected_electrolyte else 0.0 for e in ELECTROLYTES])
    conc_norm   = selected_conc / 6.0

    st.divider()
    st.subheader("🎯 Objectives")
    selected_objectives = st.multiselect("Optimisation Objectives",
        list(OBJECTIVE_INFO.keys()),
        default=["Tensile Strength", "Elongation", "icorr"])

    if selected_objectives:
        st.dataframe(pd.DataFrame([
            {'Objective': o,
             'Direction': '↑ Max' if OBJECTIVE_INFO[o][0]=='maximize' else '↓ Min',
             'Unit': OBJECTIVE_INFO[o][1]}
            for o in selected_objectives
        ]), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🧪 Alloy Constraints")
    max_elements = st.slider("Max number of elements", 2, 10, 7,
        help="Enforced as NSGA-II inequality constraint. Training range: 2–10 (mean 5.3).")

    allowed_elements = st.multiselect("Allowed elements (pool)",
        options=ELEMENTS, default=ELEMENTS)
    required_elements = st.multiselect("Required elements",
        options=allowed_elements if allowed_elements else ELEMENTS, default=[],
        help="Optional. At least one must appear (>0.005 mol fraction). Leave empty for no requirement.")

    banned_indices   = [ELEMENTS.index(e) for e in ELEMENTS if e not in allowed_elements]
    required_indices = [ELEMENTS.index(e) for e in required_elements]

    if len(allowed_elements) == 0:
        st.error("⚠️ No elements allowed — add at least one.")
    elif len(allowed_elements) < 2:
        st.warning("⚠️ Only 1 element allowed — models need at least 2.")
    else:
        if banned_indices:
            st.caption(f"🚫 {len(banned_indices)} banned: "
                       f"{', '.join(e for e in ELEMENTS if e not in allowed_elements)}")
        if required_indices:
            st.caption(f"✅ At least one required: {', '.join(required_elements)}")
        if not banned_indices and not required_indices:
            st.caption("No element constraints — all 32 elements allowed freely.")

    st.divider()
    pop_size = st.slider("Population Size", 10, 200, 50, 10)
    n_gen    = st.slider("Generations",     10, 500, 200, 10)
    seed_val = st.number_input("Random Seed", 0, 9999, 2)

    run_btn = st.button("🚀 Start Optimisation", type="primary",
                        use_container_width=True,
                        disabled=len(selected_objectives) < 2)
    if len(selected_objectives) < 2:
        st.warning("Select at least 2 objectives.")

# ── Model R² table ────────────────────────────────────────────────────────────
with st.expander("📊 Model R² performance summary", expanded=False):
    r2_data = {
        'Property'    : ['Hardness','Yield Strength','Tensile','Elongation',
                         'Ecorr','Epit','icorr (log₁₀)'],
        'Original A R²' : [0.802, 0.574, 0.662, 0.525, 0.646, 0.775, 0.451],
        'GAN-Aug A R²'  : [0.802, 0.574, 0.662, 0.525, 0.928, 0.983, 0.914],
        'Original B R²' : [0.921, 0.756, 0.708, 0.617, 0.742, 0.856, 0.598],
        'GAN-Aug B R²'  : [0.921, 0.756, 0.708, 0.617, 0.928, 0.983, 0.914],
        'Features'      : [
            '58 (32 element + 7 processing + 15 empirical + 4 phase)',
            '58 (32 element + 7 processing + 15 empirical + 4 phase)',
            '58 (32 element + 7 processing + 15 empirical + 4 phase)',
            '58 (32 element + 7 processing + 15 empirical + 4 phase)',
            '58 features + 7 electrolyte + electrolyte concentration',
            '58 features + 7 electrolyte + electrolyte concentration',
            '58 features + 7 electrolyte + electrolyte concentration',
        ],
        'Note' : ['Strongly influenced by processing route & phase',
                  'Sensitive to phase structure (FCC vs BCC)',
                  'Correlates strongly with yield strength',
                  'Trade-off with strength; phase-dependent',
                  'GAN augmentation: 616 → 200,565 training rows',
                  'GAN augmentation: 334 → 200,301 training rows',
                  'GAN augmentation: 590 → 200,565 training rows'],
    }
    st.dataframe(pd.DataFrame(r2_data), hide_index=True, use_container_width=True)
    st.caption("Mechanical models unchanged from original app. Corrosion models retrained with "
               "GAN-augmented data (Ghorbani et al. 2025, npj Materials Degradation). "
               "Processing and electrolyte categorical features encoded via one-hot encoding. "
               "PBS and Hanks excluded (n<15).")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn and len(selected_objectives) >= 2:
    run_A = "A" in pipeline_choice
    run_B = "B" in pipeline_choice

    if len(allowed_elements) == 0:
        st.error("No elements allowed — please keep at least one element in the pool.")
        st.stop()

    def post_filter_wrap(df):
        return post_filter(df, banned_indices)

    progress  = st.progress(0, "Starting optimisation…")
    result_A  = result_B = None

    if run_A:
        try:
            mech_A = download_mech_models("A")
            corr_A = load_corr_models("A")
            if mech_A is None:
                st.error("Failed to load Pipeline A mechanical models.")
            else:
                gen_A, reg_mech_A, clf_A = mech_A
                progress.progress(10, f"Pipeline A — NSGA-II ({n_gen} generations)…")
                result_A = run_optimisation(
                    selected_objectives, pop_size, n_gen, seed_val,
                    gen_A, reg_mech_A, corr_A, clf_A,
                    comp_min, comp_max, proc_names,
                    elec_onehot, conc_norm, max_elements,
                    banned_indices, required_indices)
                result_A = post_filter_wrap(result_A)
                if result_A is None:
                    st.warning("Pipeline A: No feasible solutions found. "
                               "Try relaxing element constraints or increasing generations.")
        except Exception as e:
            st.error(f"Pipeline A failed: {e}")

    if run_B:
        try:
            mech_B = download_mech_models("B")
            corr_B = load_corr_models("B")
            if mech_B is None:
                st.error("Failed to load Pipeline B mechanical models.")
            else:
                gen_B, reg_mech_B, clf_B = mech_B
                progress.progress(55 if run_A else 10,
                                  f"Pipeline B — NSGA-II ({n_gen} generations)…")
                result_B = run_optimisation(
                    selected_objectives, pop_size, n_gen, seed_val,
                    gen_B, reg_mech_B, corr_B, clf_B,
                    comp_min, comp_max, proc_names,
                    elec_onehot, conc_norm, max_elements,
                    banned_indices, required_indices)
                result_B = post_filter_wrap(result_B)
                if result_B is None:
                    st.warning("Pipeline B: No feasible solutions found. "
                               "Try relaxing element constraints or increasing generations.")
        except Exception as e:
            st.error(f"Pipeline B failed: {e}")

    progress.progress(100, "Done!")
    progress.empty()
    st.session_state.update({
        'result_A': result_A, 'result_B': result_B,
        'objectives': selected_objectives,
        'electrolyte': selected_electrolyte, 'conc': selected_conc,
        'max_elements': max_elements,
        'allowed_elements': allowed_elements,
        'required_elements': required_elements,
    })

# ── Display results ───────────────────────────────────────────────────────────
if 'result_A' in st.session_state or 'result_B' in st.session_state:
    result_A   = st.session_state.get('result_A')
    result_B   = st.session_state.get('result_B')
    objectives = st.session_state.get('objectives', [])
    max_el     = st.session_state.get('max_elements', 10)

    if result_A is None and result_B is None:
        st.stop()

    st.divider()
    allowed_el  = st.session_state.get('allowed_elements', ELEMENTS)
    required_el = st.session_state.get('required_elements', [])
    banned_el   = [e for e in ELEMENTS if e not in allowed_el]
    info_parts  = [f"🌊 **{st.session_state.get('electrolyte','')}** "
                   f"at {st.session_state.get('conc','')} M",
                   f"max **{max_el}** elements"]
    if banned_el:   info_parts.append(f"🚫 banned: {', '.join(banned_el)}")
    if required_el: info_parts.append(f"✅ required: {', '.join(required_el)}")
    st.info("  ·  ".join(info_parts))

    # ── Scatter plots ─────────────────────────────────────────────────────────
    st.subheader("📈 Pareto Fronts")

    def get_pairs(objectives, df):
        valid   = [o for o in objectives if PROP_KEY.get(o) in df.columns]
        mech_o  = [o for o in valid if o in ('Tensile Strength','Yield Strength','Elongation','Hardness')]
        corr_o  = [o for o in valid if o in ('Ecorr','Epit','icorr')]
        other_o = [o for o in valid if o not in mech_o and o not in corr_o]
        pairs   = []
        if len(mech_o) >= 2: pairs.append((mech_o[0], mech_o[1], "Mechanical"))
        if mech_o and corr_o: pairs.append((mech_o[0], corr_o[0], "Mech vs Corrosion"))
        if len(corr_o) >= 2: pairs.append((corr_o[0], corr_o[1], "Corrosion"))
        if other_o and valid:
            base = mech_o[0] if mech_o else corr_o[0] if corr_o else valid[0]
            for o in other_o:
                if o != base: pairs.append((base, o, f"{base} vs {o}"))
        if not pairs and len(valid) >= 2:
            pairs.append((valid[0], valid[1], "Objectives"))
        return pairs

    ref_df = result_A if result_A is not None else result_B
    pairs  = get_pairs(objectives, ref_df)
    both   = result_A is not None and result_B is not None

    pipeline_results = []
    if result_A is not None: pipeline_results.append((result_A, "Pipeline A", "#1f77b4"))
    if result_B is not None: pipeline_results.append((result_B, "Pipeline B", "#ff7f0e"))
    n_pipelines = len(pipeline_results)

    if not pairs:
        st.info("No plottable objective pairs — select at least 2 objectives.")
    else:
        for x_obj, y_obj, title in pairs:
            xk, yk = PROP_KEY[x_obj], PROP_KEY[y_obj]
            fig, axes = plt.subplots(1, n_pipelines, figsize=(12, 4),
                                     sharey=(n_pipelines > 1), squeeze=False)
            axes = axes[0]
            for ax, (res, lbl, col) in zip(axes, pipeline_results):
                if xk in res.columns and yk in res.columns:
                    ax.scatter(res[xk], res[yk], c=col, alpha=0.7,
                               edgecolors='white', s=60)
                ax.set_xlabel(xk); ax.set_ylabel(yk)
                ax.set_title(f"{title} — {lbl}"); ax.grid(True, ls='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    # ── Results tables ────────────────────────────────────────────────────────
    st.divider()
    HIDE_COLS = ['N Elements','FCC probability','BCC probability',
                 'HCP probability','IM probability','Al molar fraction']

    def display_df(df):
        return df.drop(columns=[c for c in HIDE_COLS if c in df.columns]
                       ).reset_index(drop=True)

    if both:
        t1, t2 = st.tabs(["Pipeline A", "Pipeline B"])
        with t1:
            st.caption(f"{len(result_A)} alloys · ≤ {max_el} elements")
            st.dataframe(display_df(result_A), use_container_width=True)
        with t2:
            st.caption(f"{len(result_B)} alloys · ≤ {max_el} elements")
            st.dataframe(display_df(result_B), use_container_width=True)
    elif result_A is not None:
        st.subheader(f"Pipeline A — {len(result_A)} alloys")
        st.dataframe(display_df(result_A), use_container_width=True)
    elif result_B is not None:
        st.subheader(f"Pipeline B — {len(result_B)} alloys")
        st.dataframe(display_df(result_B), use_container_width=True)

    st.caption("ℹ️ Training data: 2–10 elements (mean 5.3). Most reliable range: 4–7 elements.")

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        if result_A is not None: result_A.to_excel(w, sheet_name='Pipeline_A', index=False)
        if result_B is not None: result_B.to_excel(w, sheet_name='Pipeline_B', index=False)
        pd.DataFrame(r2_data).to_excel(w, sheet_name='Model_R2', index=False)
    buf.seek(0)
    st.download_button("⬇️ Download Excel (all results)", data=buf,
                       file_name="MPEA_GAN_corr_optimised.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

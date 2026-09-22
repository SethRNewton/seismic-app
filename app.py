# -*- coding: utf-8 -*-
"""
Streamlit app: find molecular-ion evidence in PCI data for EI candidates.

Run locally:
    pip install streamlit pandas numpy openpyxl
    streamlit run app.py

Pipeline (rolled up from the three scripts in this folder):
    1. Parse the EI MSP file and attach EI_spectrum to each row of the EI table.
    2. Parse the PCI MSP, compute Kovats RI from RT for each PCI feature, clean
       up the sample-area column names, attach PCI_spectrum.
    3. For every EI candidate row, search nearby PCI features (RI window, or
       RT fallback when RI is blank) for [M]+ or [M+H]+ within ppm tolerance;
       reject 13C isotopes; report ethyl / allyl adducts.
    4. Optionally blank-correct the sample abundances (see apply_blank_correction).
    5. Offer the resulting CSV for download.

Feature matrices may be supplied as .csv or .xlsx (chosen by file extension).
Sample-name data prep (e.g. the LAFires field-blank renaming) is done outside
this app; see dataprep_rename_samples.py.
"""

import io
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEMO_EI_MATRIX  = os.path.join(APP_DIR, 'SEISMIC_DEMO_EI_Matrix_260914.csv')
DEMO_EI_MSP     = os.path.join(APP_DIR, 'SEISMIC_DEMO_EI_Spectra_260914.msp')
DEMO_PCI_MATRIX = os.path.join(APP_DIR, 'SEISMIC_DEMO_PCI_Matrix_260914.csv')
DEMO_PCI_MSP    = os.path.join(APP_DIR, 'SEISMIC_DEMO_PCI_Spectra_260914.msp')

ALKANES_CSV = os.path.join(APP_DIR, 'nAlkanes_RetentionTimes.csv')

# Blank/sample-name data prep (e.g. the LAFires field-blank renaming) is done
# outside this app; see dataprep_rename_samples.py. Feature matrices are
# expected to arrive with their final sample names already in place.

FEATURE_SHEET = 'GCCICompounds'

# ---- Molecular-ion search tunables ----
RI_TOLERANCE      = 10.0
RT_TOLERANCE      = 0.03
PPM_TOLERANCE     = 5.0
ISOTOPE_SPACING   = 1.00335
ISOTOPE_MASS_TOL  = 0.005
ISOTOPE_INT_RATIO = 3.0

# ---- Monoisotopic constants (u) ----
M_ELECTRON = 0.00054857990943
M_H        = 1.00782503207
M_C        = 12.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def read_bytes(source):
    """Accept either a filesystem path or a Streamlit UploadedFile; return bytes."""
    if hasattr(source, 'getvalue'):
        return source.getvalue()
    with open(source, 'rb') as f:
        return f.read()


def read_text(source):
    return read_bytes(source).decode('utf-8', errors='replace')


def source_name(source):
    """Best-effort filename for a path or a Streamlit UploadedFile."""
    if hasattr(source, 'name'):
        return source.name
    return str(source)


def read_feature_matrix(source, sheet_name=0):
    """Read a feature matrix, choosing the reader by file extension.

    .xlsx / .xls -> pd.read_excel (using sheet_name); anything else -> pd.read_csv.
    Accepts a filesystem path or a Streamlit UploadedFile.
    """
    name = source_name(source).lower()
    data = read_bytes(source)
    if name.endswith(('.xlsx', '.xls', '.xlsm')):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
    return pd.read_csv(io.BytesIO(data))


def parse_msp_spectra(text):
    """Return spectra in file order: list of [(mass, intensity), ...]."""
    blocks = re.split(r'\r?\n\s*\r?\n', text.strip())
    spectra = []
    for block in blocks:
        lines = block.splitlines()
        peak_line_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('num peaks:'):
                peak_line_idx = i
                break
        if peak_line_idx is None:
            continue

        pair_text = ' '.join(lines[peak_line_idx + 1:])
        pairs = []
        for tok in pair_text.split(';'):
            tok = tok.strip()
            if not tok:
                continue
            parts = tok.split()
            if len(parts) < 2:
                continue
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        spectra.append(pairs)
    return spectra


def build_ri_function(alkanes_df):
    a = alkanes_df.sort_values('RT').reset_index(drop=True)
    carbons = a['Carbon #'].to_numpy(dtype=float)
    rts     = a['RT'].to_numpy(dtype=float)
    lo, hi  = rts[0], rts[-1]

    def ri(rt):
        if pd.isna(rt) or rt < lo or rt > hi:
            return np.nan
        j = int(np.searchsorted(rts, rt, side='right'))
        n     = carbons[j - 1]
        rt_n  = rts[j - 1]
        rt_n1 = rts[j]
        return 100.0 * n + 100.0 * (rt - rt_n) / (rt_n1 - rt_n)

    return ri


# ---------------------------------------------------------------------------
# Step 1: merge EI spectra into the EI feature table
# ---------------------------------------------------------------------------
def merge_ei_spectra(ei_df, ei_msp_text):
    spectra = parse_msp_spectra(ei_msp_text)

    df = ei_df.copy()
    df['_fn_int'] = df['feature_number'].str.extract(r'group_(\d+)').astype(int)
    df = df.sort_values('_fn_int', kind='stable').reset_index(drop=True)

    unique_fn_ints = df['_fn_int'].drop_duplicates().sort_values().tolist()
    if len(unique_fn_ints) != len(spectra):
        raise ValueError(
            f'EI feature count mismatch: table has {len(unique_fn_ints)}, '
            f'MSP has {len(spectra)}.'
        )

    spectrum_by_feature = {
        f'group_{fn_int}': json.dumps(spec)
        for fn_int, spec in zip(unique_fn_ints, spectra)
    }
    df['EI_spectrum'] = df['feature_number'].map(spectrum_by_feature)
    df = df.drop(columns=['_fn_int'])
    return df, len(spectra), len(unique_fn_ints)


# ---------------------------------------------------------------------------
# Step 2: prepare the PCI feature table (RI + spectra + column cleanup)
# ---------------------------------------------------------------------------
AREA_COL_RE = re.compile(r'^Area:\s*(.+?)\.raw\s*\(F\d+\)\s*$')

def cleanup_area_column(col):
    m = AREA_COL_RE.match(col)
    return f'{m.group(1)}_1' if m else col


def prepare_pci_table(pci_src, pci_msp_text, alkanes_df):
    df = read_feature_matrix(pci_src, sheet_name=FEATURE_SHEET)

    df.insert(0, 'feature_number', [f'group_{i+1}' for i in range(len(df))])

    df = df.rename(columns={'Calculated RI': 'RI'})
    ri_fn = build_ri_function(alkanes_df)
    df['RI'] = df['RT [min]'].apply(ri_fn)

    spectra = parse_msp_spectra(pci_msp_text)
    if len(spectra) != len(df):
        raise ValueError(
            f'PCI feature count mismatch: table has {len(df)}, '
            f'MSP has {len(spectra)}.'
        )
    df['PCI_spectrum'] = [json.dumps(s) for s in spectra]

    # Normalize any raw "Area: <sample>.raw (F#)" headers to plain sample names.
    # (Sample renaming — e.g. the LAFires field-blank renames — is a data-prep
    # step handled outside this app; see dataprep_rename_samples.py.)
    df.columns = [cleanup_area_column(c) for c in df.columns]

    return df, len(spectra)


# ---------------------------------------------------------------------------
# Step 3: search PCI features for molecular-ion evidence
# ---------------------------------------------------------------------------
def parse_spec(s):
    if isinstance(s, float) and pd.isna(s):
        return np.zeros((0, 2))
    arr = np.array(json.loads(s), dtype=float)
    return arr if arr.size else arr.reshape(0, 2)


def peaks_within_ppm(spec, target, ppm):
    if spec.size == 0:
        return np.zeros((0, 2))
    tol = ppm * 1e-6 * target
    mask = np.abs(spec[:, 0] - target) <= tol
    return spec[mask]


def is_c13_isotope_of_larger_peak(spec, sus_mass, sus_int):
    if spec.size == 0:
        return False
    lower = sus_mass - ISOTOPE_SPACING
    mask = np.abs(spec[:, 0] - lower) <= ISOTOPE_MASS_TOL
    if not mask.any():
        return False
    return spec[mask, 1].max() >= ISOTOPE_INT_RATIO * sus_int


def find_molecular_ions(ei_df, pci_df, consider_mask=None):
    """Search PCI features for molecular-ion evidence of the EI candidates.

    consider_mask : optional boolean sequence aligned to ei_df's rows. When
    given, rows where it is False are skipped entirely — the molecular-ion
    columns (Evidence of Molecular Ion, Type of Ion, Measured Mass, Adducts
    Found, Notes, PCI_spectrum, DeltaMass [ppm]) are left at their empty
    defaults. This is how blank subtraction gates the search: only features
    with abundance > 0 after blank subtraction are searched.
    """
    ei = ei_df.copy()

    ei_row_ri = ei['RI'].to_numpy(dtype=float)
    ei_row_rt = ei['RT'].to_numpy(dtype=float)
    ei_mw     = ei['Molecular Weight'].to_numpy(dtype=float)

    if consider_mask is None:
        consider = np.ones(len(ei), dtype=bool)
    else:
        consider = np.asarray(consider_mask, dtype=bool)

    pci_ri = pci_df[pci_df['RI'].notna()].sort_values('RI').reset_index(drop=True)
    pci_ri_arr       = pci_ri['RI'].to_numpy()
    pci_ri_spec_json = pci_ri['PCI_spectrum'].tolist()
    pci_ri_specs     = [parse_spec(s) for s in pci_ri_spec_json]

    pci_rt = pci_df.sort_values('RT [min]').reset_index(drop=True)
    pci_rt_arr       = pci_rt['RT [min]'].to_numpy()
    pci_rt_spec_json = pci_rt['PCI_spectrum'].tolist()
    pci_rt_specs     = [parse_spec(s) for s in pci_rt_spec_json]

    n = len(ei)
    evidence     = np.zeros(n, dtype=bool)
    type_ion     = [None] * n
    meas_mass    = np.full(n, np.nan)
    delta_ppm    = np.full(n, np.nan)
    adducts      = [None] * n
    notes        = [None] * n
    pci_spec_out = [None] * n
    fallback_count = 0

    for i in range(n):
        if not consider[i]:
            continue

        mw = ei_mw[i]
        if not np.isfinite(mw):
            continue

        ri = ei_row_ri[i]
        if np.isfinite(ri):
            arr, specs, spec_json, target, tol = (
                pci_ri_arr, pci_ri_specs, pci_ri_spec_json, ri, RI_TOLERANCE)
        else:
            rt = ei_row_rt[i]
            if not np.isfinite(rt):
                continue
            arr, specs, spec_json, target, tol = (
                pci_rt_arr, pci_rt_specs, pci_rt_spec_json, rt, RT_TOLERANCE)
            fallback_count += 1

        lo = np.searchsorted(arr, target - tol, side='left')
        hi = np.searchsorted(arr, target + tol, side='right')
        if lo >= hi:
            continue

        theo = {
            'M+':  mw - M_ELECTRON,
            'M+H': mw + M_H - M_ELECTRON,
        }
        theo_c2h5 = mw + 2 * M_C + 5 * M_H - M_ELECTRON
        theo_c3h5 = mw + 3 * M_C + 5 * M_H - M_ELECTRON

        valid_hits = []
        isotope_rejects = []

        for j in range(lo, hi):
            spec = specs[j]
            if spec.size == 0:
                continue
            for ion_name, theo_mass in theo.items():
                hits = peaks_within_ppm(spec, theo_mass, PPM_TOLERANCE)
                for obs_mass, obs_int in hits:
                    if is_c13_isotope_of_larger_peak(spec, obs_mass, obs_int):
                        isotope_rejects.append((ion_name, float(obs_mass)))
                        continue
                    ppm = (obs_mass - theo_mass) / theo_mass * 1e6
                    ad = []
                    if len(peaks_within_ppm(spec, theo_c2h5, PPM_TOLERANCE)):
                        ad.append('M+C2H5')
                    if len(peaks_within_ppm(spec, theo_c3h5, PPM_TOLERANCE)):
                        ad.append('M+C3H5')
                    valid_hits.append(
                        (ion_name, float(obs_mass), ppm, ad, spec_json[j])
                    )

        note_parts = []

        if valid_hits:
            valid_hits.sort(key=lambda h: abs(h[2]))
            win_ion, win_mass, win_ppm, win_ad, win_spec_json = valid_hits[0]
            evidence[i]     = True
            type_ion[i]     = win_ion
            meas_mass[i]    = win_mass
            delta_ppm[i]    = win_ppm
            adducts[i]      = ', '.join(win_ad) if win_ad else ''
            pci_spec_out[i] = win_spec_json

            other_hits = [h for h in valid_hits[1:] if h[0] != win_ion]
            if other_hits:
                oh = other_hits[0]
                note_parts.append(
                    f'Possible {oh[0]} also observed at m/z {oh[1]:.4f} '
                    f'({oh[2]:+.2f} ppm)'
                )

        reported_masses = {round(h[1], 4) for h in valid_hits}
        for ion_name, rej_mass in isotope_rejects:
            if round(rej_mass, 4) in reported_masses:
                continue
            note_parts.append(
                f'Possible {ion_name} at m/z {rej_mass:.4f} rejected as '
                f'13C isotope of a larger neighboring peak'
            )

        if note_parts:
            notes[i] = '; '.join(note_parts)

    ei['Evidence of Molecular Ion'] = evidence
    ei['Type of Ion']               = type_ion
    ei['Measured Mass']             = meas_mass
    ei['DeltaMass [ppm]']           = delta_ppm
    ei['Adducts Found']             = adducts
    ei['Notes']                     = notes
    ei['PCI_spectrum']              = pci_spec_out

    stats = {
        'candidate_rows': n,
        'rows_searched': int(consider.sum()),
        'rows_with_evidence': int(evidence.sum()),
        'features_total': int(ei['feature_number'].nunique()),
        'features_with_evidence': int(ei.loc[evidence, 'feature_number'].nunique()),
        'rt_fallback_rows': fallback_count,
        'rows_with_notes': int(sum(1 for x in notes if x)),
    }
    return ei, stats


# ---------------------------------------------------------------------------
# Blank correction
# ---------------------------------------------------------------------------
# Columns in the EI output table that are NOT sample abundances. Everything
# numeric that is not in this set (and does not start with "blank_") is treated
# as a sample column that can be blank-corrected.
NON_SAMPLE_COLS = {
    'feature_number', 'RT', 'mz', 'RI', 'Name', 'CAS Num', 'Formula',
    'Total Score', 'HRF Score', 'RHRF Score', 'SI', 'RSI',
    'Elements Found[%]', 'Molecular Weight', 'Theo. Mol. Mass',
    'Observed Mol. Mass', 'DeltaMass [Da]', 'DeltaMass [ppm]',
    'M+ In Lib', 'M+ found', 'Selected', 'Library', 'Library Hit key',
    'Library ID Number', 'Library RI', 'RI Column type', 'RI Delta',
    'RI Diff[%]', 'Avg TIC',
    # columns appended by the molecular-ion pipeline
    'EI_spectrum', 'Evidence of Molecular Ion', 'Type of Ion',
    'Measured Mass', 'Adducts Found', 'Notes', 'PCI_spectrum',
}


def find_blank_columns(df):
    """Sample columns whose name starts with 'blank_' (case-insensitive)."""
    return [c for c in df.columns if str(c).strip().lower().startswith('blank_')]


def find_sample_columns(df):
    """Numeric abundance columns that are neither metadata nor blanks."""
    blanks = set(find_blank_columns(df))
    out = []
    for c in df.columns:
        if c in NON_SAMPLE_COLS or c in blanks:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def apply_blank_correction(df, mode, x_mult, log):
    """Blank-correct the sample abundances in the EI output table.

    mode : 'multiplier' (option 1), 'stddev' (option 2), or 'none' (option 3).
    x_mult : the "X=" value.

    Returns (df_out, info) where info carries messages/counts for the UI.
    df is not mutated; corrected columns are appended (see below).
    """
    info = {
        'mode_requested': mode,
        'mode_used': mode,
        'n_blanks': 0,
        'blank_cols': [],
        'sample_cols': [],
        'x_mult': x_mult,
        'messages': [],
        'applied': False,
        # Boolean Series (aligned to df) marking rows kept after blank
        # subtraction, i.e. abundance > 0 in at least one sample. When no
        # correction is applied, every row is kept.
        'passed_mask': pd.Series(True, index=df.index),
    }

    if mode == 'none':
        info['messages'].append('No blank correction applied.')
        return df, info

    blank_cols = find_blank_columns(df)
    info['n_blanks'] = len(blank_cols)
    info['blank_cols'] = blank_cols

    if not blank_cols:
        info['messages'].append(
            "Blank correction requested, but no sample columns start with "
            "'blank_'. Skipping correction — label your blank samples as "
            "'blank_1', 'blank_2', ... and re-run."
        )
        info['mode_used'] = 'none'
        return df, info

    # Option 2 needs >= 2 blanks to compute a standard deviation. If only one
    # blank is present, fall back to the option-1 (multiplier) workflow.
    if mode == 'stddev' and len(blank_cols) < 2:
        mode = 'multiplier'
        info['mode_used'] = 'multiplier'
        info['messages'].append(
            f"Only one blank ('{blank_cols[0]}') was found, so a standard "
            f"deviation cannot be calculated. Falling back to a multiplier of "
            f"the single blank (X = {x_mult})."
        )

    sample_cols = find_sample_columns(df)
    info['sample_cols'] = sample_cols

    out = df.copy()

    blanks = out[blank_cols].apply(pd.to_numeric, errors='coerce')
    avg_blank = blanks.mean(axis=1)                 # per-feature average blank
    if len(blank_cols) >= 2:
        std_blank = blanks.std(axis=1, ddof=1)      # per-feature blank std
    else:
        std_blank = pd.Series(np.nan, index=out.index)

    # Per-feature cutoff threshold
    avg_f = avg_blank.fillna(0.0)
    if mode == 'multiplier':
        threshold = x_mult * avg_f
    else:  # 'stddev'
        threshold = avg_f + x_mult * std_blank.fillna(0.0)

    log(f'  Blank correction: {len(blank_cols)} blank column(s), '
        f'{len(sample_cols)} sample column(s), mode="{mode}", X={x_mult}')

    # Correct each sample column and append as <name>_BlankSub
    corrected_frames = {}
    for col in sample_cols:
        vals = pd.to_numeric(out[col], errors='coerce')
        below = vals < threshold
        corrected = (vals - avg_f).clip(lower=0.0)   # subtract average blank
        corrected = corrected.where(~below, 0.0)     # features below cutoff -> 0
        corrected = corrected.where(vals.notna(), np.nan)  # keep blanks as blanks
        corrected_frames[f'{col}_BlankSub'] = corrected

    # Append transparency columns: the blanks used, the average, the std,
    # then all the corrected sample abundances (built at once to avoid frame
    # fragmentation from many single-column inserts).
    appended = {f'{bc}_BlankValue': out[bc] for bc in blank_cols}
    appended['Average_Blank'] = avg_blank
    appended['Blank_StdDev'] = std_blank
    appended.update(corrected_frames)
    out = pd.concat([out, pd.DataFrame(appended, index=out.index)], axis=1)

    # A row is kept if any sample has abundance > 0 after blank subtraction.
    # (NaNs — genuinely missing values — are treated as not-detected.)
    blanksub = pd.DataFrame(corrected_frames, index=out.index)
    info['passed_mask'] = (blanksub.fillna(0.0) > 0).any(axis=1)

    info['applied'] = True
    info['messages'].append(
        f'Applied blank correction to {len(sample_cols)} sample column(s) '
        f'using {len(blank_cols)} blank(s).'
    )
    return out, info


# ---------------------------------------------------------------------------
# Pipeline driver — takes 4 sources (paths or UploadedFiles) and returns
# the final DataFrame + a stats dict.
# ---------------------------------------------------------------------------
def run_pipeline(ei_matrix_src, ei_msp_src, pci_matrix_src, pci_msp_src,
                 alkanes_df, blank_mode, blank_x, log):
    log('Reading EI feature matrix...')
    ei_df = read_feature_matrix(ei_matrix_src)
    log(f'  EI table: {len(ei_df)} rows, {ei_df.shape[1]} cols')

    log('Parsing EI MSP and merging spectra...')
    ei_df, n_ei_spec, n_ei_feat = merge_ei_spectra(ei_df, read_text(ei_msp_src))
    log(f'  Attached {n_ei_spec} EI spectra to {n_ei_feat} features')

    log('Preparing PCI feature matrix (RI + spectra + column cleanup)...')
    pci_df, n_pci_spec = prepare_pci_table(
        pci_matrix_src,
        read_text(pci_msp_src),
        alkanes_df,
    )
    ri_ok = pci_df['RI'].notna().sum()
    log(f'  PCI table: {len(pci_df)} features, {n_pci_spec} spectra, '
        f'{ri_ok} within alkane bracket for RI')

    # Blank correction runs BEFORE the molecular-ion search so the search can
    # be restricted to features that survive it (abundance > 0 after blank
    # subtraction). The corrected columns are appended to the EI table here and
    # carried through to the final output.
    log('Applying blank correction...')
    ei_df, blank_info = apply_blank_correction(ei_df, blank_mode, blank_x, log)
    keep_mask = blank_info['passed_mask']
    if blank_info['applied']:
        log(f'  Features kept after blank subtraction (abundance > 0): '
            f'{int(keep_mask.sum())} / {len(keep_mask)}')

    log('Searching PCI for molecular-ion evidence...')
    out_df, stats = find_molecular_ions(ei_df, pci_df, consider_mask=keep_mask)
    log(f'  Rows searched (after blank subtraction): '
        f'{stats["rows_searched"]} / {stats["candidate_rows"]}')
    log(f'  Rows with evidence: {stats["rows_with_evidence"]} / {stats["candidate_rows"]}')
    log(f'  Features with at least one supported candidate: '
        f'{stats["features_with_evidence"]} / {stats["features_total"]}')
    log(f'  RT-fallback rows (EI RI was blank): {stats["rt_fallback_rows"]}')
    log(f'  Rows with a Notes entry: {stats["rows_with_notes"]}')

    return out_df, stats, blank_info


# ---------------------------------------------------------------------------
# Visualization: overlay EI + PCI spectra for one candidate
# ---------------------------------------------------------------------------
EI_COLOR  = '#1f77b4'   # blue
PCI_COLOR = '#d62728'   # red

def _nearest_peak(spec, target, tol):
    """Return (mass, intensity) of the peak in spec closest to target within tol, else None."""
    if spec.size == 0:
        return None
    diffs = np.abs(spec[:, 0] - target)
    j = int(np.argmin(diffs))
    if diffs[j] > tol:
        return None
    return float(spec[j, 0]), float(spec[j, 1])


def build_candidate_figure(row):
    """Build the dual-axis EI/PCI spectrum plot for one candidate row."""
    ei_spec  = parse_spec(row.get('EI_spectrum'))
    pci_spec = parse_spec(row.get('PCI_spectrum'))
    mw       = float(row['Molecular Weight'])

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    if ei_spec.size:
        fig.add_trace(
            go.Bar(
                x=ei_spec[:, 0], y=ei_spec[:, 1],
                name='EI', marker_color=EI_COLOR,
                width=0.6, opacity=0.85,
                hovertemplate='EI  m/z %{x:.4f}<br>intensity %{y:.0f}<extra></extra>',
            ),
            secondary_y=False,
        )
    if pci_spec.size:
        fig.add_trace(
            go.Bar(
                x=pci_spec[:, 0], y=pci_spec[:, 1],
                name='PCI', marker_color=PCI_COLOR,
                width=0.6, opacity=0.85,
                hovertemplate='PCI  m/z %{x:.4f}<br>intensity %{y:.0f}<extra></extra>',
            ),
            secondary_y=True,
        )

    # Molecular-ion annotation (on PCI axis = y2)
    ion_type   = row.get('Type of Ion')
    meas_mass  = row.get('Measured Mass')
    if ion_type and pd.notna(meas_mass) and pci_spec.size:
        hit = _nearest_peak(pci_spec, float(meas_mass),
                            tol=PPM_TOLERANCE * 1e-6 * float(meas_mass) * 2)
        if hit is not None:
            m, inten = hit
            fig.add_annotation(
                x=m, y=inten, yref='y2',
                text=f'<b>{ion_type}</b><br>m/z {m:.4f}',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=PCI_COLOR, ax=0, ay=-40,
                font=dict(color=PCI_COLOR, size=12),
                bgcolor='rgba(255,255,255,0.85)', bordercolor=PCI_COLOR,
            )

    # Adduct annotations (on PCI axis)
    adducts_str = row.get('Adducts Found') or ''
    adduct_theo = {
        'M+C2H5': mw + 2 * M_C + 5 * M_H - M_ELECTRON,
        'M+C3H5': mw + 3 * M_C + 5 * M_H - M_ELECTRON,
    }
    for ad in [a.strip() for a in adducts_str.split(',') if a.strip()]:
        theo = adduct_theo.get(ad)
        if theo is None or pci_spec.size == 0:
            continue
        hit = _nearest_peak(pci_spec, theo, tol=PPM_TOLERANCE * 1e-6 * theo * 2)
        if hit is None:
            continue
        m, inten = hit
        fig.add_annotation(
            x=m, y=inten, yref='y2',
            text=f'<b>{ad}</b><br>m/z {m:.4f}',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=PCI_COLOR, ax=0, ay=-60,
            font=dict(color=PCI_COLOR, size=11),
            bgcolor='rgba(255,255,255,0.85)', bordercolor=PCI_COLOR,
        )

    fig.update_layout(
        barmode='overlay', bargap=0,
        height=520, margin=dict(l=60, r=60, t=40, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified',
    )
    fig.update_xaxes(title_text='m/z')
    fig.update_yaxes(title_text='EI intensity',  color=EI_COLOR,  secondary_y=False)
    fig.update_yaxes(title_text='PCI intensity', color=PCI_COLOR, secondary_y=True)
    return fig


def render_visualization(df):
    """Interactive candidate viewer for rows with molecular-ion evidence.

    Only features with abundance > 0 after blank subtraction are shown: the
    molecular-ion search is gated on that condition (see run_pipeline), so
    'Evidence of Molecular Ion' is only ever True for surviving features.
    """
    mask = df['Evidence of Molecular Ion'].fillna(False).astype(bool)
    hits = df[mask].reset_index(drop=True)

    st.subheader('Candidate spectra viewer')
    if hits.empty:
        st.info('No candidates with molecular-ion evidence to display.')
        return

    # Persistent index into the hits list
    if 'viz_idx' not in st.session_state:
        st.session_state.viz_idx = 0
    st.session_state.viz_idx = int(np.clip(st.session_state.viz_idx, 0, len(hits) - 1))

    unique_features = hits['feature_number'].drop_duplicates().tolist()

    # Navigation row: Prev | Jump-to-feature | Next
    c_prev, c_jump, c_jump_btn, c_next = st.columns([1, 2, 1, 1])
    with c_prev:
        if st.button('◀ Prev', use_container_width=True,
                     disabled=st.session_state.viz_idx == 0):
            st.session_state.viz_idx -= 1
            st.rerun()
    with c_jump:
        jump_target = st.text_input(
            'Jump to feature_number',
            value='',
            placeholder=f'e.g. {unique_features[0]}',
            label_visibility='collapsed',
            key='viz_jump_text',
        )
    with c_jump_btn:
        if st.button('Go', use_container_width=True):
            target = jump_target.strip()
            # Accept either "group_5" or bare "5"
            if target.isdigit():
                target = f'group_{target}'
            matches = hits.index[hits['feature_number'] == target].tolist()
            if matches:
                st.session_state.viz_idx = int(matches[0])
                st.rerun()
            else:
                st.warning(f'No candidate with evidence for feature "{target}".')
    with c_next:
        if st.button('Next ▶', use_container_width=True,
                     disabled=st.session_state.viz_idx >= len(hits) - 1):
            st.session_state.viz_idx += 1
            st.rerun()

    row = hits.iloc[st.session_state.viz_idx]

    # Feature-position context: which candidate within this feature, and
    # which feature within the overall feature list.
    feat = row['feature_number']
    same_feat_idxs = hits.index[hits['feature_number'] == feat].tolist()
    cand_pos_in_feat = same_feat_idxs.index(st.session_state.viz_idx) + 1
    feat_pos = unique_features.index(feat) + 1

    st.markdown(
        f'**Feature:** `{feat}`  &nbsp;·&nbsp;  '
        f'**Candidate:** {row.get("Name", "—")}  &nbsp;·&nbsp;  '
        f'**MW:** {row["Molecular Weight"]}'
    )
    st.caption(
        f'Candidate {cand_pos_in_feat} of {len(same_feat_idxs)} in this feature '
        f'· Feature {feat_pos} of {len(unique_features)} with evidence '
        f'· Hit {st.session_state.viz_idx + 1} of {len(hits)} overall'
    )

    fig = build_candidate_figure(row)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title='EI/PCI Molecular Ion Finder', layout='centered')
st.title('EI/PCI Molecular Ion Finder')
st.write(
    'Upload the four inputs (or check *Run demo files* to use the bundled '
    'SEISMIC demo dataset), then click **Find Molecular Ions in PCI data**.'
)

if 'result_df' not in st.session_state:
    st.session_state.result_df   = None
    st.session_state.result_name = None
    st.session_state.result_stats = None
    st.session_state.result_blank = None

demo_mode = st.checkbox('Run demo files', value=False,
                        help='Use the SEISMIC demo files (260914) shipped alongside this app.')

ei_matrix_src = st.file_uploader('EI feature matrix (CSV or XLSX)',
                                 type=['csv', 'xlsx'], disabled=demo_mode)
ei_msp_src    = st.file_uploader('EI MSP file',
                                 type=['msp', 'txt'], disabled=demo_mode)
pci_matrix_src = st.file_uploader('PCI feature matrix (CSV or XLSX)',
                                 type=['csv', 'xlsx'], disabled=demo_mode)
pci_msp_src   = st.file_uploader('PCI MSP file',
                                 type=['msp', 'txt'], disabled=demo_mode)

# ---- Blank correction options ------------------------------------------------
BLANK_OPT_MULTIPLIER = 'Use a multiplier of my average blank (use if you only have one blank)'
BLANK_OPT_STDDEV     = 'Use average plus X standard deviations (use only if you have multiple blanks)'
BLANK_OPT_NONE       = 'Do not do any blank correction'

blank_choice = st.radio(
    "Blank Correction (must label blanks as 'blank_#'):",
    options=[BLANK_OPT_MULTIPLIER, BLANK_OPT_STDDEV, BLANK_OPT_NONE],
    index=2,  # default: no blank correction
)

blank_x = 3.0
if blank_choice in (BLANK_OPT_MULTIPLIER, BLANK_OPT_STDDEV):
    c_lbl, c_box = st.columns([1, 6])
    with c_lbl:
        st.markdown('X=')
    with c_box:
        x_raw = st.text_input('X', value='3', label_visibility='collapsed')
    try:
        blank_x = float(x_raw)
    except (TypeError, ValueError):
        blank_x = 3.0
        st.warning(f'Could not read X="{x_raw}"; using X=3.')
    st.caption('All features will be corrected by the average blank level.')

if blank_choice == BLANK_OPT_MULTIPLIER:
    blank_mode = 'multiplier'
elif blank_choice == BLANK_OPT_STDDEV:
    blank_mode = 'stddev'
else:
    blank_mode = 'none'

run_clicked = st.button('Find Molecular Ions in PCI data', type='primary')

if run_clicked:
    # Resolve sources
    if demo_mode:
        for p, label in [
            (DEMO_EI_MATRIX,  'demo EI matrix'),
            (DEMO_EI_MSP,     'demo EI MSP'),
            (DEMO_PCI_MATRIX, 'demo PCI matrix'),
            (DEMO_PCI_MSP,    'demo PCI MSP'),
        ]:
            if not os.path.exists(p):
                st.error(f'{label} not found: {p}')
                st.stop()
        ei_matrix, ei_msp, pci_matrix, pci_msp = (
            DEMO_EI_MATRIX, DEMO_EI_MSP, DEMO_PCI_MATRIX, DEMO_PCI_MSP)
        out_basename = 'SEISMIC_DEMO_EI_Matrix_260914_MolIon.csv'
    else:
        missing = [name for name, src in [
            ('EI feature matrix',  ei_matrix_src),
            ('EI MSP file',        ei_msp_src),
            ('PCI feature matrix', pci_matrix_src),
            ('PCI MSP file',       pci_msp_src),
        ] if src is None]
        if missing:
            st.error('Please upload: ' + ', '.join(missing))
            st.stop()
        ei_matrix, ei_msp, pci_matrix, pci_msp = (
            ei_matrix_src, ei_msp_src, pci_matrix_src, pci_msp_src)
        stem = os.path.splitext(ei_matrix_src.name)[0]
        out_basename = f'{stem}_spectra_MolIon.csv'

    # Load bundled auxiliary file (alkane retention times for RI)
    if not os.path.exists(ALKANES_CSV):
        st.error(f'Missing bundled file: {ALKANES_CSV}')
        st.stop()
    alkanes_df = pd.read_csv(ALKANES_CSV)

    log_area = st.empty()
    log_lines = []
    def log(msg):
        log_lines.append(msg)
        log_area.code('\n'.join(log_lines))

    try:
        with st.spinner('Running pipeline...'):
            out_df, stats, blank_info = run_pipeline(
                ei_matrix, ei_msp, pci_matrix, pci_msp,
                alkanes_df, blank_mode, blank_x, log,
            )
    except Exception as e:
        st.exception(e)
        st.stop()

    # Stash for downstream reruns (Prev / Next / Jump don't re-run the pipeline)
    st.session_state.result_df    = out_df
    st.session_state.result_name  = out_basename
    st.session_state.result_stats = stats
    st.session_state.result_blank = blank_info
    st.session_state.viz_idx      = 0

# Results panel — renders on every rerun so navigation is instant
if st.session_state.result_df is not None:
    out_df       = st.session_state.result_df
    out_basename = st.session_state.result_name
    stats        = st.session_state.result_stats

    st.success(
        f'Done. {stats["rows_with_evidence"]} / {stats["candidate_rows"]} '
        f'candidate rows have molecular-ion evidence '
        f'({stats["features_with_evidence"]} / {stats["features_total"]} features).'
    )

    blank_info = st.session_state.get('result_blank')
    if blank_info is not None:
        for msg in blank_info.get('messages', []):
            if blank_info.get('mode_requested') != 'none' and not blank_info.get('applied'):
                st.warning(msg)
            elif (blank_info.get('mode_requested') == 'stddev'
                  and blank_info.get('mode_used') == 'multiplier'):
                # single-blank fallback note
                st.warning(msg) if 'Only one blank' in msg else st.info(msg)
            else:
                st.info(msg)

    csv_bytes = out_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f'Download {out_basename}',
        data=csv_bytes,
        file_name=out_basename,
        mime='text/csv',
    )

    with st.expander('Preview (first 20 rows)'):
        st.dataframe(out_df.head(20))

    st.divider()
    render_visualization(out_df)

# data/sofa.py
import numpy as np
import json
from data.constants import MEASUREMENT_IDX, TREATMENT_IDX, SOFA_MEAS_LOCAL, SOFA_TREAT_LOCAL, SOFA_MEAS_GLOBAL, SOFA_TREAT_GLOBAL
from config import CONTEXT_STEPS

# Load scaling stats once at import time
with open('/home/bbe9928/thesis_work/hirid_jepa/data/scaling_stats.json') as f:
    _stats = json.load(f)
_mean = np.array(_stats['mean'])
_std  = np.array(_stats['std'])

def unscale(global_idx, scaled_value):
    return scaled_value * _std[global_idx] + _mean[global_idx]

def rescale_percent(value):
    return value/100.0 if value is not None else None
def rescale_creatinine(value):
    """Rescale creatinine to mg/dL if it is in umol/L"""
    return value / 88.4 if value is not None else None
def rescale_bilirubin(value):
    """Rescale bilirubin to mg/dL if it is in umol/L"""
    return value / 17.1 if value is not None else None


def respiratory(pao2, fio2):
    """Calculate the respiratory component of the SOFA score."""
    if fio2 is None:
        "fio2 is percentage of oxygen, if not measured I use 21% as default because the patient is breathing room air"
        fio2 = 21
    if pao2 is None:
        return None
    ratio = pao2 / fio2
    if ratio >= 400:
        return 0
    elif ratio >= 300:
        return 1
    elif ratio >= 200:
        return 2
    elif ratio >= 100:
        return 3
    else:
        return 4
    
def cardiovascular(map_val, dobutamine, epinephrine, norepinephrine):
    if map_val is None:
        return None
    # Treat as binary presence since flow rates aren't comparable across concentrations
    has_norepi = norepinephrine is not None and norepinephrine > 0
    has_epi    = epinephrine    is not None and epinephrine    > 0
    has_dobu   = dobutamine     is not None and dobutamine     > 0

    if has_norepi or has_epi: return 3  # can't distinguish 3 vs 4 without concentration
    if has_dobu:              return 2
    if map_val < 70:          return 1
    return 0
    
def coagulation(platelets):
    """Calculate the coagulation component of the SOFA score."""
    if platelets is None:
        return None
    elif platelets >= 150:
        return 0
    elif platelets >= 100:
        return 1
    elif platelets >= 50:
        return 2
    elif platelets >= 20:
        return 3
    else:
        return 4
    
def liver(bilirubin):
    """Calculate the liver component of the SOFA score."""
    if bilirubin is None:
        return None
    elif bilirubin < 1.2:
        return 0
    elif bilirubin < 2.0:
        return 1
    elif bilirubin < 6.0:
        return 2
    elif bilirubin < 12.0:
        return 3
    else:
        return 4
    
def cns(gcs):
    """Calculate the CNS component of the SOFA score."""
    if gcs is None:
        return None
    elif gcs == 15:
        return 0
    elif gcs >= 13:
        return 1
    elif gcs >= 10:
        return 2
    elif gcs >= 6:
        return 3
    else:
        return 4

def renal_creatinine(creatinine):
    """Calculate the renal component of the SOFA score based on creatinine."""
    if creatinine is None:
        return None
    elif creatinine < 1.2:
        return 0
    elif creatinine < 2.0:
        return 1
    elif creatinine < 3.5:
        return 2
    elif creatinine < 5.0:
        return 3
    else:
        return 4

def renal_urine_output(urine_output):
    """Calculate the renal component of the SOFA score based on urine output."""
    """NOTE: This is usually based on 24h, we have to approximate this by extrapolating from the given time window"""
    if urine_output is None:
        return None
    elif urine_output >= 500:
        return 0
    elif urine_output >= 200:
        return 1
    elif urine_output >= 100:
        return 2
    elif urine_output < 100:
        return 3
    

def get_val(arr, mask, local_idx, global_idx, mode):
    obs = arr[:, local_idx][mask[:, local_idx] == 1]
    if len(obs) == 0:
        return None
    raw = {'min': obs.min, 'max': obs.max,
           'sum': obs.sum, 'mean': obs.mean}[mode]()
    return unscale(global_idx, raw)

def get_measurement_value(measurements, meas_mask, name, mode):
    return get_val(measurements, meas_mask,
                   SOFA_MEAS_LOCAL[name], SOFA_MEAS_GLOBAL[name], mode)

def get_treatment_value(treatments, treat_mask, name, mode):
    return get_val(treatments, treat_mask,
                   SOFA_TREAT_LOCAL[name], SOFA_TREAT_GLOBAL[name], mode)

def compute_sofa(measurements, treatments, meas_mask, treat_mask,
                  verbose=False):

    def log(name, scaled, unscaled):
        if verbose and scaled is not None:
            print(f"  {name:20s} scaled={scaled:8.4f}  unscaled={unscaled:8.3f}")
        elif verbose:
            print(f"  {name:20s} not observed")

    pao2    = get_measurement_value(measurements, meas_mask, 'PaO2',       'min')
    fio2    = get_treatment_value(treatments,   treat_mask, 'FiO2',      'mean')
    fio2 = rescale_percent(fio2)
    map_val = get_measurement_value(measurements, meas_mask, 'MAP',        'min')
    plts    = get_measurement_value(measurements, meas_mask, 'platelets',  'min')
    bili    = get_measurement_value(measurements, meas_mask, 'bilirubin',  'max')
    bili = rescale_bilirubin(bili)
    creat   = get_measurement_value(measurements, meas_mask, 'creatinine', 'max')
    creat = rescale_creatinine(creat)
    norepi  = get_treatment_value(treatments,   treat_mask, 'norepinephrine', 'max')
    epi     = get_treatment_value(treatments,   treat_mask, 'epinephrine',    'max')
    dobu    = get_treatment_value(treatments,   treat_mask, 'dobutamine',     'max')
    # gru predicts values for every timestep, this is a limitation
    out     = get_measurement_value(measurements, meas_mask, 'OUTurine_h', 'mean')
    if out is not None:
        out = min(out, 400.0) * 24.0

    gcs_parts = [get_measurement_value(measurements, meas_mask, k, 'min')
                 for k in ['GCS_E', 'GCS_V', 'GCS_M']]
    gcs = sum(v for v in gcs_parts if v is not None) \
          if any(v is not None for v in gcs_parts) else None

    if verbose:
        # Raw scaled values for logging
        def raw(arr, mask, local_idx):
            obs = arr[:, local_idx][mask[:, local_idx] == 1]
            return obs.min() if len(obs) > 0 else None

        print("\n  Raw values:")
        log('PaO2',          raw(measurements, meas_mask, SOFA_MEAS_LOCAL['PaO2']),       pao2    or 0)
        log('FiO2',          raw(treatments,   treat_mask, SOFA_TREAT_LOCAL['FiO2']),     fio2    or 0)
        log('MAP',           raw(measurements, meas_mask, SOFA_MEAS_LOCAL['MAP']),        map_val or 0)
        log('platelets',     raw(measurements, meas_mask, SOFA_MEAS_LOCAL['platelets']),  plts    or 0)
        log('bilirubin',     raw(measurements, meas_mask, SOFA_MEAS_LOCAL['bilirubin']),  bili    or 0)
        log('creatinine',    raw(measurements, meas_mask, SOFA_MEAS_LOCAL['creatinine']), creat   or 0)
        log('norepinephrine',raw(treatments,   treat_mask, SOFA_TREAT_LOCAL['norepinephrine']), norepi or 0)
        log('epinephrine',   raw(treatments,   treat_mask, SOFA_TREAT_LOCAL['epinephrine']),    epi    or 0)
        log('dobutamine',    raw(treatments,   treat_mask, SOFA_TREAT_LOCAL['dobutamine']),     dobu   or 0)
        log('OUT',           raw(measurements, meas_mask, SOFA_MEAS_LOCAL['OUTurine_h']),        out     or 0)
        log('GCS',           None if gcs is None else 1.0, gcs or 0)

    resp_score  = respiratory(pao2, fio2)
    card_score  = cardiovascular(map_val, dobu, epi, norepi)
    coag_score  = coagulation(plts)
    liv_score   = liver(bili)
    renal_score = max((s for s in [renal_creatinine(creat), renal_urine_output(out)]
                       if s is not None), default=None)
    cns_score   = cns(gcs)

    if verbose:
        print(f"\n  Component scores:")
        print(f"  {'respiratory':20s} {resp_score}")
        print(f"  {'cardiovascular':20s} {card_score}")
        print(f"  {'coagulation':20s} {coag_score}")
        print(f"  {'liver':20s} {liv_score}")
        print(f"  {'renal':20s} {renal_score}")
        print(f"  {'cns':20s} {cns_score}")

    scores = {
        'respiratory':    resp_score,
        'cardiovascular': card_score,
        'coagulation':    coag_score,
        'liver':          liv_score,
        'renal':          renal_score,
        'cns':            cns_score,
    }
    filled = {k: (v if v is not None else 0) for k, v in scores.items()}
    filled['total']        = sum(filled[k] for k in scores)
    filled['n_components'] = sum(1 for v in scores.values() if v is not None)
    filled['n_imputed']    = sum(1 for v in scores.values() if v is None)
    return filled
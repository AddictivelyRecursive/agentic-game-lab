"""POST-HOC / EXPLORATORY: score supplied human labels, never create labels.

Balanced flagged/unflagged sampling is case-control sampling. Recall uses inverse
sampling weights. Wilson intervals for weighted rates use Kish effective n and
are approximate; they do not account for episode clustering or labeling error.
UNCLEAR is excluded and reported; nonresponse can bias every estimate.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import table

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('labels',type=Path); ap.add_argument('--key',type=Path,default=Path(__file__).parent/'private_validation_key.csv'); a=ap.parse_args()
    labels=pd.read_csv(a.labels,keep_default_na=False); key=pd.read_csv(a.key)
    assert labels.decision_id.is_unique
    labels.human_label=labels.human_label.str.strip().str.upper()
    if not labels.human_label.isin(['','INCONSISTENT','CONSISTENT','UNCLEAR']).all(): raise ValueError('Unknown human label')
    data=key.merge(labels[['decision_id','human_label']],on='decision_id',how='left',validate='one_to_one'); data.human_label=data.human_label.fillna('')
    print('Labels:',data.human_label.value_counts().to_dict())
    if not data.human_label.isin(['INCONSISTENT','CONSISTENT']).any():
        print('Human labels pending; no precision, recall or miss rate computed.'); return
    labeled=data[data.human_label.isin(['INCONSISTENT','CONSISTENT'])].copy(); labeled['inconsistent']=labeled.human_label.eq('INCONSISTENT'); rows=[]
    for metric,subset,success in [('precision',labeled[labeled.detector_flag], 'inconsistent'),('unflagged_miss_rate',labeled[~labeled.detector_flag],'inconsistent'),('recall',labeled[labeled.inconsistent],'detector_flag')]:
        n=len(subset); k=int(subset[success].sum()); low,high=proportion_confint(k,n,method='wilson') if n else (np.nan,np.nan)
        weights=subset.sampling_weight.to_numpy(); total=weights.sum(); weighted=(weights*subset[success].to_numpy()).sum()/total if total else np.nan
        neff=total**2/np.sum(weights**2) if total else 0
        wl,wh=proportion_confint(weighted*neff,neff,method='wilson') if neff else (np.nan,np.nan)
        rows.append(dict(metric=metric,labeled_denominator=n,unweighted_sample_rate=k/n if n else np.nan,unweighted_wilson_low=low,unweighted_wilson_high=high,population_weighted_rate=weighted,approx_weighted_wilson_low=wl,approx_weighted_wilson_high=wh,effective_n=neff,note='Use weighted recall; naive case-control sample recall is invalid. Weighted Wilson is approximate, not episode-cluster inference.'))
    out=pd.DataFrame(rows); table(out,'04_validation_scores'); print(out.to_string(index=False))
    print('UNCLEAR:',int(data.human_label.eq('UNCLEAR').sum()),'Unlabeled:',int(data.human_label.eq('').sum()),'Nonresponse/UNCLEAR excluded; estimates may be biased.')

if __name__=='__main__': main()

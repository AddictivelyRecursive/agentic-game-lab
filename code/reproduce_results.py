"""Verify the released primary tables from the released episode summaries."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'analysis'))
from common import contrast, heterogeneity, COUNTS, MODELS
from fit_glmm_and_curve import fit_primary, coefficient, curve_comparison

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'results/MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Released result digest mismatch: {name}')
    folder = ROOT / 'results/primary'
    ep = pd.read_csv(folder / '00_episode_outcomes.csv')
    assert len(ep) == 376 and ep.episode_id.is_unique
    assert ep.status.eq('completed').all()
    counts = ep.groupby(['model', 'mode', 'N']).size()
    assert len(counts) == 48
    for (model, mode, n), count in counts.items():
        assert mode in {'structured', 'free_form'} and n in {2, 3, 4, 5}
        assert count == COUNTS[model]
    ep['c'] = ep.cooperation_percent / 100
    cells = ep.groupby(['model', 'mode', 'N']).cooperation_percent.agg(
        episodes='size', cooperation_percent='mean', variance_pp_squared='var').reset_index()
    published = pd.read_csv(folder / '00_cells.csv')
    pd.testing.assert_frame_equal(cells, published, check_exact=False, rtol=1e-10, atol=1e-10)
    effects = contrast(ep)
    published = pd.read_csv(folder / '00_effects.csv')
    pd.testing.assert_frame_equal(effects, published, check_exact=False, rtol=1e-10, atol=1e-10)
    hetero = heterogeneity(effects)
    expected_hetero = pd.read_csv(folder / '00_heterogeneity.csv').iloc[0]
    for key, value in hetero.items():
        np.testing.assert_allclose(value, expected_hetero[key], rtol=1e-10, atol=1e-10)
    data = ep.rename(columns={'mode': 'reasoning_mode', 'c': 'cooperation'}).copy()
    data['free_mode'] = data.reasoning_mode.eq('free_form').astype(int)
    expected = json.loads((folder / 'confirmatory_results.json').read_text())
    curves = curve_comparison(data)
    for model in ('flat', 'dilution'):
        for metric in ('aic', 'bic'):
            np.testing.assert_allclose(curves[model][metric], expected['H2'][model][metric], atol=1e-6, rtol=1e-7)
    fit, attempts = fit_primary(data)
    assert fit is not None, attempts
    h1 = coefficient(fit, 'N')
    for key, value in h1.items():
        np.testing.assert_allclose(value, expected['H1'][key], atol=1e-5, rtol=1e-4)
    print('PASS: 376 episodes, 48 cells, cell means/variances, six interactions,')
    print('Welch intervals, Holm correction, heterogeneity, pooled H1, and H2 AIC/BIC.')
    print('This verifies derived summaries; run code/run_saved_log_analysis.py to reconstruct them from released logs.')


if __name__ == '__main__':
    main()

"""H1/H2 unchanged; named-model H4 and heterogeneity primary; pooled H4 descriptive."""
import argparse
import math
from pathlib import Path
import warnings

from experiments.study_common import (read_study, require_complete, new_output, write_json,
                                      REASONING_MODELS, TRAINING_NOTE, file_digest)


def fit_primary(data):
    import numpy as np
    import statsmodels.formula.api as smf
    attempts = []
    # free_mode=0 is structured: N is exactly the structured slope (H1).
    for random_formula in ("1 + N", "1"):
        for method in ("lbfgs", "powell"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = smf.mixedlm("cooperation ~ N * free_mode", data,
                                      groups=data["model"], re_formula=random_formula).fit(
                                          reml=False, method=method, maxiter=2000, disp=False)
                    valid = (fit.converged and np.isfinite(fit.llf)
                             and np.isfinite(fit.fe_params).all() and np.isfinite(fit.bse_fe).all()
                             and (fit.bse_fe > 0).all())
                    attempts.append({"random_formula": random_formula, "optimizer": method,
                                     "converged": bool(fit.converged), "usable": bool(valid),
                                     "warnings": [str(w.message) for w in caught]})
                    if valid:
                        return fit, attempts
                except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                    attempts.append({"random_formula": random_formula, "optimizer": method,
                                     "usable": False, "error": str(exc),
                                     "warnings": [str(w.message) for w in caught]})
    return None, attempts


def coefficient(fit, term):
    low, high = fit.conf_int().loc[term]
    return {"coefficient": float(fit.fe_params[term]), "ci_low": float(low), "ci_high": float(high),
            "p_two_sided": float(fit.pvalues[term])}


def cochran_heterogeneity(effects):
    """Q and I^2 using each main-data interaction's sampling variance."""
    from scipy.stats import chi2
    if len(effects) != 6 or any(e['status'] != 'estimated' or e['endpoint_standard_error'] <= 0 for e in effects):
        return {'status': 'not_estimable', 'reason': 'Requires six positive, estimable interaction standard errors'}
    weights = [1/e['endpoint_standard_error']**2 for e in effects]
    mean = sum(w*e['endpoint_interaction'] for w,e in zip(weights,effects))/sum(weights)
    q = sum(w*(e['endpoint_interaction']-mean)**2 for w,e in zip(weights,effects))
    return {'status': 'estimated', 'Q': float(q), 'df': 5, 'p_value': float(chi2.sf(q,5)),
            'I_squared_percent': float(100*max(0,(q-5)/q)) if q > 0 else 0.,
            'fixed_weighted_mean_descriptive': float(mean),
            'caveat': 'Chi-square approximation; six models and estimated standard errors, not proof of a power ceiling'}


def fit_per_model_h4(data, models):
    import numpy as np
    import statsmodels.formula.api as smf
    from scipy.stats import t
    from statsmodels.stats.multitest import multipletests
    effects=[]
    for model in models:
        subset=data[data.model==model]
        fit=smf.ols('cooperation ~ N * free_mode',subset).fit()
        cells=subset.groupby(['N','free_mode']).cooperation.agg(['mean','var','count'])
        contributions=[]; denominator=0.
        for (n,free),cell in cells.iterrows():
            weight=(1 if free else -1)*.6*(n-3.5)
            component=weight**2*cell['var']/cell['count']
            contributions.append(component)
            if cell['count']>1:denominator+=component**2/(cell['count']-1)
        variance=float(sum(contributions))
        effect=3*float(fit.params['N:free_mode'])
        row={'model':model,'status':'not_estimable','slope_interaction_per_player':effect/3,
             'endpoint_interaction':effect,'endpoint_standard_error':None,
             'q_m':float(variance*cells['count'].iloc[0]),
             'se_method':'eight-cell episode variance contrast with Welch-Satterthwaite df',
             'episodes':len(subset),'reason':'Zero/nonfinite variance or insufficient replication'}
        if len(cells)==8 and (cells['count']>=2).all() and np.isfinite(variance) and variance>0 and denominator>0:
            se=math.sqrt(variance); df=variance**2/denominator
            low,high=effect-t.ppf(.975,df)*se,effect+t.ppf(.975,df)*se
            row.update(status='estimated',endpoint_standard_error=se,welch_df=float(df),
                       ci_low=float(low),ci_high=float(high),p_two_sided=float(2*t.sf(abs(effect/se),df)))
            row.pop('reason')
        effects.append(row)
    valid=[e for e in effects if e['status']=='estimated']
    if valid:
        adjusted=multipletests([e['p_two_sided'] for e in valid],method='holm')[1]
        for e,p in zip(valid,adjusted):e['p_holm_supplemental']=float(p)
    return effects,cochran_heterogeneity(effects)


def curve_comparison(data):
    import numpy as np
    import statsmodels.formula.api as smf
    structured = data[data["free_mode"] == 0].copy()
    structured["dilution"] = 1 / (structured["N"] - 1)
    result = {}
    for name, formula in (("flat", "cooperation ~ 1"), ("dilution", "cooperation ~ dilution")):
        fit = smf.ols(formula, structured).fit()
        # Gaussian ML AIC/BIC count residual variance as a fitted parameter too.
        k = int(fit.df_model + 1) + 1
        if not np.isfinite(fit.llf) or fit.ssr <= 1e-20:
            raise ValueError("H2 Gaussian likelihood is degenerate (zero residual variance)")
        result[name] = {"aic": float(-2 * fit.llf + 2 * k),
                        "bic": float(-2 * fit.llf + math.log(len(structured)) * k),
                        "coefficients": {key: float(value) for key, value in fit.params.items()},
                        "episodes": len(structured), "parameters_including_residual_variance": k}
    result["delta_aic_flat_minus_dilution"] = result["flat"]["aic"] - result["dilution"]["aic"]
    result["delta_bic_flat_minus_dilution"] = result["flat"]["bic"] - result["dilution"]["bic"]
    return result


def analyze(input_path, output):
    import numpy as np
    import pandas as pd
    import statsmodels
    import statsmodels.formula.api as smf

    manifest, rows = read_study(input_path, "main")
    require_complete(manifest, rows)
    data = pd.DataFrame([{k: r[k] for k in ("model", "N", "reasoning_mode", "seed", "cooperation")} for r in rows])
    data["free_mode"] = (data.reasoning_mode == "free_form").astype(int)
    output = new_output(output)
    data.to_csv(output / "episode_analysis_data.csv", index=False)
    if data.cooperation.var() <= 1e-20:
        fit, attempts = None, [{"usable": False, "error": "Constant cooperation; mixed-model uncertainty cannot be estimated"}]
    else:
        fit, attempts = fit_primary(data)
    results = {"source_sha256": file_digest(input_path), "statsmodels_version": statsmodels.__version__,
               "family": "Gaussian", "link": "identity", "unit": "episode mean",
               "estimation": "maximum likelihood", "primary_attempts": attempts}
    per_model, heterogeneity = fit_per_model_h4(data, manifest['plan']['models'])
    results['H4'] = {'role': 'PRIMARY: six named-model fixed-effects interactions',
                     'per_model': per_model, 'heterogeneity': heterogeneity,
                     'multiplicity': 'alpha .05 marginal per model; supplemental Holm p-values; Q is an additional primary test'}
    results['analysis_amendment'] = '2026-10-07: configs/design.json'
    lines = ["# Confirmatory results: H1, H2, named-model H4", "One independent episode mean per observation; no round-level pseudoreplication.",
             "H1 and SECONDARY/DESCRIPTIVE pooled H4 use the unchanged Gaussian identity-link mixed model: cooperation ~ N * free_mode, with model identity as random intercept "
             "and a random N slope if a usable converged fit is obtained. Structured is the reference (free_mode=0).",
             "Cooperation is graded, not a count of binary successes. This is a Gaussian mixed model, not a binomial GLMM.",
             "Wald 95% CIs and asymptotic two-sided p-values; alpha=0.05, unadjusted per hypothesis. "
             "Only six model groups: interpret asymptotic uncertainty cautiously. Residuals/predictions are exported for diagnostics."]
    lines += ['## Amendment: H4 primary estimand',
              'H4 is PRIMARY separately within each named model: endpoint interaction = 3 times the N-by-mode coefficient. '
              'OLS point estimates with model-specific eight-cell episode variance and Welch-Satterthwaite t uncertainty. '
              'No pooling for these six tests; raw alpha=.05 per model, with supplemental Holm p-values. '
              'Power targets marginal tests, not simultaneous detection or family-wise power.',
              'H1 and H2 are UNCHANGED. H1 remains the pooled structured-reference N slope in the existing mixed model; '
              'H2 uses structured episodes only. H1 has not been replaced by a structured-only refit.',
              '| Model | Endpoint interaction | SE | 95% CI | p | Supplemental Holm p |',
              '|---|---:|---:|---|---:|---:|']
    for e in per_model:
        if e['status']=='estimated':
            lines.append(f"| {e['model']} | {e['endpoint_interaction']:.6g} | {e['endpoint_standard_error']:.6g} | "
                         f"[{e['ci_low']:.6g}, {e['ci_high']:.6g}] | {e['p_two_sided']:.6g} | {e['p_holm_supplemental']:.6g} |")
        else:lines.append(f"| {e['model']} | {e['endpoint_interaction']:.6g} | Not estimable | — | — | — |")
    lines += ['## PRIMARY H4 heterogeneity: Cochran Q and I-squared']
    if heterogeneity['status']=='estimated':
        lines.append(f"Q={heterogeneity['Q']:.6g}, df=5, p={heterogeneity['p_value']:.6g}; I^2={heterogeneity['I_squared_percent']:.6g}%. "
                     'These quantify heterogeneity; neither significant Q nor I^2 alone proves inadequate power. '
                     'Six estimated standard errors make the chi-square approximation uncertain.')
    else:lines.append(heterogeneity['reason'])
    if fit is None:
        results["primary_status"] = "failed"
        lines += ["## H1", "Not estimable: no usable converged mixed model.",
                  "## SECONDARY/DESCRIPTIVE pooled H4", "Not estimable: no usable converged mixed model. This does not invalidate estimable per-model primary results."]
    else:
        results.update(primary_status="converged", selected_random_formula=attempts[-1]["random_formula"],
                       H1=coefficient(fit, "N"), H4_secondary_pooled=coefficient(fit, "N:free_mode"))
        results['H4_secondary_pooled']['endpoint_interaction']=3*results['H4_secondary_pooled']['coefficient']
        for hypothesis, key in (("H1", "H1"), ("SECONDARY/DESCRIPTIVE pooled H4", "H4_secondary_pooled")):
            r = results[key]
            term='N' if key=='H1' else 'N:free_mode'
            lines += [f"## {hypothesis}", f"Term: {term}; coefficient={r['coefficient']:.6g}; "
                      f"95% CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}]; two-sided p={r['p_two_sided']:.6g}."]
            if key=='H1':
                supported = r['p_two_sided'] < .05 and r['coefficient'] < 0
                lines.append('Evidence supports H1.' if supported else 'H1 is not supported at alpha=0.05.')
            else:lines.append('Descriptive only; this pooled p-value is not primary confirmatory evidence for H4.')
        lines += [f"Random effects selected: {attempts[-1]['random_formula']}.",
                  "Null per-model H4 results remain valid reportable outcomes; they do not trigger new post-hoc hypotheses."]
        # Marginal fitted means remain available even if a boundary covariance prevents BLUPs.
        predictions = np.asarray(fit.model.exog @ fit.fe_params)
        diagnostics = data.copy()
        diagnostics["marginal_prediction"] = predictions
        diagnostics["marginal_residual"] = data.cooperation - predictions
        diagnostics.to_csv(output / "primary_diagnostics.csv", index=False)
        results["predictions_outside_unit_interval"] = int(((predictions < 0) | (predictions > 1)).sum())
    try:
        curves = curve_comparison(data)
        results["H2"] = curves
        lines += ["## H2: pooled structured condition", "Gaussian ML fits on all six models' structured episode means: "
                  "flat C=a versus dilution C=a+b/(N-1). Both fits include an intercept; residual variance counts in AIC/BIC.",
                  "| Curve | AIC | BIC |", "|---|---:|---:|"]
        for name in ("flat", "dilution"):
            lines.append(f"| {name} | {curves[name]['aic']:.6g} | {curves[name]['bic']:.6g} |")
        lines += [f"Delta AIC (flat minus dilution): {curves['delta_aic_flat_minus_dilution']:.6g}; "
                  f"delta BIC: {curves['delta_bic_flat_minus_dilution']:.6g}. Positive favors dilution.",
                  f"Dilution coefficient b={curves['dilution']['coefficients']['dilution']:.6g}; b>0 implies declining cooperation with N."]
    except ValueError as exc:
        results["H2"] = {"error": str(exc)}
        lines += ["## H2", f"Not estimable: {exc}"]
    results['pooled_power_limitation']={'label':'prior pilot-based six-model population sensitivity, not main-data achieved power',
                                       'normal_ceiling_5_10_15_points':[.1845985783592366,.5613817471837477,.887168191857226],
                                       'df5_ceiling_5_10_15_points':[.1395343917271507,.4026946596902156,.7185408040967691]}
    results['pooled_fit_status']=results['primary_status']
    results['primary_status']='converged' if all(e['status']=='estimated' for e in per_model) else 'partially_not_estimable'
    lines += ['## SECONDARY pooled H4: known power limitation',
              'Prior pilot-based population-interaction normal power ceilings with six models: '
              '18.46%, 56.14%, 88.72% at 5/10/15-point effects, even with infinite episode seeds. '
              'df=5 sensitivities: 13.95%, 40.27%, 71.85%. These are historical planning estimates, not achieved main power. '
              'Between-model interaction variance creates the floor. The primary Q/I^2 result above assesses the heterogeneity '
              'underlying that concern, rather than attributing it solely to insufficient seeds. '
              'The unchanged pooled fit lacks a random interaction slope; its p-value is descriptive under this amendment.',
              "## Pooled fit diagnostics", "```json", __import__('json').dumps(attempts, indent=2), "```", TRAINING_NOTE]
    (output / "confirmatory_results_summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "confirmatory_results.json", results)

    # Fully interacted fixed effects permit genuinely per-model slopes in each mode.
    fixed = smf.ols("cooperation ~ N * free_mode * C(model)", data).fit()
    breakdown = []
    for model in manifest["plan"]["models"]:
        for mode, free in (("structured", 0), ("free_form", 1)):
            predictions = fixed.predict(pd.DataFrame({"model": [model, model], "N": [2, 5], "free_mode": [free, free]}))
            breakdown.append({"model": model, "mode": mode, "N_slope": float((predictions.iloc[1] - predictions.iloc[0]) / 3),
                              "training_group": "reasoning-tuned" if model in REASONING_MODELS else "standard"})
    table = pd.DataFrame(breakdown)
    table.to_csv(output / "descriptive_model_slopes.csv", index=False)
    exploratory = ["# Exploratory and descriptive results", "E1 is exploratory, non-pre-registered, and not confirmatory.",
                   "## Descriptive model breakdown", "Fixed-effect refit: cooperation ~ N * free_mode * C(model). "
                   "Model interactions allow separate slopes. This is not a second confirmatory test.",
                   "| Model | Mode | N slope |", "|---|---|---:|"]
    for row in breakdown:
        exploratory.append(f"| {row['model']} | {row['mode']} | {row['N_slope']:.6g} |")
    e1 = {}
    exploratory += ["## E1: exploratory, non-pre-registered comparison", "Equal model weights; two reasoning-tuned models versus four standard models. No confirmatory p-value."]
    for mode in ("structured", "free_form"):
        grouped = table[table["mode"] == mode].groupby("training_group")["N_slope"].mean()
        delta = float(grouped["reasoning-tuned"] - grouped["standard"])
        e1[mode] = {"reasoning_mean_slope": float(grouped["reasoning-tuned"]),
                    "standard_mean_slope": float(grouped["standard"]), "difference": delta}
        exploratory.append(f"{mode}: reasoning slope={grouped['reasoning-tuned']:.6g}; standard slope={grouped['standard']:.6g}; "
                           f"difference={delta:.6g} (negative means a steeper decline in the reasoning group).")
    exploratory += ["Training process, architecture, and model scale are confounded; this comparison is descriptive.", TRAINING_NOTE]
    from scipy.stats import fisher_exact
    directions=[dict(model=e['model'],reasoning_tuned=e['model'] in REASONING_MODELS,
                     sign=1 if e['endpoint_interaction']>0 else -1 if e['endpoint_interaction']<0 else 0) for e in per_model]
    binary=[int(e['reasoning_tuned']) for e in directions]; signs=[e['sign'] for e in directions]
    correlation=float(np.corrcoef(binary,signs)[0,1]) if np.std(signs)>0 else None
    contingency=[[sum(e['reasoning_tuned']==group and (e['sign']>0)==positive for e in directions)
                  for positive in (True,False)] for group in (True,False)]
    fisher_p=float(fisher_exact(contingency)[1])
    clean=all(len({e['sign'] for e in directions if e['reasoning_tuned']==group})==1 for group in (True,False)) and len(set(signs))==2
    direction_result={'label':'exploratory only','directions':directions,'sign_status_correlation':correlation,
                      'positive_vs_nonpositive_fisher_p':fisher_p,'contingency_reasoning_standard_by_positive_nonpositive':contingency,
                      'clean_separation':clean,'caveat':'Two reasoning-tuned models, four standard; sign ignores magnitude and uncertainty; no causal attribution'}
    exploratory += ['## Exploratory interaction direction versus reasoning-tuned status',
                    'Clean separation of signs.' if clean else 'No clean separation of interaction directions.',
                    f'Sign/status correlation={correlation}; exploratory Fisher p={fisher_p:.6g}. '
                    'Six models only; confounding and uncertain effect signs prevent causal or confirmatory interpretation.',
                    '```json',__import__('json').dumps(direction_result,indent=2),'```']
    (output / "exploratory_results_summary.md").write_text("\n\n".join(exploratory) + "\n", encoding="utf-8")
    write_json(output / "exploratory_results.json", {"label": "exploratory/non-pre-registered", "E1": e1, "descriptive_slopes": breakdown,
                                                  'interaction_direction_vs_reasoning_status':direction_result})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.input, args.output)
        print(f"Wrote confirmatory_results_summary.md and exploratory_results_summary.md; primary status: {result['primary_status']}")
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

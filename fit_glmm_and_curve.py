"""H1/H4 Gaussian mixed model, H2 pooled curve comparison, descriptive E1."""
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
    lines = ["# Confirmatory results: H1, H2, H4", "One independent episode mean per observation; no round-level pseudoreplication.",
             "Gaussian identity-link mixed model: cooperation ~ N * free_mode, with model identity as random intercept "
             "and a random N slope if a usable converged fit is obtained. Structured is the reference (free_mode=0).",
             "Cooperation is graded, not a count of binary successes. This is a Gaussian mixed model, not a binomial GLMM.",
             "Wald 95% CIs and asymptotic two-sided p-values; alpha=0.05, unadjusted per hypothesis. "
             "Only six model groups: interpret asymptotic uncertainty cautiously. Residuals/predictions are exported for diagnostics."]
    if fit is None:
        results["primary_status"] = "failed"
        lines += ["## H1", "Not estimable: no usable converged mixed model.",
                  "## H4", "Not estimable: no usable converged mixed model. This is not a null result."]
    else:
        results.update(primary_status="converged", selected_random_formula=attempts[-1]["random_formula"],
                       H1=coefficient(fit, "N"), H4=coefficient(fit, "N:free_mode"))
        for hypothesis, term in (("H1", "N"), ("H4", "N:free_mode")):
            r = results[hypothesis]
            lines += [f"## {hypothesis}", f"Term: {term}; coefficient={r['coefficient']:.6g}; "
                      f"95% CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}]; two-sided p={r['p_two_sided']:.6g}."]
            supported = r["p_two_sided"] < .05 and (hypothesis == "H4" or r["coefficient"] < 0)
            lines.append("Evidence supports the specified hypothesis." if supported else "The specified hypothesis is not supported at alpha=0.05.")
        lines += [f"Random effects selected: {attempts[-1]['random_formula']}.",
                  "A null H4 is a valid reportable outcome. It does not trigger new post-hoc hypotheses."]
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
    lines += ["## Fit diagnostics", "```json", __import__('json').dumps(attempts, indent=2), "```", TRAINING_NOTE]
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
    (output / "exploratory_results_summary.md").write_text("\n\n".join(exploratory) + "\n", encoding="utf-8")
    write_json(output / "exploratory_results.json", {"label": "exploratory/non-pre-registered", "E1": e1, "descriptive_slopes": breakdown})
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

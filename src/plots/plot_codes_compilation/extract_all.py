"""
Comprehensive data extractor across all 4 experiment axes.
Uses os.walk to avoid Windows mixed-separator glob issues.
"""
import json, os, csv
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_ROOT = Path("src/results")

def parse_traces(trace_path):
    rows = []
    with open(trace_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def extract_episode(phase_dir, experiment_type, condition_label):
    phase_dir = Path(phase_dir)
    meta_path = phase_dir / "episode_meta.json"
    if not meta_path.exists():
        return []

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    config = meta.get('config', {})
    extra  = meta.get('extra_meta', {})
    C = config.get('payoff', {}).get('C', 8.0)
    K = config.get('payoff', {}).get('K', 0.0)
    N = config.get('N')
    M = config.get('M')
    p     = config.get('p_perception', extra.get('noise_p', 0.0))
    theta = config.get('streak', {}).get('theta', extra.get('theta', 0.6))
    lam   = config.get('streak', {}).get('lam', extra.get('streak_lambda', 0.25))

    # Build seat-id -> full label
    seat_assign   = extra.get('seat_assignment', {})
    lineup_labels = extra.get('lineup_labels', [])
    lineup_short  = extra.get('lineup_short_labels', [])
    short_to_full = {s: lineup_labels[i] for i, s in enumerate(lineup_short) if i < len(lineup_labels)}
    seat_to_full  = {int(k): short_to_full.get(v, v) for k, v in seat_assign.items()}

    # Load all agent traces with os.walk
    agents = {}
    agents_root = phase_dir / "agents"
    if agents_root.is_dir():
        for agent_dir in agents_root.iterdir():
            trace_path = agent_dir / "agent_traces.jsonl"
            # Windows long path fix
            abs_trace_path = "\\\\?\\" + str(trace_path.absolute())
            if not os.path.isfile(abs_trace_path):
                continue
            parts = agent_dir.name.split("__")
            try:
                agent_id = int(parts[0][1:])
            except Exception:
                continue
            raw_label = parts[1] if len(parts) > 1 else f"agent_{agent_id}"
            model = seat_to_full.get(agent_id, short_to_full.get(raw_label, raw_label))
            agents[agent_id] = {'model': model, 'rounds': parse_traces(abs_trace_path)}

    if not agents:
        return []

    # Round-level cooperation map (for computing real reward)
    round_coops = defaultdict(dict)
    for aid, adata in agents.items():
        for r in adata['rounds']:
            rnd = r.get('round')
            act = r.get('decision', {}).get('a')
            if rnd is None or act is None:
                continue
            m = r.get('M', M) or M
            cmap = [1.0] if m == 1 else [i/(m-1) for i in range(m-1, -1, -1)]
            tc = cmap[act] if act < len(cmap) else 0.0
            round_coops[rnd][aid] = tc

    records = []
    for aid, adata in agents.items():
        prev_action = None
        for r in sorted(adata['rounds'], key=lambda x: x.get('round', 0)):
            rnd  = r.get('round')
            act  = r.get('decision', {}).get('a')
            reason     = r.get('decision', {}).get('reason', '')
            confidence = r.get('decision', {}).get('confidence')
            b_eff = r.get('B_eff') or r.get('payoff', {}).get('B_effective', 12.0)
            m_loc = r.get('M', M) or M

            cmap = [1.0] if m_loc == 1 else [i/(m_loc-1) for i in range(m_loc-1, -1, -1)]
            true_coop = cmap[act] if (act is not None and act < len(cmap)) else None

            candidates = r.get('candidates', [])
            eu = next((c.get('EU') for c in candidates if c.get('action') == act), None)

            strat = r.get('strategy_summary', {})
            retaliation    = strat.get('retaliation_pressure', 0)
            forgiveness    = strat.get('forgiveness_pressure', 0)
            target_coop    = strat.get('target_coop')
            expected_other = strat.get('expected_other_true_coop')
            belief_other   = strat.get('belief_other_coop')

            streak_prev = r.get('streak_rule', {}).get('streak_prev', 0)
            action_changed = (prev_action is not None and act != prev_action)
            prev_action = act

            others = [v for k, v in round_coops[rnd].items() if k != aid and v is not None]
            avg_other = sum(others)/len(others) if others else 0.0
            real_reward = b_eff * avg_other - C * (true_coop or 0.0) + K

            opp_styles  = r.get('opponent_style_summary', {})
            n_defective = sum(1 for v in opp_styles.values() if v.get('style','') == 'defective_sticky')
            n_reciprocal= sum(1 for v in opp_styles.values() if v.get('style','') == 'reciprocal')

            # B_eff rate of change (momentum)
            payoff_drift = r.get('drift_rule', {}).get('r_obs_prev')

            records.append(dict(
                experiment=experiment_type,
                condition=condition_label,
                phase_dir=phase_dir.name,
                N=N, M=m_loc, p_noise=p, theta=theta, lam=lam,
                round=rnd, agent_id=aid, model=adata['model'],
                action=act, true_coop=true_coop, B_eff=b_eff,
                EU=eu, real_reward=real_reward, avg_other_coop=avg_other,
                expected_other_coop=expected_other, belief_other_coop=belief_other,
                retaliation_pressure=retaliation, forgiveness_pressure=forgiveness,
                target_coop=target_coop, streak_prev=streak_prev,
                action_changed=action_changed, confidence=confidence,
                n_opp_defective=n_defective, n_opp_reciprocal=n_reciprocal,
                payoff_drift=payoff_drift,
                reason=str(reason)[:250],
            ))
    return records


all_records = []

# ── EXPERIMENT 1: causal_N_progressive (True Results Only) ─────────────────
true_cnp_dir = RESULTS_ROOT / "causal_N_progressive" / "cnp__N5_M5_T50__p0.05__lam0.25__eta0.35__seeds1__n2-3-4-5__progressive__m5__t50__20260423_140952"
if true_cnp_dir.is_dir():
    for phase_dir in sorted(true_cnp_dir.iterdir()):
        if not phase_dir.name.startswith("progressive__n"):
            continue
        try:
            n_val = int(phase_dir.name.split("__n")[1].split("__")[0])
        except Exception:
            continue
        all_records.extend(extract_episode(phase_dir, "causal_N_progressive", f"N={n_val}"))

# ── EXPERIMENT 2: causal_M (True Results Only) ──────────────────────────────
true_cm_dir = RESULTS_ROOT / "causal_M" / "cm__N5__M2-3-4-5__p0.05__lam0.25__seeds1__t50__20260423_103251"
if true_cm_dir.is_dir():
    for match_dir in sorted(true_cm_dir.iterdir()):
        if "__n5__m" not in match_dir.name:
            continue
        try:
            m_val = int(match_dir.name.split("__m")[1].split("__")[0])
        except Exception:
            m_val = None
        all_records.extend(extract_episode(match_dir, "causal_M", f"M={m_val}"))

# ── EXPERIMENT 3: causal_noise (True Results Only) ──────────────────────────
true_cn_dir = RESULTS_ROOT / "causal_noise" / "cn_n5_m5_0423_131931"
if true_cn_dir.is_dir():
    for match_dir in sorted(true_cn_dir.iterdir()):
        if not match_dir.is_dir() or not match_dir.name.startswith("m_"):
            continue
        try:
            p_val = int(match_dir.name.split("_p")[1].split("_")[0]) / 100.0
        except Exception:
            p_val = None
        all_records.extend(extract_episode(match_dir, "causal_noise", f"p={p_val}"))

# ── EXPERIMENT 4: causal_theta (True Results Only) ──────────────────────────
true_ct_dir = RESULTS_ROOT / "causal_theta" / "ct__N5__M5__th0p10-0p25-0p50-0p75__p0.05__lam0.25__seeds1__t50__20260423_134651"
if true_ct_dir.is_dir():
    for match_dir in sorted(true_ct_dir.iterdir()):
        if "__th" not in match_dir.name:
            continue
        try:
            th_str = match_dir.name.split("__th")[1].split("__")[0]
            th_val = float(th_str.replace("p", "."))
        except Exception:
            th_val = None
        all_records.extend(extract_episode(match_dir, "causal_theta", f"theta={th_val}"))


print(f"Total records: {len(all_records)}")
exp_counts = Counter(r['experiment'] for r in all_records)
print("Per experiment:", dict(exp_counts))
print("\nConditions:")
cond_counts = Counter((r['experiment'], r['condition']) for r in all_records)
for (exp, cond), cnt in sorted(cond_counts.items()):
    print(f"  {exp:25s} | {cond:15s}: {cnt} rows")

# Write master CSV
if all_records:
    with open('master_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print("\nMaster CSV written: master_data.csv")

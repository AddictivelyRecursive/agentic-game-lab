import json
import glob
import os
import csv
from collections import defaultdict

base_path = r"c:\Users\Arjun\agentic-game-lab\src\results\causal_N_progressive\cnp__N5_M5_T50__p0.05__lam0.25__eta0.35__seeds1__n2-3-4-5__progressive__m5__t5__20260423_003152"
phase_dirs = sorted(glob.glob(os.path.join(base_path, "progressive__n*")))

for phase_dir in phase_dirs:
    phase_name = os.path.basename(phase_dir)
    print(f"\n========================================")
    print(f"Analyzing Phase: {phase_name}")
    print(f"========================================")
    
    meta_path = os.path.join(phase_dir, "episode_meta.json")
    if not os.path.exists(meta_path):
        print("No episode_meta.json found.")
        continue
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    config = meta.get('config', {})
    C = config.get('payoff', {}).get('C', 8.0)
    K = config.get('payoff', {}).get('K', 0.0)
    print(f"N: {config.get('N')}, M: {config.get('M')}, T: {config.get('T')}")

    agents_data = {}
    trace_files = glob.glob(os.path.join(phase_dir, "agents", "*", "agent_traces.jsonl"))

    for path in trace_files:
        agent_dir = os.path.basename(os.path.dirname(path))
        agent_id = int(agent_dir.split("__")[0][1:])
        model_name = agent_dir.split("__", 1)[1] if "__" in agent_dir else agent_dir
        
        rounds = []
        with open(path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                rounds.append(data)
        
        agents_data[agent_id] = {
            'model': model_name,
            'rounds': rounds
        }

    print(f"Loaded data for {len(agents_data)} agents.")

    # Calculate real rewards
    round_actions = {}
    for r_idx in range(1, 51):
        round_actions[r_idx] = {}
        for aid in agents_data:
            round_data = next((r for r in agents_data[aid]['rounds'] if r['round'] == r_idx), None)
            if round_data:
                act = round_data.get('decision', {}).get('a')
                coop_mapping = [1.0, 0.75, 0.5, 0.25, 0.0]
                true_coop = coop_mapping[act] if act is not None and act < len(coop_mapping) else None
                round_actions[r_idx][aid] = true_coop

    agent_total_real_reward = defaultdict(float)

    csv_path = f"summary_{phase_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['round', 'agent_id', 'model', 'action', 'true_coop', 'B_eff', 'EU', 'real_reward', 'expected_other_coop', 'reason', 'action_changed'])
        
        for r_idx in range(1, 51):
            for aid in sorted(agents_data.keys()):
                round_data = next((r for r in agents_data[aid]['rounds'] if r['round'] == r_idx), None)
                if not round_data:
                    continue
                    
                decision = round_data.get('decision', {})
                action = decision.get('a')
                reason = decision.get('reason', '')
                b_eff = round_data.get('B_eff', 12.0)
                
                candidates = round_data.get('candidates', [])
                eu = next((c.get('EU') for c in candidates if c.get('action') == action), None)
                
                strategy = round_data.get('strategy_summary', {})
                expected_other_coop = strategy.get('expected_other_true_coop')
                
                action_changed = False
                if r_idx > 1:
                    prev_round = next((r for r in agents_data[aid]['rounds'] if r['round'] == r_idx - 1), None)
                    if prev_round:
                        prev_action = prev_round.get('decision', {}).get('a')
                        action_changed = (action != prev_action)
                
                true_coop = round_actions[r_idx].get(aid)
                
                other_coops = [round_actions[r_idx][other] for other in round_actions[r_idx] if other != aid and round_actions[r_idx][other] is not None]
                avg_other_coop = sum(other_coops) / len(other_coops) if other_coops else 0.0
                
                real_reward = b_eff * avg_other_coop - C * (true_coop if true_coop else 0.0) + K
                agent_total_real_reward[aid] += real_reward
                
                writer.writerow([r_idx, aid, agents_data[aid]['model'], action, true_coop, b_eff, eu, real_reward, expected_other_coop, reason, action_changed])

    print("\n--- Agent Stats ---")
    for aid in sorted(agents_data.keys()):
        model = agents_data[aid]['model']
        rounds = agents_data[aid]['rounds']
        action_changes = 0
        prev_action = None
        cum_eu = 0.0
        initial_coop = None
        
        for r in rounds:
            act = r.get('decision', {}).get('a')
            if initial_coop is None and act is not None:
                coop_mapping = [1.0, 0.75, 0.5, 0.25, 0.0]
                initial_coop = coop_mapping[act] if act < len(coop_mapping) else None
                
            if prev_action is not None and act != prev_action:
                action_changes += 1
            prev_action = act
            
            cands = r.get('candidates', [])
            eu = next((c.get('EU', 0) for c in cands if c.get('action') == act), 0)
            cum_eu += eu
            
        print(f"Agent {aid} ({model}): {len(rounds)} rounds played, {action_changes} action changes, initial coop={initial_coop}, total real reward={agent_total_real_reward[aid]:.2f}")


"""
agents/best_response_freq.py

Purpose:
- Best-response baseline using action frequency matrix action_freq_NxM inferred
  from obs.observed_history_packed (observed actions up to t-1).

How it works:
- Estimate opponents' expected cooperation as E[c_other] from their empirical action frequencies.
- For each candidate action a, compute expected reward:
    u(a) = B_eff * E[avg_other_coop] - C * coop(a) + K
  where coop(a) is mapping from action index to cooperation in [0,1].

Notes:
- This is "myopic" best-response to current estimates (no long-term effects).
- Uses observed history only; robust and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from game_engine.env.types import AgentMeta, Observation
from .base import BaseAgent


@dataclass(frozen=True)
class ActionSemantics:
    index_to_cooperation: List[float]
    note: str = "Lower index = more cooperation"

    @staticmethod
    def default_for_M(M: int) -> "ActionSemantics":
        if M == 5:
            return ActionSemantics([1.0, 0.75, 0.5, 0.25, 0.0])
        if M < 2:
            return ActionSemantics([1.0])
        step = 1.0 / (M - 1)
        return ActionSemantics([1.0 - i * step for i in range(M)])


def _unpack_row(s: str, M: int) -> List[int]:
    if not isinstance(s, str) or len(s) == 0:
        return []
    if M > 10:
        raise ValueError("Packed history uses single digits; requires M <= 10.")
    out: List[int] = []
    for ch in s:
        if "0" <= ch <= "9":
            a = int(ch)
            if 0 <= a < M:
                out.append(a)
    return out


def _freq_from_history_rows(history_rows: List[List[int]], M: int) -> List[List[float]]:
    out: List[List[float]] = []
    for row in history_rows:
        counts = [0] * M
        total = 0
        for a in row:
            if 0 <= a < M:
                counts[a] += 1
                total += 1
        if total == 0:
            out.append([0.0] * M)
        else:
            out.append([c / total for c in counts])
    return out


class BestResponseToFrequencies(BaseAgent):
    def __init__(self, name: str = "best_response_freq"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> Tuple[int, AgentMeta]:
        sem = ActionSemantics.default_for_M(obs.M)

        # Build NxK observed history matrix (oldest->newest)
        hist_rows = [_unpack_row(s, obs.M) for s in obs.observed_history_packed]

        # If no history yet, start cooperative
        if all(len(r) == 0 for r in hist_rows):
            return 0, AgentMeta(agent_name=self.name, extra={"reason": "no_history"})

        # Empirical action frequencies for each player
        freq_nxm = _freq_from_history_rows(hist_rows, obs.M)

        # Expected cooperation of each opponent under their empirical distribution
        opp_coops: List[float] = []
        for j in range(obs.N):
            if j == obs.agent_id:
                continue
            pj = freq_nxm[j]
            ec = 0.0
            for a in range(obs.M):
                ec += pj[a] * sem.index_to_cooperation[a]
            opp_coops.append(ec)

        # Expected avg cooperation of others
        if len(opp_coops) == 0:
            e_avg_other = 0.0
        else:
            e_avg_other = sum(opp_coops) / len(opp_coops)

        # Myopic best response: maximize expected one-step reward with current B_eff
        best_a = 0
        best_u = float("-inf")
        for a in range(obs.M):
            own_coop = sem.index_to_cooperation[a]
            u = float(obs.B_eff) * e_avg_other - float(obs.C) * own_coop + float(obs.K)
            if u > best_u:
                best_u = u
                best_a = a

        return best_a, AgentMeta(
            agent_name=self.name,
            extra={
                "e_avg_other_coop": e_avg_other,
                "best_u": best_u,
            },
        )
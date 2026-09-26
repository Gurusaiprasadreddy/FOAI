"""
agents/response.py
Response Agent — Minimax with alpha-beta pruning, then Expectimax for stochastic
exploit outcomes.

Algorithm justification:
  - Minimax models the adversarial Red vs Blue game; Blue maximises utility,
    Red minimises it.
  - Alpha-beta pruning reduces the effective branching factor to ≈√b (b=branching)
    making depth-3 searches feasible even on 100+ node graphs.
  - Expectimax replaces the Red min-node with a CHANCE node when exploit
    success is probabilistic (Red doesn't control whether the exploit succeeds).
  - The game tree is pruned to only consider Red actions on the top-k most-likely
    attacker locations (from Monitor's belief) — keeps the tree tractable.

Utility function:
  utility = w_uptime * uptime_fraction
           - w_compromise * compromised_value
           - w_cost * total_action_cost
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.environment.events import ActionEvent
from sentinelnet.environment.network_graph import NetworkGraph, NodeState
from sentinelnet.coordination.message_bus import MessageBus


# ---------------------------------------------------------------------------
# Lightweight state snapshot for the game tree
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Immutable snapshot of network state used in the minimax search tree.

    We store only the minimal information needed for the utility function to
    avoid deep-copying the full NetworkX graph at every tree node.
    """
    node_states: Dict[str, str]          # {node_id: state_value}
    node_values: Dict[str, int]          # {node_id: asset_value}
    node_patched: Dict[str, bool]        # {node_id: patched}
    node_vulns: Dict[str, float]         # {node_id: effective_vuln}
    crown_jewel: str
    total_action_cost: int = 0
    is_terminal_flag: bool = False

    @classmethod
    def from_graph(cls, graph: NetworkGraph, total_cost: int = 0) -> "GameState":
        states, values, patched, vulns = {}, {}, {}, {}
        for n in graph.all_nodes():
            nd = graph.node_data(n)
            states[n] = nd.state.value
            values[n] = nd.value
            patched[n] = nd.patched
            vulns[n] = nd.effective_vuln
        return cls(
            node_states=states,
            node_values=values,
            node_patched=patched,
            node_vulns=vulns,
            crown_jewel=graph.crown_jewel,
            total_action_cost=total_cost,
            is_terminal_flag=(states.get(graph.crown_jewel) == NodeState.COMPROMISED.value),
        )

    def is_terminal(self) -> bool:
        return self.is_terminal_flag or \
               self.node_states.get(self.crown_jewel) == NodeState.COMPROMISED.value

    def uptime_fraction(self) -> float:
        total = len(self.node_states)
        if total == 0:
            return 1.0
        down = sum(1 for s in self.node_states.values()
                   if s in (NodeState.ISOLATED.value, NodeState.COMPROMISED.value))
        return (total - down) / total

    def compromised_value(self) -> int:
        return sum(self.node_values[n]
                   for n, s in self.node_states.items()
                   if s == NodeState.COMPROMISED.value)


# ---------------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------------

def utility(
    state: GameState,
    w_uptime: float = 10.0,
    w_compromise: float = 5.0,
    w_cost: float = 0.5,
) -> float:
    """Weighted utility for the Blue team.

    Higher uptime and lower compromise value are better for Blue.

    Args:
        state: The game state to evaluate.
        w_uptime: Weight for service uptime (Blue wants to maximise).
        w_compromise: Weight for compromised asset value (Blue wants to minimise).
        w_cost: Weight for response cost (Blue wants to minimise).

    Returns:
        Scalar utility score.
    """
    return (
        w_uptime * state.uptime_fraction()
        - w_compromise * state.compromised_value()
        - w_cost * state.total_action_cost
    )


# ---------------------------------------------------------------------------
# Action generators
# ---------------------------------------------------------------------------

def blue_actions(state: GameState, top_suspect_nodes: List[str]) -> List[Tuple[str, str]]:
    """Generate candidate Blue actions focused on suspect areas.

    Returns list of (action_type, node) tuples.
    """
    actions = []
    for node in top_suspect_nodes:
        s = state.node_states.get(node, "safe")
        if s in ("safe", "suspected"):
            actions.append(("isolate", node))
        if not state.node_patched.get(node, False) and s not in ("isolated", "compromised"):
            actions.append(("patch", node))
        if s == "compromised":
            actions.append(("restore", node))
    if not actions:
        # Fallback: no-op
        actions.append(("noop", ""))
    return actions


def red_actions(state: GameState, top_suspect_nodes: List[str]) -> List[Tuple[str, float]]:
    """Generate plausible Red actions based on belief (node, exploit_prob) pairs."""
    actions = []
    for node in top_suspect_nodes:
        s = state.node_states.get(node, "safe")
        if s not in ("isolated",):
            vuln = state.node_vulns.get(node, 0.5)
            actions.append((node, vuln))
    return actions or [("", 0.0)]


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def apply_blue_action(state: GameState, action: Tuple[str, str]) -> GameState:
    """Return a new state after Blue applies action."""
    action_type, node = action
    new_states = dict(state.node_states)
    new_patched = dict(state.node_patched)
    cost = state.total_action_cost

    if action_type == "isolate" and node:
        new_states[node] = NodeState.ISOLATED.value
        cost += 1
    elif action_type == "patch" and node:
        new_states[node] = NodeState.PATCHING.value
        new_patched[node] = True
        cost += 2
    elif action_type == "restore" and node:
        new_states[node] = NodeState.SAFE.value
        cost += 3
    elif action_type == "noop":
        pass

    new_is_terminal = new_states.get(state.crown_jewel) == NodeState.COMPROMISED.value

    return GameState(
        node_states=new_states,
        node_values=state.node_values,
        node_patched=new_patched,
        node_vulns=state.node_vulns,
        crown_jewel=state.crown_jewel,
        total_action_cost=cost,
        is_terminal_flag=new_is_terminal,
    )


def apply_red_action(state: GameState, node: str, success: bool) -> GameState:
    """Return a new state after Red attempts to compromise a node."""
    new_states = dict(state.node_states)
    if success and node:
        new_states[node] = NodeState.COMPROMISED.value
    new_is_terminal = new_states.get(state.crown_jewel) == NodeState.COMPROMISED.value
    return GameState(
        node_states=new_states,
        node_values=state.node_values,
        node_patched=state.node_patched,
        node_vulns=state.node_vulns,
        crown_jewel=state.crown_jewel,
        total_action_cost=state.total_action_cost,
        is_terminal_flag=new_is_terminal,
    )


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def minimax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    top_suspects: List[str],
) -> Tuple[float, Optional[Tuple[str, str]]]:
    """Depth-limited minimax search with alpha-beta pruning.

    Blue (maximising) chooses from blue_actions; Red (minimising) chooses
    the worst-case action for Blue from red_actions.

    Args:
        state: Current game state snapshot.
        depth: Remaining search depth (0 = evaluate).
        alpha: Best value Blue can guarantee so far.
        beta: Best value Red can guarantee so far.
        maximizing: True when it is Blue's turn.
        top_suspects: Nodes to focus action generation on.

    Returns:
        (best_value, best_action) — best_action is None at leaf nodes.
    """
    if depth == 0 or state.is_terminal():
        return utility(state), None

    best_action: Optional[Tuple[str, str]] = None

    if maximizing:  # Blue's turn — maximise utility
        value = float("-inf")
        for action in blue_actions(state, top_suspects):
            child = apply_blue_action(state, action)
            score, _ = minimax(child, depth - 1, alpha, beta, False, top_suspects)
            if score > value:
                value, best_action = score, action
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # β-cutoff
        return value, best_action

    else:  # Red's turn — minimise utility (worst case for Blue)
        value = float("inf")
        for node, _prob in red_actions(state, top_suspects):
            # Red always succeeds in pure minimax (worst case)
            child = apply_red_action(state, node, success=True)
            score, _ = minimax(child, depth - 1, alpha, beta, True, top_suspects)
            if score < value:
                value = score
            beta = min(beta, value)
            if alpha >= beta:
                break  # α-cutoff
        return value, best_action


# ---------------------------------------------------------------------------
# Expectimax (stochastic Red exploits)
# ---------------------------------------------------------------------------

def expectimax(
    state: GameState,
    depth: int,
    maximizing: bool,
    top_suspects: List[str],
) -> Tuple[float, Optional[Tuple[str, str]]]:
    """Expectimax search — Red's nodes become CHANCE nodes.

    At Red's turn the value is E[utility] = sum_over_outcomes(p * value(outcome))
    where p is the exploit success probability (vuln_score).

    Args:
        state: Current game state.
        depth: Remaining depth.
        maximizing: True when it is Blue's turn.
        top_suspects: Focus nodes.

    Returns:
        (expected_value, best_action).
    """
    if depth == 0 or state.is_terminal():
        return utility(state), None

    best_action: Optional[Tuple[str, str]] = None

    if maximizing:  # Blue — deterministic maximiser
        value = float("-inf")
        for action in blue_actions(state, top_suspects):
            child = apply_blue_action(state, action)
            score, _ = expectimax(child, depth - 1, False, top_suspects)
            if score > value:
                value, best_action = score, action
        return value, best_action

    else:  # Red — CHANCE node (expected value over exploit success/failure)
        expected_value = 0.0
        red_acts = red_actions(state, top_suspects)
        if not red_acts or red_acts == [("", 0.0)]:
            return utility(state), None

        # Evenly weight each possible Red target (uniform Red policy)
        weight = 1.0 / len(red_acts)
        for node, exploit_prob in red_acts:
            # Success branch
            success_state = apply_red_action(state, node, success=True)
            v_success, _ = expectimax(success_state, depth - 1, True, top_suspects)
            # Failure branch
            fail_state = apply_red_action(state, node, success=False)
            v_fail, _ = expectimax(fail_state, depth - 1, True, top_suspects)
            # Expected value for this Red action
            ev = exploit_prob * v_success + (1.0 - exploit_prob) * v_fail
            expected_value += weight * ev

        return expected_value, None


# ---------------------------------------------------------------------------
# Response Agent
# ---------------------------------------------------------------------------

class ResponseAgent(BaseAgent):
    """Response agent that selects defensive actions via minimax/expectimax.

    Args:
        depth: Minimax/expectimax search depth (2-3 recommended).
        top_k: Number of top-belief suspects to include in action generation.
        use_expectimax: If True use expectimax; otherwise plain minimax.
        resource_limited: If True, allow only 1 action per ``resource_interval`` ticks.
        resource_interval: Ticks between actions when resource_limited=True.
        bus: Shared message bus.
    """

    def __init__(
        self,
        depth: int = 2,
        top_k: int = 5,
        use_expectimax: bool = True,
        resource_limited: bool = False,
        resource_interval: int = 2,
        bus: Optional[MessageBus] = None,
    ) -> None:
        super().__init__(name="ResponseAgent", bus=bus)
        self.depth = depth
        self.top_k = top_k
        self.use_expectimax = use_expectimax
        self.resource_limited = resource_limited
        self.resource_interval = resource_interval
        self._total_cost: int = 0
        self._last_action: Optional[ActionEvent] = None

    def decide(
        self,
        graph: NetworkGraph,
        belief: Dict[str, float],
    ) -> Optional[ActionEvent]:
        """Run minimax/expectimax and return the chosen action.

        Args:
            graph: Current network graph.
            belief: Current belief dict from Monitor.

        Returns:
            An :class:`~sentinelnet.environment.events.ActionEvent` or None.
        """
        # Resource-limited defence: skip if not the right tick
        if self.resource_limited and self.tick % self.resource_interval != 0:
            return None

        from sentinelnet.agents.monitor import top_k_nodes
        top_suspects = [n for n, _ in top_k_nodes(belief, self.top_k)]

        state = GameState.from_graph(graph, self._total_cost)

        if self.use_expectimax:
            _, action = expectimax(state, self.depth, True, top_suspects)
        else:
            _, action = minimax(
                state, self.depth,
                float("-inf"), float("inf"),
                True, top_suspects,
            )

        if action is None:
            return None

        action_type, node = action
        if action_type == "noop" or not node:
            return None

        # Apply to real graph
        graph.apply_action(action_type, node)
        self._total_cost += 1

        event = ActionEvent(
            tick=self.tick,
            action_type=action_type,
            node=node,
            agent=self.name,
            cost=1,
        )
        self._last_action = event
        return event

    def step(self, graph: NetworkGraph) -> Optional[ActionEvent]:
        """Pull belief from bus, decide action, publish to bus."""
        self.tick += 1
        belief: Dict[str, float] = self.bus.latest("belief") or {}
        if not belief:
            return None
        action = self.decide(graph, belief)
        if action and self.bus:
            self.bus.publish("actions", action, tick=self.tick)
        return action

    def on_reset(self) -> None:
        super().on_reset()
        self._total_cost = 0
        self._last_action = None

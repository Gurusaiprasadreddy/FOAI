"""
agents/monitor.py
Monitor Agent — Bayesian belief tracking of attacker location.

Algorithm: Bayesian grid belief update + belief diffusion.

Each tick:
  1. Receive noisy alert(s) from the message bus.
  2. Run Bayesian update: P(attacker at node | alert) ∝ P(alert | at node) × prior.
  3. Run diffusion step: spread belief mass to neighbors (attacker might have moved).
  4. Publish updated belief dict to the 'belief' topic.

Stretch goal (particle filter):
  - If the graph is large (>200 nodes), switch to a particle filter variant
    that maintains a set of weighted particles instead of a dense dict.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sentinelnet.agents.base_agent import BaseAgent
from sentinelnet.environment.events import AlertEvent
from sentinelnet.environment.network_graph import NetworkGraph, NodeState
from sentinelnet.coordination.message_bus import MessageBus


# ---------------------------------------------------------------------------
# Core belief functions (pure — exposed for unit tests)
# ---------------------------------------------------------------------------

def bayesian_update(
    belief: Dict[str, float],
    alert_node: Optional[str],
    detection_rate: float,
    false_alarm_rate: float,
    graph: NetworkGraph,
) -> Dict[str, float]:
    """Apply a single Bayesian update step given an alert observation.

    P(attacker at node | alert_node observed) ∝
        P(alert_node | attacker at node) × P(attacker at node)

    where:
        P(alert | attacker at alerting node) = detection_rate
        P(alert | attacker NOT at alerting node) = false_alarm_rate

    Args:
        belief: Current belief dict {node_name: probability}.
        alert_node: The node that triggered the IDS alert, or None if no alert.
        detection_rate: True-positive rate of the IDS.
        false_alarm_rate: False-positive rate of the IDS.
        graph: The network graph (used to enumerate all nodes).

    Returns:
        Normalised posterior belief dict.
    """
    if alert_node is None:
        # No alert — small uniform update (attacker might be hiding)
        return dict(belief)

    new_belief: Dict[str, float] = {}
    for node, prior in belief.items():
        if node == alert_node:
            likelihood = detection_rate
        else:
            likelihood = false_alarm_rate
        new_belief[node] = prior * likelihood

    total = sum(new_belief.values())
    if total < 1e-12:
        # Numerical underflow — reset to uniform
        n = len(new_belief)
        return {node: 1.0 / n for node in new_belief}

    return {node: p / total for node, p in new_belief.items()}


def diffuse_belief(
    belief: Dict[str, float],
    graph: NetworkGraph,
    stay_prob: float = 0.6,
) -> Dict[str, float]:
    """Spread belief mass to graph neighbours to model attacker movement.

    With probability ``stay_prob`` the attacker stays at its current node;
    the remaining mass is split equally among reachable neighbours.

    Args:
        belief: Current belief dict {node_name: probability}.
        graph: The network graph.
        stay_prob: Probability that the attacker remains in place this tick.

    Returns:
        Diffused belief dict (sums to 1).
    """
    new_belief: Dict[str, float] = {n: 0.0 for n in belief}

    for node, prob in belief.items():
        # Attacker might stay
        new_belief[node] += prob * stay_prob

        # Or move to a neighbour
        neighbors = [
            nb for nb in graph.g.successors(node)
            if graph.node_data(nb).state != NodeState.ISOLATED
        ]
        if neighbors:
            share = prob * (1.0 - stay_prob) / len(neighbors)
            for nb in neighbors:
                if nb in new_belief:
                    new_belief[nb] += share
                # else: node not in belief space (shouldn't happen unless graph changed)

    # Renormalise
    total = sum(new_belief.values())
    if total < 1e-12:
        n = max(1, len(new_belief))
        return {node: 1.0 / n for node in new_belief}
    return {node: p / total for node, p in new_belief.items()}


def top_k_nodes(belief: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
    """Return the k most-likely attacker locations, sorted descending."""
    return sorted(belief.items(), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Particle Filter (stretch goal for large graphs)
# ---------------------------------------------------------------------------

def particle_filter_update(
    particles: List[str],
    weights: List[float],
    alert_node: Optional[str],
    detection_rate: float,
    false_alarm_rate: float,
    graph: NetworkGraph,
    rng,
    stay_prob: float = 0.6,
) -> Tuple[List[str], List[float]]:
    """One step of a particle filter for belief tracking.

    Args:
        particles: Current particle positions (list of node names).
        weights: Current importance weights (must sum to 1).
        alert_node: Observed alert node (or None).
        detection_rate: IDS true-positive rate.
        false_alarm_rate: IDS false-positive rate.
        graph: The network graph.
        rng: Random number generator.
        stay_prob: Probability particle stays at current node.

    Returns:
        (new_particles, new_weights) after resample.
    """
    n = len(particles)
    # --- Transition step: move particles ---
    new_particles = []
    for p in particles:
        if rng.random() < stay_prob:
            new_particles.append(p)
        else:
            neighbors = [nb for nb in graph.g.successors(p)
                         if graph.node_data(nb).state != NodeState.ISOLATED]
            if neighbors:
                new_particles.append(rng.choice(neighbors))
            else:
                new_particles.append(p)

    # --- Update weights based on observation ---
    new_weights = []
    for p, w in zip(new_particles, weights):
        if alert_node is None:
            new_weights.append(w)
        elif p == alert_node:
            new_weights.append(w * detection_rate)
        else:
            new_weights.append(w * false_alarm_rate)

    total = sum(new_weights)
    if total < 1e-12:
        new_weights = [1.0 / n] * n
    else:
        new_weights = [w / total for w in new_weights]

    # --- Low-variance resample ---
    resampled = []
    cumulative = []
    c = 0.0
    for w in new_weights:
        c += w
        cumulative.append(c)

    step = 1.0 / n
    u = rng.uniform(0, step)
    j = 0
    for _ in range(n):
        while u > cumulative[j]:
            j += 1
        resampled.append(new_particles[j])
        u += step

    uniform_w = 1.0 / n
    return resampled, [uniform_w] * n


def particles_to_belief(
    particles: List[str], graph: NetworkGraph
) -> Dict[str, float]:
    """Convert particle list to a normalised belief dict."""
    counts: Dict[str, int] = {}
    for p in particles:
        counts[p] = counts.get(p, 0) + 1
    n = max(1, len(particles))
    return {node: counts.get(node, 0) / n for node in graph.all_nodes()}


# ---------------------------------------------------------------------------
# Monitor agent
# ---------------------------------------------------------------------------

class MonitorAgent(BaseAgent):
    """Monitor agent that maintains a Bayesian belief over attacker location.

    Args:
        graph: The network graph (used to initialise the belief).
        detection_rate: P(alert | attacker at alerting node).
        false_alarm_rate: P(alert | attacker not at alerting node).
        stay_prob: Diffusion stay probability.
        use_particle_filter: If True, use particle filter instead of grid belief
            (recommended for graphs with >200 nodes).
        n_particles: Number of particles for the particle filter.
        seed: RNG seed.
        bus: Shared message bus.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        detection_rate: float = 0.7,
        false_alarm_rate: float = 0.05,
        stay_prob: float = 0.6,
        use_particle_filter: bool = False,
        n_particles: int = 500,
        seed: int = 42,
        bus: Optional[MessageBus] = None,
    ) -> None:
        super().__init__(name="MonitorAgent", bus=bus)
        self.detection_rate = detection_rate
        self.false_alarm_rate = false_alarm_rate
        self.stay_prob = stay_prob
        self.use_particle_filter = use_particle_filter

        import random
        self.rng = random.Random(seed)

        # Initialise uniform belief
        nodes = graph.all_nodes()
        n = len(nodes)
        self.belief: Dict[str, float] = {node: 1.0 / n for node in nodes}

        # Particle filter state
        self.particles: List[str] = [self.rng.choice(nodes) for _ in range(n_particles)]
        self.particle_weights: List[float] = [1.0 / n_particles] * n_particles

        # Track time-to-detect
        self._detected_at: Optional[int] = None
        self._true_location_history: List[str] = []

    def update(
        self,
        alert: Optional[AlertEvent],
        graph: NetworkGraph,
    ) -> Dict[str, float]:
        """Run one belief update step and return the new belief.

        Args:
            alert: The latest IDS alert (or None if no alert this tick).
            graph: Current network graph.

        Returns:
            Updated belief dict {node -> probability}.
        """
        alert_node = alert.node if alert else None

        if self.use_particle_filter:
            self.particles, self.particle_weights = particle_filter_update(
                self.particles, self.particle_weights,
                alert_node, self.detection_rate, self.false_alarm_rate,
                graph, self.rng, self.stay_prob,
            )
            self.belief = particles_to_belief(self.particles, graph)
        else:
            # Bayesian grid update
            self.belief = bayesian_update(
                self.belief, alert_node,
                self.detection_rate, self.false_alarm_rate, graph,
            )
            # Diffuse to model movement
            self.belief = diffuse_belief(self.belief, graph, self.stay_prob)

        # Zero out isolated nodes — attacker can't be there
        for node in graph.all_nodes():
            if graph.node_data(node).state == NodeState.ISOLATED:
                self.belief[node] = 0.0

        # Renormalise after zeroing
        total = sum(self.belief.values())
        if total > 1e-12:
            self.belief = {n: p / total for n, p in self.belief.items()}

        return self.belief

    def step(self, graph: NetworkGraph) -> Dict[str, float]:
        """Pull latest alert from bus, update belief, publish back to bus."""
        self.tick += 1
        alert = self.bus.latest("alerts") if self.bus else None
        belief = self.update(alert, graph)
        if self.bus:
            self.bus.publish("belief", belief, tick=self.tick)
        return belief

    def top_suspects(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k most-likely attacker locations."""
        return top_k_nodes(self.belief, k)

    def time_to_detect(self, true_node: str, threshold: float = 0.5) -> Optional[int]:
        """Return tick at which belief on the true node exceeded threshold."""
        if self.belief.get(true_node, 0) >= threshold and self._detected_at is None:
            self._detected_at = self.tick
        return self._detected_at

    def on_reset(self) -> None:
        super().on_reset()
        self._detected_at = None
        self._true_location_history = []

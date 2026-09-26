"""
environment/network_graph.py
NetworkX-backed enterprise network graph with full node/edge state management.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import yaml


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class NodeState(Enum):
    """Observable state of a network node from Blue's perspective."""
    SAFE = "safe"
    SUSPECTED = "suspected"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"
    PATCHING = "patching"
    HONEYPOT = "honeypot"


@dataclass
class NodeData:
    """All attributes attached to a node in the network graph.

    Args:
        name: Unique node identifier (matches the graph node key).
        node_type: One of 'workstation', 'server', 'database', 'crown_jewel'.
        value: Asset value used in the utility/performance metric.
        vuln_score: Probability [0, 1] that an exploit attempt succeeds.
        patched: Whether a patch has been applied (lowers effective vuln_score).
        state: Current observable state of the node.
        services: List of running service names (informational).
    """
    name: str
    node_type: str
    value: int
    vuln_score: float
    patched: bool = False
    state: NodeState = NodeState.SAFE
    services: List[str] = field(default_factory=list)

    @property
    def effective_vuln(self) -> float:
        """Effective vulnerability after patching."""
        return self.vuln_score * (0.1 if self.patched else 1.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "value": self.value,
            "vuln_score": self.vuln_score,
            "effective_vuln": self.effective_vuln,
            "patched": self.patched,
            "state": self.state.value,
            "services": self.services,
        }


# ---------------------------------------------------------------------------
# NetworkGraph
# ---------------------------------------------------------------------------

class NetworkGraph:
    """Wraps a NetworkX DiGraph representing the enterprise network.

    Nodes carry :class:`NodeData`; edges carry a ``cost`` (traversal risk/
    difficulty for Red) and an ``bandwidth`` attribute.

    Loading from YAML::

        graph = NetworkGraph("config/network_small.yaml")

    YAML format::

        nodes:
          - id: ws1
            type: workstation
            value: 2
            vuln_score: 0.6
            services: [ssh]
        edges:
          - from: ws1
            to: srv1
            cost: 3
    """

    def __init__(self, config_path: Optional[str] = None):
        self.g: nx.DiGraph = nx.DiGraph()
        self._crown_jewel: Optional[str] = None
        self._entry_points: List[str] = []
        if config_path:
            self._load_from_yaml(config_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_from_yaml(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for node_cfg in cfg.get("nodes", []):
            nid = node_cfg["id"]
            data = NodeData(
                name=nid,
                node_type=node_cfg.get("type", "workstation"),
                value=node_cfg.get("value", 1),
                vuln_score=node_cfg.get("vuln_score", 0.5),
                services=node_cfg.get("services", []),
            )
            self.g.add_node(nid, data=data)
            if node_cfg.get("type") == "crown_jewel":
                self._crown_jewel = nid
            if node_cfg.get("entry_point", False):
                self._entry_points.append(nid)

        for edge_cfg in cfg.get("edges", []):
            self.g.add_edge(
                edge_cfg["from"],
                edge_cfg["to"],
                cost=edge_cfg.get("cost", 1),
                bandwidth=edge_cfg.get("bandwidth", 100),
            )
            # Undirected traversal: add reverse edge with same cost if not directed
            if not edge_cfg.get("directed", False):
                self.g.add_edge(
                    edge_cfg["to"],
                    edge_cfg["from"],
                    cost=edge_cfg.get("cost", 1),
                    bandwidth=edge_cfg.get("bandwidth", 100),
                )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def crown_jewel(self) -> str:
        if self._crown_jewel is None:
            raise ValueError("No crown_jewel node defined in the network config.")
        return self._crown_jewel

    @property
    def entry_points(self) -> List[str]:
        if not self._entry_points:
            # Fall back to first node
            return [next(iter(self.g.nodes()))]
        return self._entry_points

    def node_data(self, node: str) -> NodeData:
        return self.g.nodes[node]["data"]

    def neighbors_with_cost(self, node: str) -> List[Tuple[str, float]]:
        """Return list of (neighbor, traversal_cost) for Red's A* search."""
        result = []
        for neighbor in self.g.successors(node):
            nd = self.node_data(neighbor)
            # Isolated and honeypot nodes block traversal
            if nd.state in (NodeState.ISOLATED,):
                continue
            cost = self.g[node][neighbor]["cost"]
            # Increase cost through suspected/high-security nodes
            if nd.state == NodeState.SUSPECTED:
                cost *= 1.5
            result.append((neighbor, cost))
        return result

    def all_nodes(self) -> List[str]:
        return list(self.g.nodes())

    def traversable_nodes(self) -> List[str]:
        """Nodes that are not isolated."""
        return [n for n in self.g.nodes()
                if self.node_data(n).state != NodeState.ISOLATED]

    def crown_jewel_compromised(self) -> bool:
        return self.node_data(self.crown_jewel).state == NodeState.COMPROMISED

    def pending_jobs(self) -> List[NodeData]:
        """Return nodes that need patching (not yet patched, not isolated)."""
        return [
            self.node_data(n) for n in self.g.nodes()
            if not self.node_data(n).patched
            and self.node_data(n).state not in (NodeState.ISOLATED, NodeState.PATCHING)
            and self.node_data(n).vuln_score > 0.3
        ]

    def uptime_fraction(self) -> float:
        """Fraction of non-isolated, non-compromised nodes (service uptime proxy)."""
        total = len(self.g.nodes())
        if total == 0:
            return 1.0
        down = sum(
            1 for n in self.g.nodes()
            if self.node_data(n).state in (NodeState.ISOLATED, NodeState.COMPROMISED)
        )
        return (total - down) / total

    def compromised_value(self) -> int:
        """Total asset value of compromised nodes."""
        return sum(
            self.node_data(n).value
            for n in self.g.nodes()
            if self.node_data(n).state == NodeState.COMPROMISED
        )

    # ------------------------------------------------------------------
    # Action primitives (called by Blue agents and the simulator)
    # ------------------------------------------------------------------

    def isolate(self, node: str) -> bool:
        """Isolate a node — Red can no longer traverse through it."""
        nd = self.node_data(node)
        if nd.state in (NodeState.ISOLATED, NodeState.PATCHING):
            return False
        nd.state = NodeState.ISOLATED
        return True

    def patch(self, node: str) -> bool:
        """Begin patching a node — sets state to PATCHING then SAFE on completion."""
        nd = self.node_data(node)
        if nd.patched or nd.state == NodeState.PATCHING:
            return False
        nd.state = NodeState.PATCHING
        return True

    def complete_patch(self, node: str) -> None:
        """Called after patching is done — marks node as patched and SAFE."""
        nd = self.node_data(node)
        nd.patched = True
        nd.state = NodeState.SAFE

    def restore(self, node: str) -> bool:
        """Restore an isolated or compromised node from backup."""
        nd = self.node_data(node)
        if nd.state in (NodeState.COMPROMISED, NodeState.ISOLATED):
            nd.state = NodeState.SAFE
            nd.patched = False  # Reset patch status — needs re-patching
            return True
        return False

    def deploy_honeypot(self, node: str) -> bool:
        """Deploy a honeypot on a node to attract and detect Red."""
        nd = self.node_data(node)
        if nd.state == NodeState.SAFE:
            nd.state = NodeState.HONEYPOT
            return True
        return False

    def mark_suspected(self, node: str) -> None:
        nd = self.node_data(node)
        if nd.state == NodeState.SAFE:
            nd.state = NodeState.SUSPECTED

    def mark_compromised(self, node: str) -> None:
        nd = self.node_data(node)
        nd.state = NodeState.COMPROMISED

    def mark_safe(self, node: str) -> None:
        nd = self.node_data(node)
        if nd.state == NodeState.SUSPECTED:
            nd.state = NodeState.SAFE

    def apply_action(self, action_type: str, node: str) -> bool:
        """Dispatch a string action type to the appropriate method."""
        dispatch = {
            "isolate": self.isolate,
            "patch": self.patch,
            "restore": self.restore,
            "deploy_honeypot": self.deploy_honeypot,
        }
        fn = dispatch.get(action_type)
        if fn:
            return fn(node)
        return False

    # ------------------------------------------------------------------
    # Graph generation helpers (for medium/large random topologies)
    # ------------------------------------------------------------------

    @classmethod
    def generate_random(
        cls,
        n_nodes: int,
        n_workstations: int = None,
        n_servers: int = None,
        n_databases: int = None,
        seed: int = 42,
    ) -> "NetworkGraph":
        """Generate a random enterprise-like network topology.

        Args:
            n_nodes: Total number of nodes.
            n_workstations: Defaults to 60% of nodes.
            n_servers: Defaults to 25% of nodes.
            n_databases: Defaults to 14% of nodes (1 crown_jewel always added).
            seed: RNG seed for reproducibility.
        """
        rng = random.Random(seed)
        ng = cls()

        # Allocate node types
        n_ws = n_workstations or int(n_nodes * 0.60)
        n_srv = n_servers or int(n_nodes * 0.25)
        n_db = n_databases or max(1, n_nodes - n_ws - n_srv - 1)

        type_list: List[Tuple[str, str, int, float]] = []
        for i in range(n_ws):
            type_list.append((f"ws{i}", "workstation", rng.randint(1, 3), rng.uniform(0.3, 0.8)))
        for i in range(n_srv):
            type_list.append((f"srv{i}", "server", rng.randint(3, 7), rng.uniform(0.2, 0.6)))
        for i in range(n_db):
            type_list.append((f"db{i}", "database", rng.randint(5, 10), rng.uniform(0.1, 0.4)))
        type_list.append(("crown_jewel", "crown_jewel", 20, 0.05))

        rng.shuffle(type_list)
        node_ids = []
        for nid, ntype, nval, nvuln in type_list:
            data = NodeData(name=nid, node_type=ntype, value=nval, vuln_score=nvuln)
            ng.g.add_node(nid, data=data)
            node_ids.append(nid)
            if ntype == "crown_jewel":
                ng._crown_jewel = nid

        # Create a connected Barabási–Albert-like topology
        # First connect as a chain, then add random shortcuts
        for i in range(1, len(node_ids)):
            cost = rng.randint(1, 5)
            ng.g.add_edge(node_ids[i - 1], node_ids[i], cost=cost, bandwidth=100)
            ng.g.add_edge(node_ids[i], node_ids[i - 1], cost=cost, bandwidth=100)

        extra_edges = n_nodes // 2
        for _ in range(extra_edges):
            u, v = rng.sample(node_ids, 2)
            if not ng.g.has_edge(u, v):
                cost = rng.randint(1, 5)
                ng.g.add_edge(u, v, cost=cost, bandwidth=100)
                ng.g.add_edge(v, u, cost=cost, bandwidth=100)

        # Entry points: workstations on the periphery
        ng._entry_points = [nid for nid, ntype, _, _ in type_list
                            if ntype == "workstation"][:max(1, n_ws // 5)]
        return ng

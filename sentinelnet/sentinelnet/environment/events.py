"""
environment/events.py
Dataclasses for simulation events that flow through the message bus.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertEvent:
    """An IDS alert emitted when Red moves through a node.

    Attributes:
        tick: Simulation tick at which the alert was generated.
        node: The node that triggered the alert (may be a false positive
              if ``is_false_positive`` is True).
        confidence: Probability that this alert is genuine [0, 1].
        is_false_positive: Ground-truth flag (hidden from Blue agents).
    """
    tick: int
    node: str
    confidence: float = 1.0
    is_false_positive: bool = False


@dataclass
class ExploitEvent:
    """Records a Red exploit attempt on a node.

    Attributes:
        tick: Simulation tick.
        attacker_node: Node from which Red launched the exploit.
        target_node: Node Red attempted to compromise.
        success: Whether the exploit succeeded (based on vuln_score).
    """
    tick: int
    attacker_node: str
    target_node: str
    success: bool


@dataclass
class PatchEvent:
    """Records a Blue patch action on a node.

    Attributes:
        tick: Simulation tick.
        node: The node being patched.
        scheduled_by: Which agent scheduled the patch ('response' or 'patch_scheduler').
    """
    tick: int
    node: str
    scheduled_by: str = "patch_scheduler"


@dataclass
class ActionEvent:
    """A generic Blue action applied to the graph.

    Attributes:
        tick: Simulation tick.
        action_type: One of 'isolate', 'patch', 'restore', 'deploy_honeypot', 'reroute', 'alert'.
        node: Target node of the action.
        agent: Which Blue agent issued this action.
        cost: Resource cost of performing this action.
    """
    tick: int
    action_type: str
    node: str
    agent: str = "response"
    cost: int = 1

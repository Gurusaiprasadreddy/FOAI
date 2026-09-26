"""
viz/dashboard.py
SentinelNet — Live Defense Simulation Dashboard (Streamlit + Pyvis)

Layout:
  Left column  (60%): Live network graph with colour-coded node states
  Right column (40%): Belief bar chart, running metrics, action log

Controls:
  - Scenario selector
  - Topology selector (small / medium / large)
  - Play / Pause / Step buttons + speed slider
  - Reset button

Colour coding:
  safe        → #00C896  (green)
  suspected   → #F5A623  (amber)
  compromised → #E74C3C  (red)
  isolated    → #95A5A6  (grey)
  patching    → #3498DB  (blue)
  honeypot    → #9B59B6  (purple)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import streamlit as st
import yaml

# Make sure the package root is importable whether run from repo root or subdir
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sentinelnet.environment.network_graph import NetworkGraph, NodeState
from sentinelnet.environment.simulator import Simulator, build_simulator
from sentinelnet.coordination.message_bus import MessageBus

# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

STATE_COLORS = {
    NodeState.SAFE.value: "#00C896",
    NodeState.SUSPECTED.value: "#F5A623",
    NodeState.COMPROMISED.value: "#E74C3C",
    NodeState.ISOLATED.value: "#95A5A6",
    NodeState.PATCHING.value: "#3498DB",
    NodeState.HONEYPOT.value: "#9B59B6",
}

TYPE_SHAPES = {
    "workstation": "dot",
    "server": "square",
    "database": "diamond",
    "crown_jewel": "star",
}

# ---------------------------------------------------------------------------
# Build Pyvis HTML
# ---------------------------------------------------------------------------

def build_pyvis_html(graph: NetworkGraph, red_node: str, belief: Dict[str, float]) -> str:
    """Render the network graph as a Pyvis HTML string."""
    from pyvis.network import Network

    net = Network(height="520px", width="100%", bgcolor="#0d1117", font_color="white")
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 11, "face": "Inter, sans-serif"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": {"color": "#444466"},
        "smooth": {"type": "continuous"},
        "width": 1.5
      },
      "physics": {
        "stabilization": {"iterations": 100},
        "barnesHut": {"gravitationalConstant": -3000, "centralGravity": 0.3, "springLength": 120}
      },
      "interaction": {"tooltipDelay": 100, "hover": true}
    }
    """)

    for node_id in graph.all_nodes():
        nd = graph.node_data(node_id)
        color = STATE_COLORS.get(nd.state.value, "#888888")
        shape = TYPE_SHAPES.get(nd.node_type, "dot")

        # Belief probability for tooltip
        belief_pct = f"{belief.get(node_id, 0) * 100:.1f}%" if belief else "–"

        # Red attacker position indicator
        is_red = (node_id == red_node)
        border_color = "#FF0000" if is_red else "#222244"
        border_width = 5 if is_red else 2
        size = 30 if nd.node_type == "crown_jewel" else (22 if is_red else 18)

        tooltip = (
            f"<b>{node_id}</b><br>"
            f"Type: {nd.node_type}<br>"
            f"State: {nd.state.value}<br>"
            f"Vuln: {nd.vuln_score:.2f} | Effective: {nd.effective_vuln:.2f}<br>"
            f"Value: {nd.value} | Patched: {nd.patched}<br>"
            f"Belief P(Red here): {belief_pct}"
        )

        net.add_node(
            node_id,
            label=node_id,
            color={"background": color, "border": border_color},
            shape=shape,
            size=size,
            borderWidth=border_width,
            title=tooltip,
        )

    for u, v, data in graph.g.edges(data=True):
        net.add_edge(u, v, label=str(data.get("cost", "")), title=f"Cost: {data.get('cost', 1)}")

    # Generate HTML and return
    html = net.generate_html()
    return html


# ---------------------------------------------------------------------------
# Streamlit page layout
# ---------------------------------------------------------------------------

def render_legend():
    st.markdown("""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;font-size:12px">
      <span>🟢 Safe</span>
      <span>🟡 Suspected</span>
      <span>🔴 Compromised</span>
      <span>⚪ Isolated</span>
      <span>🔵 Patching</span>
      <span>🟣 Honeypot</span>
      <span style="border:2px solid red;padding:1px 4px">🎯 Red</span>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(tick, uptime, compromised_count, ttd, belief):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tick", tick)
    col2.metric("Uptime", f"{uptime*100:.1f}%")
    col3.metric("Compromised Nodes", compromised_count)
    col4.metric("TTD", f"{ttd} ticks" if ttd else "–")


def render_belief_chart(belief: Dict[str, float], top_k: int = 8):
    if not belief:
        return
    import pandas as pd
    top = sorted(belief.items(), key=lambda x: -x[1])[:top_k]
    df = pd.DataFrame(top, columns=["Node", "P(Red here)"])
    st.bar_chart(df.set_index("Node"))


def render_results_table(results: List[dict]):
    import pandas as pd
    if not results:
        return
    df = pd.DataFrame(results)
    if not df.empty:
        display_cols = [
            "scenario", "strategy", "total_ticks",
            "crown_jewel_compromised", "max_compromised_nodes",
            "mean_ttd_ticks", "mean_uptime_pct", "avg_nodes_expanded_astar",
            "total_wall_ms",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="SentinelNet — Network Defense Simulator",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0d1117; }
    .block-container { padding-top: 1rem; }
    h1 { color: #00C896; letter-spacing: -0.5px; }
    h2, h3 { color: #E2E8F0; }
    .stMetric > div { background: #161b22; border-radius: 8px; padding: 8px; }
    .stMetricLabel { color: #8B949E !important; font-size: 12px !important; }
    .stMetricValue { color: #F0F6FC !important; font-weight: 700 !important; }
    .log-box { background: #161b22; border-radius: 8px; padding: 8px; font-size: 11px;
                font-family: monospace; color: #8B949E; height: 180px; overflow-y: auto; }
    .status-banner { border-radius: 8px; padding: 12px 16px; margin: 8px 0;
                     font-weight: 600; font-size: 14px; }
    .status-ok { background: #1a3a2a; color: #00C896; border-left: 4px solid #00C896; }
    .status-warn { background: #3a2a1a; color: #F5A623; border-left: 4px solid #F5A623; }
    .status-danger { background: #3a1a1a; color: #E74C3C; border-left: 4px solid #E74C3C; }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("# 🛡️ SentinelNet")
    st.markdown("**Multi-Agent Adversarial Network Defense Simulation** — Red vs Blue")

    # --- Sidebar config ---
    st.sidebar.markdown("## ⚙️ Configuration")

    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    scenarios_path = os.path.join(config_dir, "scenarios.yaml")

    scenarios = {}
    if os.path.exists(scenarios_path):
        with open(scenarios_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for s in data.get("scenarios", []):
            scenarios[s["name"]] = s

    if not scenarios:
        st.error(f"No scenarios found at {scenarios_path}. Run from the repo root.")
        return

    scenario_name = st.sidebar.selectbox("Scenario", list(scenarios.keys()), index=0)
    speed = st.sidebar.slider("Speed (ticks/sec)", 0.5, 10.0, 2.0, 0.5)
    show_red = st.sidebar.checkbox("Show Red's position (debug)", value=True)
    top_k = st.sidebar.slider("Belief chart top-K nodes", 3, 10, 6)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Defender")
    defender_override = st.sidebar.selectbox(
        "Override defender", ["(use scenario)", "random", "greedy", "minimax"], index=0
    )

    # --- State management ---
    if "sim" not in st.session_state:
        st.session_state.sim = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "action_log" not in st.session_state:
        st.session_state.action_log = []
    if "results" not in st.session_state:
        st.session_state.results = []
    if "last_belief" not in st.session_state:
        st.session_state.last_belief = {}
    if "ttd" not in st.session_state:
        st.session_state.ttd = None

    # --- Control buttons ---
    col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 1])

    with col_b1:
        if st.button("▶ Run", use_container_width=True, type="primary"):
            cfg = dict(scenarios[scenario_name])
            if defender_override != "(use scenario)":
                cfg["defender"] = defender_override
            sim = build_simulator(cfg, scenarios_path)
            st.session_state.sim = sim
            st.session_state.running = True
            st.session_state.action_log = []
            st.session_state.last_belief = {}
            st.session_state.ttd = None

    with col_b2:
        if st.button("⏸ Pause/Resume", use_container_width=True):
            st.session_state.running = not st.session_state.running

    with col_b3:
        if st.button("⏭ Step", use_container_width=True):
            if st.session_state.sim and not st.session_state.sim.terminated:
                st.session_state.running = False
                info = st.session_state.sim.step()
                _process_tick_info(info, st.session_state)

    with col_b4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim = None
            st.session_state.running = False
            st.session_state.action_log = []
            st.session_state.results = []
            st.session_state.last_belief = {}
            st.session_state.ttd = None

    # --- Layout ---
    graph_col, info_col = st.columns([3, 2])

    graph_placeholder = graph_col.empty()
    legend_placeholder = graph_col.empty()
    status_placeholder = graph_col.empty()

    metrics_placeholder = info_col.empty()
    belief_placeholder = info_col.empty()
    log_placeholder = info_col.empty()

    # --- Initial static state ---
    if st.session_state.sim is None:
        cfg = dict(scenarios[scenario_name])
        sim_preview = build_simulator(cfg, scenarios_path)
        html = build_pyvis_html(sim_preview.graph, "", {})
        with graph_placeholder.container():
            legend_placeholder.markdown("")
            render_legend()
            st.components.v1.html(html, height=540, scrolling=False)
        return

    sim: Simulator = st.session_state.sim

    # --- Auto-run loop ---
    if st.session_state.running and not sim.terminated:
        info = sim.step()
        _process_tick_info(info, st.session_state)
        time.sleep(1.0 / speed)

    # --- Render current state ---
    red_node = sim.red.current_node if show_red else ""
    html = build_pyvis_html(sim.graph, red_node, st.session_state.last_belief)

    with graph_placeholder.container():
        render_legend()
        st.components.v1.html(html, height=540, scrolling=False)

    # Status banner
    cj_comp = sim.graph.crown_jewel_compromised()
    if sim.terminated:
        reason = sim.termination_reason
        if "Blue defended" in reason:
            cls = "status-ok"
        elif "Crown jewel" in reason or "Red reached" in reason:
            cls = "status-danger"
        else:
            cls = "status-warn"
        status_placeholder.markdown(
            f'<div class="status-banner {cls}">🏁 {reason}</div>',
            unsafe_allow_html=True,
        )

    # Metrics panel
    with metrics_placeholder.container():
        st.markdown("### 📊 Live Metrics")
        render_metrics(
            tick=sim.current_tick,
            uptime=sim.graph.uptime_fraction(),
            compromised_count=sum(
                1 for n in sim.graph.all_nodes()
                if sim.graph.node_data(n).state.value == "compromised"
            ),
            ttd=st.session_state.ttd,
            belief=st.session_state.last_belief,
        )

    # Belief chart
    with belief_placeholder.container():
        st.markdown("#### 🔍 Belief Distribution (Top Suspects)")
        render_belief_chart(st.session_state.last_belief, top_k=top_k)

    # Action log
    with log_placeholder.container():
        st.markdown("#### 📋 Action Log")
        log_html = "<br>".join(
            st.session_state.action_log[-20:][::-1]
        ) or "<i>No actions yet</i>"
        st.markdown(
            f'<div class="log-box">{log_html}</div>',
            unsafe_allow_html=True,
        )

    # Results table (shown when terminated)
    if sim.terminated:
        summary = sim.logger.summary()
        existing = [r for r in st.session_state.results
                    if r.get("scenario") != summary.get("scenario") or
                       r.get("strategy") != summary.get("strategy")]
        st.session_state.results = existing + [summary]

        st.markdown("---")
        st.markdown("## 📈 Results Table")
        render_results_table(st.session_state.results)

        csv_path = os.path.join(config_dir, "..", "results.csv")
        sim.logger.to_csv(csv_path)
        st.success(f"Results saved to `{os.path.abspath(csv_path)}`")

    # Auto-rerun to animate
    if st.session_state.running and not sim.terminated:
        st.rerun()


def _process_tick_info(info: dict, state) -> None:
    """Update session state from a tick info dict."""
    belief = info.get("belief", {})
    if belief:
        state.last_belief = belief

    action = info.get("action")
    alert = info.get("alert")

    tick = info.get("tick", "?")
    parts = [f"<b>T{tick}</b>"]
    if alert:
        parts.append(f"🚨 Alert@<b>{alert.node}</b>")
    if action:
        parts.append(
            f"🛡 {action.action_type.upper()}({action.node})"
        )
    if info.get("red_captured"):
        parts.append("💥 Red captured!")
    if info.get("crown_jewel_compromised"):
        parts.append("🔴 CROWN JEWEL COMPROMISED")

    if len(parts) > 1:
        state.action_log.append(" | ".join(parts))

    # TTD estimate: if belief > 40% on current Red node
    belief = info.get("belief", {})
    red_node = info.get("red_node", "")
    if belief and red_node and belief.get(red_node, 0) > 0.4 and state.ttd is None:
        state.ttd = tick


if __name__ == "__main__":
    main()

# SentinelNet — Multi-Agent Adversarial Network Defense Planner

A simulated enterprise network under attack. A **Red agent** navigates a graph of servers, workstations, and databases toward a crown jewel node using A\*. Three **Blue agents** — Monitor, Response, and Patch Scheduler — cooperate via a message bus to detect, contain, and recover.

## Quick Start

```bash
cd sentinelnet
pip install -r requirements.txt
pytest tests/ -v
streamlit run sentinelnet/viz/dashboard.py
```

## Project Structure

```
sentinelnet/
├── config/
│   ├── network_small.yaml       # 20-node demo topology
│   ├── network_medium.yaml      # 100-node topology
│   ├── network_large.yaml       # 500-node scalability test
│   └── scenarios.yaml           # 5 demo scenario definitions
├── sentinelnet/
│   ├── environment/
│   │   ├── network_graph.py     # NetworkX wrapper, node/edge state
│   │   ├── events.py            # Alert, ExploitEvent, PatchEvent dataclasses
│   │   └── simulator.py         # Tick loop, scenario runner
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── red_attacker.py      # A* pathing
│   │   ├── monitor.py           # Belief tracking (Bayesian / particle filter)
│   │   ├── response.py          # Minimax / expectimax + alpha-beta
│   │   └── patch_scheduler.py   # CSP: backtracking + MRV + forward checking
│   ├── coordination/
│   │   └── message_bus.py       # Pub/sub blackboard for Blue agents
│   ├── baselines/
│   │   ├── random_defender.py
│   │   └── greedy_defender.py
│   ├── metrics/
│   │   └── logger.py
│   └── viz/
│       └── dashboard.py         # Streamlit app
└── tests/
    ├── test_astar.py
    ├── test_belief_tracking.py
    ├── test_minimax.py
    ├── test_csp_scheduler.py
    └── test_integration_scenarios.py
```

## Algorithms

| Agent | Algorithm | Justification |
|-------|-----------|--------------|
| Red Attacker | A* with hop-count heuristic | Optimal path planning over weighted attack graph |
| Monitor | Bayesian belief update + diffusion | Tracks hidden attacker from noisy IDS alerts |
| Response | Minimax → Expectimax + alpha-beta | Adversarial decisions; chance nodes for stochastic exploits |
| Patch Scheduler | CSP + backtracking + MRV + FC | Maintenance windows, dependency ordering, technician limits |

## Demo Scenarios

1. **Baseline** — naive Red vs random Blue defender
2. **Stealthy attacker** — low detection rate; belief tracking essential
3. **Multi-entry attack** — two simultaneous Red entry points
4. **Resource-starved defense** — Blue gets one action per 2 ticks
5. **Scalability run** — 20 / 100 / 500 nodes; nodes expanded + wall-clock time

## Environment Properties

- **Partially observable**: attacker position hidden; only noisy alerts visible
- **Stochastic**: exploit success is probabilistic
- **Sequential**: actions have lasting effects across ticks
- **Dynamic**: network state changes each tick
- **Discrete**: finite nodes, edges, and action set
- **Multi-agent**: competitive Red vs Blue, cooperative within Blue

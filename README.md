# SentinelNet: Multi-Agent Adversarial Network Defense Planner

SentinelNet is a multi-agent adversarial network defense simulator built with standard Python libraries. A simulated enterprise network (graph of servers, databases, and workstations) is attacked by a **Red agent** trying to reach a "Crown Jewel" node. A team of **Blue defender agents** detects, isolates, patches, and recovers network assets. Everything operates in an abstract simulation environment without real-world exploits or scanning.

---

## 🧠 Project Explanation

### The Big Picture
Imagine your company's computer network as a map of connected rooms. Each room is a server, workstation, database, or a "Crown Jewel" (most valuable asset). A hacker (Red agent) is trying to sneak through the rooms and reach the Crown Jewel. Your security team (Blue agents) is trying to stop them.

SentinelNet simulates this battle, tick by tick, in an abstract way — no real exploits, no real scanning, just pure logic and state-of-the-art algorithms.

---

### The Network (Environment)
The network is represented as a directed graph (flowchart of connected nodes):

```
Perimeter Workstations
   ws0 ─────────────────────────────┐
   ws1 ──→ DMZ Servers ──→ Internal Servers ──→ Databases ──→ [CROWN JEWEL]
   ws2       srv_dmz0      srv_int0     db0
             srv_dmz1      srv_int1     db1
                           srv_int2     db2
                              │
                         [HONEYPOT] ──→ [CROWN JEWEL]   ← tempting shortcut!
```

Each node contains:
- `vuln_score`: How easy it is for Red to compromise (0.0 = impossible, 1.0 = trivial)
- `value`: Severity of compromise (Crown Jewel = 20, Workstation = 1)
- `state`: `SAFE` | `SUSPECTED` | `COMPROMISED` | `ISOLATED` | `PATCHING` | `HONEYPOT`

---

### 🛡️ Agents & Algorithms Overview

#### 1. The Red Agent (Attacker)
* **Algorithm**: **A* Search with BFS Heuristic**
* **Role**: Red starts at a perimeter workstation and seeks the shortest/cheapest path to the Crown Jewel.
* **Mechanism**:
  - Costs: `g(n)` = total edge cost from entry to node `n`
  - Heuristic: `h(n)` = estimated remaining hops to Crown Jewel (BFS-based, admissible)
  - Evaluation: `f(n) = g(n) + h(n)`
  - Dynamic Replanning: If Blue isolates a node on Red's route, Red automatically replans using A* from its current position.
  - Noisy Alerts: Triggers IDS alerts with probability `detection_rate` on exploit attempt, and `false_alarm_rate` on random clean nodes.

#### 2. Blue Agent 1 — Monitor (Belief Tracking)
* **Algorithm**: **Bayesian Belief Update + Spatial Diffusion**
* **Role**: The Monitor never directly sees Red's position; it only observes noisy IDS alerts. It maintains a probability distribution over all nodes ("how likely is Red to be at each node right now?").
* **Mechanism**:
  - **Bayesian Update**: When an alert fires at node $X$:
    $$\begin{aligned} P(\text{Red at } X \mid \text{alert at } X) &\propto \text{detection\_rate} \times \text{prior} \\ P(\text{Red at } Y \mid \text{alert at } X) &\propto \text{false\_alarm\_rate} \times \text{prior} \end{aligned}$$
    (Normalized so probabilities across all nodes sum to 1.0)
  - **Spatial Diffusion**: Models Red's movement between ticks:
    - $P(\text{Red stays}) = 0.6$
    - $P(\text{Red moves to neighbor}) = 0.4 / N_{\text{neighbors}}$
  - **Output**: Publishes a live "heat map" of suspicion to the Message Bus.

#### 3. Blue Agent 2 — Response (Defensive Actions)
* **Algorithm**: **Minimax with Alpha-Beta Pruning → Expectimax**
* **Role**: Reads the Monitor's belief map and evaluates defensive actions (ISOLATE, PATCH, RESTORE).
* **Mechanism**:
  - **Minimax**: Models the scenario as a 2-player zero-sum game with Alpha-Beta pruning to prune subtrees that cannot yield better outcomes.
  - **Expectimax**: Replaces deterministic adversary nodes with chance nodes, taking expectations over Red's exploit success probabilities:
    $$\mathbb{E}[\text{Value}] = \text{vuln\_score} \times \text{Value}(\text{success}) + (1 - \text{vuln\_score}) \times \text{Value}(\text{failure})$$
  - **Utility Function**:
    $$\text{Utility} = 10 \times \text{uptime\_fraction} - 5 \times \text{compromised\_value} - 0.5 \times \text{response\_cost}$$

#### 4. Blue Agent 3 — Patch Scheduler (CSP)
* **Algorithm**: **Constraint Satisfaction Problem (CSP) with Backtracking + MRV + Forward Checking**
* **Role**: Manages long-term maintenance scheduling across vulnerable nodes while minimizing disruption.
* **Mechanism**:
  - **Variables**: Pending patch jobs per vulnerable node.
  - **Domains**: Maintenance time slots.
  - **Constraints**: Maintenance window bounds, dependency ordering ($\text{slot}(A) < \text{slot}(B)$), and resource exclusivity (AllDifferent technician assignment).
  - **Optimization**: **MRV (Minimum Remaining Values)** heuristic selects constrained variables first; **Forward Checking** prunes domain spaces to prevent deep backtracking dead-ends.

---

### 💬 The Message Bus (Multi-Agent Blackboard)

The Blue team agents coordinate asynchronously over a pub/sub blackboard architecture:

```
Monitor  ──[belief]──→  Message Bus  ──[belief]──→  Response
                             ↑                          │
                        [alerts]                   [actions]
                             │                          ↓
                          Red ────────────────→  Graph (Environment)
                                                        │
Scheduler ←────[pending_jobs]──────────────────────────┘
    │
    └──[patch_plan]──→ Message Bus
```

---

### 🔄 The Simulation Loop (One Tick)

1. **`Red.step()`** → Moves via A*, attempts exploit, generates IDS alerts.
2. **`bus.publish('alerts', alert)`** → Emits detection events.
3. **`Monitor.step()`** → Runs Bayesian update + diffusion, publishes belief state.
4. **`Response.step()`** → Evaluates minimax/expectimax game tree, dispatches defensive move.
5. **`Scheduler.step()`** → Executes CSP patch assignments.
6. **`Logger.record()`** → Records metrics (TTD, uptime, compromise state).
7. **Check Termination** → Crown jewel compromised, Red captured, or max ticks reached.

---

### 🖥️ Dashboard Layout

```
Left (60%)              Right (40%)
┌─────────────────┐    ┌─────────────────────────┐
│  Live Network   │    │  📊 Tick | Uptime | ...  │
│  Graph (Pyvis)  │    ├─────────────────────────┤
│                 │    │  🔍 Belief Bar Chart     │
│  🟢 safe        │    │  (top-6 suspect nodes)   │
│  🟡 suspected   │    ├─────────────────────────┤
│  🔴 compromised │    │  📋 Action Log           │
│  ⚪ isolated    │    │  T5 | Alert@ws2 |        │
│  🔵 patching    │    │  ISOLATE(srv_int0)       │
│  🟣 honeypot    │    └─────────────────────────┘
│  🎯 Red here    │
└─────────────────┘
[▶ Run] [⏸ Pause] [⏭ Step] [🔄 Reset]   Speed: ────●──── 
```

---

### 🧪 Scenarios

| # | Scenario | What to Observe |
|---|---|---|
| 1 | **Baseline**: Naive Red vs Random Defender | Crown jewel compromised rapidly (baseline benchmark) |
| 2 | **Stealthy Attacker** (`detection_rate=0.25`) | Bayesian belief tracking concentrates on stealthy Red |
| 3 | **Multi-Entry Attack** (Two Red entries) | Response agent prioritizes high-value branches |
| 4 | **Resource-Starved Defense** (1 action / 2 ticks) | CSP scheduling becomes critical to maintain uptime |
| 5 | **Scalability Test** (20 → 100 → 500 nodes) | Graph scaling performance and search efficiency |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
cd sentinelnet
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
cd sentinelnet
python -m pytest tests/ -v
```

### 3. Launch the Live Streamlit Dashboard
```bash
cd sentinelnet
python -m streamlit run sentinelnet/viz/dashboard.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to view the interactive simulation.

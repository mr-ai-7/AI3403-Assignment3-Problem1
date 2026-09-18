# AI3403 MAS Assignment #3 — Problem 1

## Name Formation: AJAY

This repository contains the solution for Problem 1 of AI3403 Multi-Agent Systems Assignment #3.

### Setup
- Number of agents: **N = 20**
- Graph model: **Erdos-Renyi G(N,p)**
- Probability: **p = 0.18**
- The graph is regenerated until it is connected.
- Initial agent positions are random points in R^2.
- Formation-control sequence: **A → J → A → Y**

### Formation-control law

For agent i,

u_i = -K sum_j a_ij [ (x_i - x_j) - (q_i - q_j) ]

where:
- x_i is the current position of agent i,
- q_i is the desired position assigned to agent i,
- a_ij is the adjacency-matrix entry,
- K is the control gain.

The controller uses only relative information from graph neighbours. For a connected graph, the relative formation converges to the desired formation.

### Run

```bash
pip install -r requirements.txt
python formation_control_AJAY.py
```

The script generates:

`AJAY_formation_control.mp4`

### Reproducibility

A fixed random seed (`3403`) is used, so the same connected graph and initial positions are obtained when the script is run again.


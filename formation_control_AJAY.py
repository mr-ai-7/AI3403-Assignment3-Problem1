import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.optimize import linear_sum_assignment

# AI3403 MAS Assignment #3 - Problem 1
# Name formation: AJAY
# N = 20 agents, connected Erdos-Renyi graph, formation control

SEED = 3403
N = 20
P = 0.18
K = 2.2
DT = 0.08
STEPS_PER_LETTER = 90
HOLD_STEPS = 15

rng = np.random.default_rng(SEED)

# 1. Generate a connected Erdos-Renyi graph G(N,p)

while True:
    G = nx.erdos_renyi_graph(N, P, seed=int(rng.integers(1_000_000_000)))
    if nx.is_connected(G):
        break

A = nx.to_numpy_array(G, dtype=float)

# Random initial positions in R^2
X = rng.uniform(-5, 5, size=(N, 2))

def polyline_sample(points, n):
    pts = np.asarray(points, dtype=float)
    seg = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total = cumulative[-1]

    if total == 0:
        return np.repeat(pts[:1], n, axis=0)

    s = np.linspace(0, total, n)
    result = []

    for value in s:
        idx = np.searchsorted(cumulative, value, side="right") - 1
        idx = min(max(idx, 0), len(lengths) - 1)

        if lengths[idx] > 0:
            alpha = (value - cumulative[idx]) / lengths[idx]
        else:
            alpha = 0.0

        result.append(pts[idx] + alpha * seg[idx])

    return np.asarray(result)

def sample_segments(segments, n):
    lengths = np.array([
        np.linalg.norm(np.asarray(b) - np.asarray(a))
        for a, b in segments
    ])

    counts = np.floor(n * lengths / lengths.sum()).astype(int)
    counts = np.maximum(counts, 1)

    while counts.sum() > n:
        i = np.argmax(counts)
        if counts[i] > 1:
            counts[i] -= 1
        else:
            break

    while counts.sum() < n:
        i = np.argmax(lengths / (counts + 1e-9))
        counts[i] += 1

    pieces = []
    for (a, b), count in zip(segments, counts):
        pieces.append(polyline_sample([a, b], int(count)))

    points = np.vstack(pieces)

    # Normalize the letter size
    points -= points.mean(axis=0)
    scale = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    points *= 4.0 / scale

    return points

# 2. Define 20 desired positions for each capital letter

letters = {}

letters["A"] = sample_segments([
    ((-1.8, -2.0), (0, 2.0)),
    ((0, 2.0), (1.8, -2.0)),
    ((-0.9, -0.2), (0.9, -0.2))
], N)

theta = np.linspace(0, np.pi, 7)
curve = np.c_[1.0 + 0.9*np.cos(theta),
              -1.15 - 0.85*np.sin(theta)]

J_segments = [
    ((1.35, 2.0), (-1.35, 2.0)),
    ((1.35, 2.0), (1.35, -1.15))
]
for a, b in zip(curve[:-1], curve[1:]):
    J_segments.append((tuple(a), tuple(b)))

letters["J"] = sample_segments(J_segments, N)

letters["Y"] = sample_segments([
    ((-1.8, 2.0), (0, 0.0)),
    ((1.8, 2.0), (0, 0.0)),
    ((0, 0.0), (0, -2.1))
], N)

# Keep the center fixed because the formation controller is
# translation invariant.
center = X.mean(axis=0)
targets = {ch: pts + center for ch, pts in letters.items()}

sequence = list("AJAY")


# 3. Simulate the formation-control dynamics

frames = [X.copy()]
labels = ["Initial random positions"]

for letter in sequence:
    target_points = targets[letter]

    # Assign the agents to the target points with minimum total distance.
    cost = ((X[:, None, :] - target_points[None, :, :])**2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)

    q = np.empty_like(target_points)
    q[rows] = target_points[cols]

    for _ in range(STEPS_PER_LETTER):
        # u_i = -K sum_j a_ij [
        #          (x_i - x_j) - (q_i - q_j)
        #       ]
        #
        # This is a distributed formation-control law because
        # agent i only needs relative position information from
        # its graph neighbours.
        relative_x = X[:, None, :] - X[None, :, :]
        relative_q = q[:, None, :] - q[None, :, :]

        U = -K * np.sum(
            A[:, :, None] * (relative_x - relative_q),
            axis=1
        )

        X = X + DT * U
        frames.append(X.copy())
        labels.append(letter)

    for _ in range(HOLD_STEPS):
        frames.append(X.copy())
        labels.append(letter)

frames = np.asarray(frames)

# 4. Create animation

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(center[0] - 4.5, center[0] + 4.5)
ax.set_ylim(center[1] - 4.5, center[1] + 4.5)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)
ax.set_xlabel("x")
ax.set_ylabel("y")

scat = ax.scatter([], [], s=55)

edge_lines = []
for i, j in G.edges():
    line, = ax.plot([], [], linewidth=0.7, alpha=0.25)
    edge_lines.append((line, i, j))

title = ax.set_title("AI3403 MAS — Formation Control")

def update(frame_id):
    Xf = frames[frame_id]

    scat.set_offsets(Xf)

    for line, i, j in edge_lines:
        line.set_data(
            [Xf[i, 0], Xf[j, 0]],
            [Xf[i, 1], Xf[j, 1]]
        )

    label = labels[frame_id]
    if label == "Initial random positions":
        title.set_text(
            "AI3403 MAS — N=20 Connected Erdos-Renyi Graph\n"
            "Initial random positions"
        )
    else:
        title.set_text(
            f"AI3403 MAS — Formation Control: {label}\n"
            "Name sequence: A → J → A → Y"
        )

    return [scat, title] + [line for line, _, _ in edge_lines]

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False
)

writer = FFMpegWriter(
    fps=20,
    metadata={"title": "AJAY Formation Control - AI3403 MAS"},
    bitrate=1200
)

ani.save("AJAY_formation_control.mp4", writer=writer)
plt.close(fig)

print("Video saved as AJAY_formation_control.mp4")
print("Graph connected:", nx.is_connected(G))
print("Number of agents:", N)
print("Number of edges:", G.number_of_edges())
print("Sequence:", " -> ".join(sequence))

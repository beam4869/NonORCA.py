"""Plot the manuscript's reported illustrative-example matrix without resampling.

Claim: f1 and f2 form the only multi-objective group at fixed K=3.
Contract: single quantitative network; Python/Matplotlib; 89 x 89 mm;
editable PDF/SVG and 600-dpi PNG; six edges, no loops or thresholding.
Source: reported three-decimal matrix, reproduced in the adjacent CSV.
Style: square arrangement with distinct blue, orange, and green communities.
"""

import os
from pathlib import Path


import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def main():
    matrix = np.loadtxt(HERE / "illustrative_correlation_matrix.csv",
                        delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    assert matrix.shape == (4, 4) and np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 1) and np.all((0 <= matrix) & (matrix <= 1))
    # At K=3 only the first merge occurs; f1/f2 uniquely maximize affinity.
    pairs = [(matrix[i, j], i, j) for i in range(4) for j in range(i + 1, 4)]
    ranked = sorted(pairs, reverse=True)
    assert ranked[0][1:] == (0, 1) and ranked[0][0] > ranked[1][0]
    groups = [0, 0, 1, 2]

    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_weighted_edges_from((i, j, matrix[i, j])
                                 for i in range(4) for j in range(i + 1, 4))
    positions = {0: (-0.85, 0.85), 1: (0.85, 0.85),
                 2: (-0.85, -0.85), 3: (0.85, -0.85)}
    palette = ["#4477AA", "#EEAA33", "#228833"]
    colors = [palette[group] for group in groups]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 89 / 25.4))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)

    # A positive floor keeps the weakest relationships visible.
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#333333",
                           width=[0.45 + 2.6 * data["weight"]
                                  for _, _, data in graph.edges(data=True)])
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors,
                           node_size=5000, edgecolors="black", linewidths=0.75)
    for node, (x, y) in positions.items():
        ax.text(x, y, rf"$f_{{{node + 1}}}$", ha="center", va="center",
                fontsize=12, color="black" if groups[node] == 1 else "white")

    label_positions = {(0, 3): 0.68, (1, 2): 0.68}
    for left, right, data in graph.edges(data=True):
        nx.draw_networkx_edge_labels(
            graph, positions, ax=ax,
            edge_labels={(left, right): f'{data["weight"]:.3f}'},
            label_pos=label_positions.get((left, right), 0.5),
            font_size=8.5, font_color="black", rotate=True,
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white",
                  "edgecolor": "none", "alpha": 1.0})
    ax.set(xlim=(-1.23, 1.23), ylim=(-1.23, 1.23), aspect="equal")
    ax.axis("off")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"illustrative_objective_graph_square_v3.{extension}", dpi=600)
    plt.close(fig)
    print("Verified: 4 nodes, 6 edges, exact displayed weights, K=3 partition [0,0,1,2].")


if __name__ == "__main__":
    main()

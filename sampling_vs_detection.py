from math import comb, ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import hypergeom

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'

L = 34       # Llama-3.1-8B layers
T = 136      # training steps (2048 samples / batch_size=16)
S_b = 16     # batch size
n_p = 205    # poisoned samples (10% of 2048)

def p_detect_uniform(N, k, pi):
    m = round(pi * N)
    if k <= 0: return 0.0
    if k >= N: return 1.0
    if m >= N: return 1.0
    if m == 0: return 0.0
    return 1 - comb(N - k, m) / comb(N, m)

def p_detect_input_only(N_blocks, N_input, k_input, pi_all):
    m = min(round(pi_all * N_blocks), N_input)
    if k_input <= 0: return 0.0
    if k_input >= N_input: return 1.0
    if m >= N_input: return 1.0
    if m == 0: return 0.0
    return 1 - comb(N_input - k_input, m) / comb(N_input, m)

rows = []

# Under-training (B_L=4, B_S=8 only, for reference)
for alpha, label in [(0.50, "Skip 50\\% steps"), (0.25, "Skip 25\\% steps")]:
    BL, BS = 4, 8
    N = (L // BL) * (T // BS)
    k = round(alpha * N)
    pi = 0.05
    rows.append(("Under-training", label, f"$B_L={BL},B_S={BS}$", "Uniform", pi, p_detect_uniform(N, k, pi), k, N))

# Data poisoning: B_S=8 and B_S=4, both strategies
for BS, BS_label in [(8, "B_S=8"), (4, "B_S=4")]:
    BL = 4
    N = (L // BL) * (T // BS)
    N_input = T // BS
    k_uniform = ceil(n_p / (S_b * BS))
    k_input   = ceil(n_p / (S_b * BS))
    pi_u, pi_i = 0.20, 0.10
    rows.append(("Data poisoning", "10\\% poison rate", f"$B_L={BL},{BS_label}$", "Uniform",    pi_u, p_detect_uniform(N, k_uniform, pi_u),              k_uniform, N))
    rows.append(("Data poisoning", "10\\% poison rate", f"$B_L={BL},{BS_label}$", "Input-only", pi_i, p_detect_input_only(N, N_input, k_input, pi_i), k_input,   N))

# Model substitution: B_L=4 and B_L=2, B_S=8
for BL, BL_label in [(4, "B_L=4"), (2, "B_L=2")]:
    BS = 8
    N = (L // BL) * (T // BS)
    for beta, label in [(1.00, "Replace 100\\% layers"), (0.75, "Replace 75\\% layers")]:
        k = ceil(L * beta / BL)
        pi = 0.20
        rows.append(("Model substitution", label, f"${BL_label},B_S={BS}$", "Uniform", pi, p_detect_uniform(N, k, pi), k, N))

# Parameter tampering: B_L=4 and B_L=2, B_S=8
for BL, BL_label, pi_20, pi_10 in [(4, "B_L=4", 0.30, 0.50), (2, "B_L=2", 0.30, 0.50)]:
    BS = 8
    N = (L // BL) * (T // BS)
    for beta, label, pi in [(0.20, "Modify 20\\% params", pi_20), (0.10, "Modify 10\\% params", pi_10)]:
        k = max(1, ceil(L * beta / BL))
        rows.append(("Parameter tampering", label, f"${BL_label},B_S={BS}$", "Uniform", pi, p_detect_uniform(N, k, pi), k, N))

print(f"{'Attack Type':<22} {'Strength':<24} {'Config':<18} {'Strategy':<12} {'pi':>5}  {'k':>4}  {'N':>4}  {'P_detect':>9}")
print("-" * 105)
for attack, strength, config, strategy, pi, p, k, N in rows:
    print(f"{attack:<22} {strength:<24} {config:<18} {strategy:<12} {pi*100:>4.0f}%  {k:>4}  {N:>4}  {p*100:>8.1f}%")



N = 17 * 8  # 136 blocks: Llama-3.1-8B, B_L=4, B_S=8, 2048 steps
p_vals = np.linspace(0, 0.5, 500)
k_vals = [2, 4, 8, 16, 32, 64]

fig, ax = plt.subplots(figsize=(5, 3.5))
colors = plt.cm.tab10(np.linspace(0, 0.5, len(k_vals)))

for color, k in zip(colors, k_vals):
    m_vals = np.round(p_vals * N).astype(int)

    P_exact = np.array([
        1 - hypergeom.pmf(0, N, k, m) for m in m_vals
    ])
    P_approx = 1 - (1 - p_vals) ** k

    ax.plot(p_vals, P_exact, color=color, linewidth=1.5, label=f"$k={k}$")
    ax.plot(p_vals, P_approx, color=color, linewidth=1.5, linestyle="--", alpha=1)

ax.axhline(0.9, color="gray", linestyle=":", linewidth=1.0)
ax.text(0.505, 0.905, "90%", va="bottom", fontsize=8, color="gray")

legend_k = ax.legend(fontsize=8, ncol=1, loc="lower right", title="Compromised\nblocks $k$")
ax.add_artist(legend_k)

handle_exact = mlines.Line2D([], [], color="black", linewidth=1.5, label=f"$N={N}$")
handle_approx = mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="--", alpha=1, label=r"$N\to\infty$")
ax.legend(handles=[handle_exact, handle_approx], fontsize=8, loc="upper left")

ax.set_xlabel("Sampling Rate $\\pi$")
ax.set_ylabel("Detection Probability $P_{\\mathrm{detect}}$")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig("sampling_detection.png", dpi=600, bbox_inches="tight")

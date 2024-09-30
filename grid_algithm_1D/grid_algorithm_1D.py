import math

import matplotlib.pyplot as plt
import numpy as np

from Z_sqrt2 import Z_sqrt2, inv_lamb, lamb

A: np.ndarray = np.array([-5, 0])
B: np.ndarray = np.array([7, 15])
deltaA: float = A[1] - A[0]

n_scaling: int = math.ceil(-math.log(deltaA) / math.log(inv_lamb))
A_scaled: np.ndarray = A * float(inv_lamb**n_scaling)
B_scaled: np.ndarray = B * float((-lamb) ** n_scaling)
assert (A_scaled[1] - A_scaled[0]) < 1 and (A_scaled[1] - A_scaled[0]) >= float(inv_lamb)

if n_scaling % 2 == 0:
    b_interval_scaled: np.ndarray = np.array(
        [
            (A_scaled[0] - B_scaled[1]) / math.sqrt(8),
            (A_scaled[1] - B_scaled[0]) / math.sqrt(8),
        ]
    )
else:
    b_interval_scaled: np.ndarray = np.array(
        [
            (A_scaled[0] - B_scaled[0]) / math.sqrt(8),
            (A_scaled[1] - B_scaled[1]) / math.sqrt(8),
        ]
    )

b_start: int = math.ceil(b_interval_scaled[0])
b_end: int = math.floor(b_interval_scaled[-1])
assert b_start <= b_end

b_list: list[int] = list(range(b_start, b_end + 1))
alpha: list[Z_sqrt2] = []

for bi in b_list:
    a_interval_scaled: list[float] = [
        A_scaled[0] - bi * math.sqrt(2),
        A_scaled[1] - bi * math.sqrt(2),
    ]
    if math.ceil(a_interval_scaled[0]) == math.floor(a_interval_scaled[1]):
        alpha_scaled = math.ceil(a_interval_scaled[0]) + bi*math.sqrt(2)
        alpha_conjugate_scaled = math.ceil(a_interval_scaled[0]) - bi*math.sqrt(2)
        if (alpha_scaled >= A_scaled[0] and alpha_scaled <= A_scaled[1] and alpha_conjugate_scaled >= B_scaled[0] and alpha_conjugate_scaled <= B_scaled[1]):
            alpha.append(Z_sqrt2(math.ceil(a_interval_scaled[0]), bi) * lamb**n_scaling)

if len(alpha) == 0:
    print("No solutions were found for the grid problem")
else:
    print(f"Solutions: {alpha}")

for alpha_i in alpha:
    assert float(alpha_i) <= A[1] and float(alpha_i) >= A[0]
    assert float(alpha_i.conjugate()) <= B[1] and float(alpha_i.conjugate()) >= B[0]

plt.figure(figsize=(8,3))
plt.axhline(color="k", linestyle = "--", linewidth = 0.7)
plt.axvline(color="k", linestyle = "--", linewidth = 0.7)
plt.grid(axis = 'x')
plt.scatter(
    [float(i) for i in alpha],
    [0] * len(alpha),
    color="blue",
    s=25,
    label=r"$\alpha$",
)
plt.scatter(
    [float(i.conjugate()) for i in alpha],
    [0] * len(alpha),
    color="red",
    s=20,
    marker="x",
    label=r"$\alpha^\bullet$",
)

plt.ylim((-1, 1))
plt.axvspan(A[0], A[1], color="blue", alpha=0.2, label="A")
plt.axvspan(B[0], B[1], color="red", alpha=0.2, label="B")

plt.title("Solutions for the 1 dimensional grid problem for A and B")
plt.yticks([])

plt.legend()
plt.savefig("Solutions/solutions.png", dpi=200)

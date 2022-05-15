import math

import matplotlib.pyplot as plt
from numpy import *

labels1 = [
    "Job-Art 5: Priorität 12",
    "Job-Art 4: Priorität 10",
    "Job-Art 3: Priorität 8",
    "Job-Art 2: Priorität 6",
    "Job-Art 1: Priorität 4",
    "Job-Art 0: Priorität 2",
]

t = linspace(5, 1, 400)
discontfactor = 0.75
a = 12 * discontfactor ** t
b = 10 * discontfactor ** t
c = 8 * discontfactor ** t
d = 6 * discontfactor ** t
e = 4 * discontfactor ** t
f = 2 * discontfactor ** t

plt.plot(t, a, "r")  # plotting t, a separately
plt.plot(t, b, "b")  # plotting t, b separately
plt.plot(t, c, "g")  # plotting t, c separately
plt.plot(t, d, "m")
plt.plot(t, e, "c")
plt.plot(t, f, "k")
plt.axhline(y=(12) * discontfactor ** 5, color="r", linestyle="dashed")
# plt.axhline(y = (10)*discontfactor**5, color = 'b', linestyle = 'dashed')
# plt.axhline(y = (8)*discontfactor**5, color = 'g', linestyle = 'dashed')
# plt.axhline(y = (6)*discontfactor**5, color = 'm', linestyle = 'dashed')
# plt.axhline(y = (4)*discontfactor**5, color = 'c', linestyle = 'dashed')
# plt.axhline(y = 2/5, color = 'k', linestyle = '-')
plt.scatter(1, 10 * discontfactor, marker="x", c="r")
plt.scatter(1, 8 * discontfactor, marker="x", c="r")
plt.scatter(1, 6 * discontfactor, marker="x", c="r")
plt.scatter(1, 4 * discontfactor, marker="x", c="r")
plt.scatter(2, 10 * discontfactor ** 2, marker="x", c="r")
plt.scatter(2, 8 * discontfactor ** 2, marker="x", c="r")
plt.scatter(2, 6 * discontfactor ** 2, marker="x", c="r")
plt.scatter(3, 10 * discontfactor ** 3, marker="x", c="r")
plt.scatter(3, 8 * discontfactor ** 3, marker="x", c="r")
plt.scatter(4, 10 * discontfactor ** 4, marker="x", c="r")
plt.grid(linestyle="dashed")
plt.xticks(range(1, 6))
plt.xlabel("Verbleibende Restzeit")
plt.ylabel("Diskontiertes Reward-Verhältnis der Job-Art")
plt.legend(labels=labels1, ncol=2)
plt.ylim(bottom=0, top=15)
plt.savefig("C:/Users/lenna/Desktop/Diskontierte Reward-Verhältnisse 075.png", dpi=300)

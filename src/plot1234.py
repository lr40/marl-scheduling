import math

import matplotlib.pyplot as plt
from numpy import *

labels1 = ["Preis der Job-Art 1: Preis-Höhe 7", "Job-Art 0: Priorität 3"]

t1 = linspace(6, 0, 400)
t2 = linspace(3, 0, 400)
discontfactor = 0.8733333
a = 10 * discontfactor ** t2
b = 3 * discontfactor ** t1
c = 7 * discontfactor ** t2
d = 2 * discontfactor ** t1


plt.plot(t2, c, "orange")  # plotting t, a separately
plt.plot(t1, b, "blue")  # plotting t, b separately


# plt.axhline(y = (12)*discontfactor**5, color = 'r', linestyle = 'dashed')
# plt.axhline(y = (10)*discontfactor**5, color = 'b', linestyle = 'dashed')
# plt.axhline(y = (8)*discontfactor**5, color = 'g', linestyle = 'dashed')
# plt.axhline(y = (6)*discontfactor**5, color = 'm', linestyle = 'dashed')
# plt.axhline(y = (4)*discontfactor**5, color = 'c', linestyle = 'dashed')
# plt.axhline(y = 2/5, color = 'k', linestyle = '-')
# plt.scatter(1,10*discontfactor,marker="x",c="r")
# plt.scatter(1,8*discontfactor,marker="x",c="r")
# plt.scatter(1,6*discontfactor,marker="x",c="r")
# plt.scatter(1,4*discontfactor,marker="x",c="r")
# plt.scatter(2,10*discontfactor**2,marker="x",c="r")
# plt.scatter(2,8*discontfactor**2,marker="x",c="r")
# plt.scatter(2,6*discontfactor**2,marker="x",c="r")
# plt.scatter(3,10*discontfactor**3,marker="x",c="r")
# plt.scatter(3,8*discontfactor**3,marker="x",c="r")
# plt.scatter(4,10*discontfactor**4,marker="x",c="r")
plt.grid(linestyle="dashed")
plt.xticks(range(0, 7))
plt.yticks(range(0, 12))
plt.xlabel("Verbleibende Restzeit")
plt.ylabel("Diskontierter Reward der Job-Art")
plt.legend(labels=labels1)
plt.ylim(bottom=0, top=11)
plt.savefig("C:/Users/lenna/Desktop/Diskontierter Reward Exp1.png", dpi=300)

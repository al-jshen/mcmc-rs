import data
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots

fig = plt.figure()

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
for p1, p2, p3 in data.chains:
    ax1.plot(p1, p2)
    ax1.set_title("p1 x p2")
    ax2.plot(p2, p3)
    ax2.set_title("p2 x p3")
    ax3.plot(p1, p3)
    ax3.set_title("p1 x p3")
    ax4.plot(p1)
    ax4.set_title("p1")
    ax5.plot(p2)
    ax5.set_title("p2")
    ax6.plot(p3)
    ax6.set_title("p3")

plt.show()


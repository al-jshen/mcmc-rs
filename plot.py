import data
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots

fig = plt.figure()

ax1 = fig.add_subplot(321)
ax1.hist(data.d)

ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
for p1, p2 in data.chains:
    ax2.plot(p1, p2)
    ax3.plot(p1)
    ax4.plot(p2)
    tsaplots.plot_acf(np.array(p1), ax=ax5, lags=100)
    tsaplots.plot_acf(np.array(p2), ax=ax6, lags=100)

plt.show()


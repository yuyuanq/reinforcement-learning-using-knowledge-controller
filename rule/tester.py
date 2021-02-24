import matplotlib.pyplot as plt
import numpy as np
from rule_mountaincarcontinuous import MountainCarContinuousRule

rule = MountainCarContinuousRule()
x1 = np.linspace(-0.1, 0.11, 1000)
x2 = rule.s1_po(x1)

plt.plot(x1, x2)
plt.show()

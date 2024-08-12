import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sbn

sbn.set_theme()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)
plt.title("Linear Regression")
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()
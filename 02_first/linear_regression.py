import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sbn

def predict(x_array, w):
    return x_array*w

def loss(x_array, y_array, w):
    return np.average((predict(x_array, w) - y_array)**2)

def train(x_array, y_array, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(x_array, y_array, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        
        # lr: learning rate
        if loss(x_array, y_array, w + lr) < current_loss:
            w += lr
        elif loss(x_array, y_array, w - lr) < current_loss:
            w -= lr
        else:
            return w

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w = train(X,Y, 10000, 0.01)
print("%.6f" % (w))
       
sbn.set_theme()
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)
plt.title("Linear Regression")
plt.plot(X, Y, "bo")
plt.plot([0, x_edge], [0, predict(x_edge, w)], linewidth=1.0, color="g")
plt.show()
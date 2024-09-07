import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sbn

def predict(x_array, w, b):
    return x_array*w + b

def loss(x_array, y_array, w, b):
    return np.average((predict(x_array, w, b) - y_array)**2)

def train(x_array, y_array, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        current_loss = loss(x_array, y_array, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        
        # lr: learning rate
        if loss(x_array, y_array, w + lr, b) < current_loss:
            w += lr
        elif loss(x_array, y_array, w - lr, b) < current_loss:
            w -= lr
        elif loss(x_array, y_array, w, b + lr) < current_loss:
            b += lr
        elif loss(x_array, y_array, w, b - lr) < current_loss:
            b -= lr    
        else:
            return w, b

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w , b= train(X,Y, 10000, 0.01)
print("w => %.6f, b => %.6f" % (w, b))
       
sbn.set_theme()
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)
plt.title("Linear Regression")
plt.plot(X, Y, "bo")
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()
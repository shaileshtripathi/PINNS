import re
import numpy as np
import pandas as pd
import deepxde as dde
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme_minimal, ggsave

# Define parameters as variables

C1 = dde.Variable(2.5)  # Initial guess for the growth rate
C2 = dde.Variable(70)  # Initial guess for the carrying capacity

# Define the ODE system for logistic growth
def ode_system(x, y):
    r = C1
    K = C2
    P = y
    dP_dt = dde.grad.jacobian(y, x)
    return [dP_dt - r * P * (1 - P / K)]

# Define the initial condition function
def initial_condition(x):
    return np.ones(x.shape[0])  # Ensure it returns an array with the same shape as x

# Define the exact solution for the logistic growth equation
def exact_solution(t):
    P0 = 1  # Initial population size
    r = 0.5  # Growth rate
    K = 100  # Carrying capacity
    return (K * P0 * np.exp(r * t)) / ((K - P0) + P0 * np.exp(r * t))

# Define observation points for boundary conditions
def ode_system_obs(num):
    xvals = np.linspace(0, 100, num, endpoint=False)
    yvals = exact_solution(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))

# Define the domain
geom = dde.geometry.TimeDomain(0, 100)

def boundary(x, on_initial):
    return on_initial

def mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))
# Define initial condition using dde.IC
# Use lambda x: True to specify the initial condition is applied throughout the domain
ic = dde.IC(geom, initial_condition, lambda x, on_initial: on_initial)

# Define boundary condition using PointSetBC
ob_x, ob_u = ode_system_obs(30)
observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)

# Define the PDE data object
data = dde.data.PDE(
    geom,
    ode_system,
    solution=exact_solution,
    bcs=[observe_u],
    num_domain=380,
    num_boundary=20,
    train_distribution="uniform",
    num_test=100
)

# Define the neural network
net = dde.nn.FNN([1] + [50] * 5 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: abs(y))

# Define and compile the model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss="MSE", metrics=["l2 relative error"],  external_trainable_variables=[C1, C2])
#model.compile("L-BFGS")

#model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=[C1, C2])
variable = dde.callbacks.VariableValue([C1, C2], period=200, filename="variables1.dat")

# Train the model
print(variable.value)
losshistory, train_state = model.train(iterations=100000, callbacks=[variable], display_every=200)


# Read and process variable data
lines = open("variables1.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(r"\[(.*?)\]", line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

# Define true values for comparison
C1true = 0.5
C2true = 100

# Prepare data for plotting
epochs = range(0, 200 * vkinfer.shape[0], 200)
data_r = pd.DataFrame({'epochs': epochs, 'True': np.ones(len(epochs)) * C1true, 'Predicted': vkinfer[:, 0]})
data_K = pd.DataFrame({'epochs': epochs, 'True': np.ones(len(epochs)) * C2true, 'Predicted': vkinfer[:, 1]})

# Plot results using plotnine

# Plot for C1
plot_r = (ggplot(data_r, aes(x='epochs')) +
          geom_line(aes(y='True', color='"Exact"'), linetype='dashed') +
          geom_line(aes(y='Predicted', color='"Pred"')) +
          labs(title='Growth Rate (r) Over Epochs', x='Epoch', y='Value') +
          theme_minimal())

print(plot_r)

# Plot for C2
plot_K = (ggplot(data_K, aes(x='epochs')) +
          geom_line(aes(y='True', color='"Exact"'), linetype='dashed') +
          geom_line(aes(y='Predicted', color='"Pred"')) +
          labs(title='Carrying Capacity (K) Over Steps', x='Steps', y='Value') +
          theme_minimal())

print(plot_K)

# Save plots to files
ggsave(plot_r, filename='growth_rate_plot.pdf', dpi=300)
ggsave(plot_K, filename='carrying_capacity_plot.pdf', dpi=300)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
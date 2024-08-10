import random
import numpy as np
import pandas as pd
import deepxde as dde

random.seed(46)

#observation at epochs
PE =500

#learning rate 
LR = 0.001
# obsservation samples
n=30
# number of hidden layers in the neural networks
H=3
# number of iterations for
IT=200000
# number of output neurons
OP = 1
# number of neurons in a hidden layer
NE = 50
# Time interval 
t0 = 0 # start time
t1 = 100 # end time

# selection: parameter estimation or Forward 
S = True # True for inverse problem, False for forward problem

# Define parameters as variables
if S:
    C1 = dde.Variable(2.5)  # Initial guess for the growth rate
    C2 = dde.Variable(70)  # Initial guess for the carrying capacity
else:
    C1 = 0.5
    C2 = 100
    
# Define the ODE system for logistic growth
def ode_system(x, y, C1, C2):
    r = C1
    K = C2
    P = y
    dP_dt = dde.grad.jacobian(y, x)
    return [dP_dt - r * P * (1 - P / K)]

# Define the initial condition function
def initial_condition(x):
    return np.ones(x.shape[0])  # Ensure it returns an array with the same shape as x

def ode_system_obs(num):
    xvals = np.linspace(t0, t1, num, endpoint=False)
    yvals = exact_solution(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))

# Define the exact solution for the logistic growth equation
def exact_solution(t):
    P0 = 1  # Initial population size
    r = 0.5  # Growth rate
    K = 100  # Carrying capacity
    return (K * P0 * np.exp(r * t)) / ((K - P0) + P0 * np.exp(r * t))

# Define the domain
geom = dde.geometry.TimeDomain(t0, t1)
ob_x, ob_y = ode_system_obs(n)
observe_u = dde.icbc.PointSetBC(ob_x, ob_y, component=0)

# Define initial condition using dde.IC
ic = dde.IC(geom, initial_condition, lambda x, on_initial: on_initial)

# Define the PDE data object for the ODE
data = dde.data.PDE(
    geom,
    ode_system,
    bcs=[ic,observe_u],
    num_domain=350,
    num_boundary=3,
    train_distribution="uniform",
    solution=exact_solution,
    num_test=100
)

# Define the neural network
net = dde.nn.FNN([1] + [NE] * H + [OP], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: abs(y))


checker = dde.callbacks.ModelCheckpoint(
    "modelx/model.ckpt", save_better_only=False, period=PE
)

# Define and compile the model
model = dde.Model(data, net)
if S: 
    model.compile("adam", lr=LR, loss="MSE", metrics=["l2 relative error"],  external_trainable_variables=[C1, C2])
    variable = dde.callbacks.VariableValue([C1, C2], period=PE, filename="variables1.dat")
    # Train the model
    losshistory, train_state = model.train(iterations=IT, callbacks=[variable, checker], display_every=PE)
else:
   model.compile("adam", lr=LR, loss="MSE", metrics=["l2 relative error"])
   # Train the model
   losshistory, train_state = model.train(iterations=IT, callbacks=[checker], display_every=PE)

# Save and plot the results
# files loss.dat, train.dat and test.dat are saved
dde.saveplot(losshistory, train_state, issave=True, isplot=True)















import re
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme_minimal, ggsave

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
epochs = range(0, 500 * vkinfer.shape[0], 500)
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

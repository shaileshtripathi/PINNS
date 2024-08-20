import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import deepxde as dde
import re
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, ggsave

# Define true parameters
C1_true = 1.0 / 60  # True value for kon
C2_true = 0.5 / 60  # True value for koff

# Define the ODE system for generating synthetic data
def ode_system_true(y, t):
    prom, tf, promtf = y
    kon = C1_true
    koff = C2_true

    dprom_dt = (-kon * prom * tf + koff * promtf)
    dtf_dt = (-kon * prom * tf + koff * promtf)
    dpromtf_dt = (kon * prom * tf - koff * promtf)

    return [dprom_dt, dtf_dt, dpromtf_dt]

# Initial conditions
y0 = [17.0, 25, 0]  # Initial values for prom, tf, and promtf

# Time points at which the solution is to be computed
t = np.linspace(0, 100, 300)
tsol = t[:, None]

# Solve the ODE system to generate synthetic data
solution = odeint(ode_system_true, y0, t)

# Define variables for PINN
C1 = dde.Variable(0.1)  # Initial guess close to true value
C2 = dde.Variable(0.1)  # Initial guess close to true value

# Define the ODE system for PINN
def ode_system(t, y):
    prom, tf, promtf = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    kon = C1
    koff = C2

    dprom_dt = dde.grad.jacobian(y, t, i=0)
    dtf_dt = dde.grad.jacobian(y, t, i=1)
    dpromtf_dt = dde.grad.jacobian(y, t, i=2)

    a = dprom_dt - (-kon * prom * tf + koff * promtf)
    b = dtf_dt - (-kon * prom * tf + koff * promtf)
    c = dpromtf_dt - (kon * prom * tf - koff * promtf)

    return [a, b, c]

# Initial condition functions for PINN
def initial_prom(t):
    return np.full((len(t), 1), 17)

def initial_tf(t):
    return np.full((len(t), 1), 25)

def initial_promtf(t):
    return np.full((len(t), 1), 0)

# Time domain for PINN
geom = dde.geometry.TimeDomain(0, 100)

# Initial conditions for PINN
ic_prom = dde.IC(geom, initial_prom, lambda _, on_initial: on_initial, component=0)
ic_tf = dde.IC(geom, initial_tf, lambda _, on_initial: on_initial, component=1)
ic_promtf = dde.IC(geom, initial_promtf, lambda _, on_initial: on_initial, component=2)

# Observational data as PointSetBC
observe_prom = dde.icbc.PointSetBC(tsol, solution[:, 0:1], component=0)
observe_tf = dde.icbc.PointSetBC(tsol, solution[:, 1:2], component=1)
observe_promtf = dde.icbc.PointSetBC(tsol, solution[:, 2:3], component=2)

# Data object for PINN
data = dde.data.TimePDE(
    geom,
    ode_system,
    ic_bcs=[ic_prom, ic_tf, ic_promtf, observe_prom, observe_tf, observe_promtf],
    num_domain=500,
    num_boundary=6,
    train_distribution="uniform",
    num_test=400
)

# Neural network definition for PINN
net = dde.nn.FNN([1] + [50] * 4 + [3], "tanh", "Glorot normal")

# Model for PINN
model = dde.Model(data, net)

# Compile the model for PINN
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2])

# Define a callback to monitor variable values 
variable = dde.callbacks.VariableValue([C1, C2], period=50, filename="variableszzx.dat")
########################
# Define a callback to save model at every 50 iterations
checker = dde.callbacks.ModelCheckpoint(
    "modelgene/model.ckpt", save_better_only=False, period=50
)
#########################

# Train the model for PINN
losshistory, train_state = model.train(epochs=10000, callbacks=[variable, checker], display_every=50)
#losshistory, train_state = model.train(epochs=10000)


# Plot results for PINN
t_pred = np.linspace(0, 100, 1000)[:, None]
y_pred = model.predict(t_pred)

plt.figure(figsize=(10, 5))
plt.plot(t, solution[:, 0], 'r-', label='True prom')
plt.plot(t, solution[:, 1], 'b-', label='True tf')
plt.plot(t, solution[:, 2], 'g-', label='True promtf')
plt.plot(t_pred, y_pred[:, 0], 'r--', label='Predicted prom')
plt.plot(t_pred, y_pred[:, 1], 'b--', label='Predicted tf')
plt.plot(t_pred, y_pred[:, 2], 'g--', label='Predicted promtf')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Constitutive Gene Expression Model')
plt.legend()
plt.grid(True)
plt.show()

# Analyze and plot inferred variables
lines = open("variableszzx.dat", "r").readlines()
vkinfer = np.array(
    [np.fromstring(min(re.findall(r"\[(.*?)\]", line), key=len), sep=",") for line in lines]
)

# Prepare data for plotting inferred variables
epochs = range(0, 500 * vkinfer.shape[0], 500)
data_r = pd.DataFrame({'epochs': epochs, 'True': np.ones(len(epochs)) * C1_true, 'Predicted': vkinfer[:, 0]})
data_K = pd.DataFrame({'epochs': epochs, 'True': np.ones(len(epochs)) * C2_true, 'Predicted': vkinfer[:, 1]})

# Plot inferred variables using plotnine
plot_r = (
    ggplot(data_r, aes(x='epochs')) +
    geom_line(aes(y='True', color='"Exact"'), linetype='dashed') +
    geom_line(aes(y='Predicted', color='"Pred"')) +
    labs(title='Inferred kon (C1) Over Epochs', x='Epoch', y='Value') +
    theme_minimal()
)
print(plot_r)

plot_K = (
    ggplot(data_K, aes(x='epochs')) +
    geom_line(aes(y='True', color='"Exact"'), linetype='dashed') +
    geom_line(aes(y='Predicted', color='"Pred"')) +
    labs(title='Inferred koff (C2) Over Steps', x='Steps', y='Value') +
    theme_minimal()
)
print(plot_K)

# Save plots to files
ggsave(plot_r, filename='inferred_kon_plot.pdf', dpi=300)
ggsave(plot_K, filename='inferred_koff_plot.pdf', dpi=300)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

##################################################################
## loading model at different checkpoints and making predictions##
## getting all models ############################################
import os
import glob
import re
import pickle



sptrn = os.path.join("modelgene/", '*.meta')
meta_files = glob.glob(sptrn)

pattern = r'ckpt-(\d+)\.ckpt\.meta'

# Function to extract the numeric part from a filename
def extract_number(filename):
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None

# Sort the filenames based on the extracted numeric part
sorted_filenames = sorted(meta_files, key=extract_number)

# Remove the '.meta' part from each filename
processed_filenames = [filename.replace('.meta', '') for filename in sorted_filenames]

######################################################################################
###### loading model one by one and makeing predictions ##############################
###### of trainnig and testing data ##################################################

X_pred = data.train_x
X_test = data.test_x
predictions_list = []
test_prediction_list=[]
num_models = len(processed_filenames)

for filename in processed_filenames:
    # Restore the model
    model.restore(filename, verbose=1)

    # Predict values
    y_pred = model.predict(X_pred)
    y_test = model.predict(X_test)

    #print(np.sum((train_state.y_train-data.train_y)**2))

    # Flatten the predictions and store them
    predictions_list.append(y_pred)
    test_prediction_list.append(y_test)

# Convert the list of predictions to a DataFrame
#column_names = [filename.split('/')[-1].replace('.ckpt', '') for filename in processed_filenames]
predictions_list.append(X_pred)
test_prediction_list.append(X_test)

with open('modelgene/predictions_train_df.pkl', 'wb') as file:
    pickle.dump(predictions_list, file)

with open('modelgene/predictions_test_df.pkl', 'wb') as file:
    pickle.dump(test_prediction_list, file)





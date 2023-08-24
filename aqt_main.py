import os
import torch
from aqt_nn import AQT_NN
from pfhedge.nn import Hedger
from pfhedge.nn import ExpectedShortfall
import time
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from cost_functions import RelativeCostFunction 
from qiskit import Aer
from qiskit_aqt_provider import AQTProvider

# IBM simulator/hardware as backend
# backend = Aer.get_backend('qasm_simulator')

# AQT backend. For hardware experiment: Change "offline_simulator_no_noise" to name of AQT hardware
provider = AQTProvider(os.environ['AQT_TOKEN'])
backend = provider.get_backend("offline_simulator_no_noise")

n_qubits = 5
n_layers = 5
in_features = 3
maturity = 4

underlier = BrownianStock(cost=RelativeCostFunction(cost=3.0e-4))
derivative = EuropeanOption(underlier, strike=1.1, maturity=maturity)


# Load data from jax trained model
backup_data = torch.load('model_weights.pth')
backup_data['quantum.params'] = backup_data.pop('quantum.weights')

criterion = ExpectedShortfall(0.1)
model = AQT_NN(n_qubits, n_layers, in_features, n_averages=1, backend=backend, shots=200)

# Feed jax trained parameters into model
model.load_state_dict(backup_data)
hedger = Hedger(model, ["log_moneyness", "time_to_maturity", "volatility"], criterion)

start = time.time()

# Performs n_batch*n_averages*shots circuit executions, where n_batch=250*maturity*n_paths
# A run for a batch of size 2 with 5 by 5 quantum circuit, 100 averages and 200 shots should
# roughly have a runtime of 7.5 seconds on Qiskit and  10 seconds on AQT (both simulators)
pnl = criterion(hedger.compute_pnl(derivative, n_paths=1)).item()

duration = time.time()-start
print(f'The execution lasted for {duration} seconds.')
print(pnl)

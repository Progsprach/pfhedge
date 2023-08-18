import torch
from pfhedge.nn import Hedger, ExpectedShortfall
from models import AQT_NN
from pfhedge.instruments import BrownianStock, EuropeanOption
from cost_functions import RelativeCostFunction
from quantum_circuits import SimpleQuantumCircuit


n_qubits = 5
n_layers = 5
in_features = 3
n_paths = 5000
maturity = 0.4
cost = 3e-4


underlier = BrownianStock(cost=RelativeCostFunction(cost=cost))
derivative = EuropeanOption(underlier, maturity=0.4, call=True, strike=1.1)


criterion = ExpectedShortfall(0.1)
circuit = SimpleQuantumCircuit(n_qubits, n_layers)
model = AQT_NN(circuit, in_features)
hedger = Hedger(model, inputs=['log_moneyness', 'time_to_maturity', 'volatility'])
backup_data = torch.load('model_weights.pth')
model.load_state_dict(backup_data)

import time
start = time.time()
print('Starting evaluation')
pnl = criterion(hedger.compute_pnl(derivative, n_paths=5000)).item()
duration = time.time()-start
print(f'This run took {duration} seconds')
print(pnl)


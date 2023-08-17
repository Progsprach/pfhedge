import torch
from aqt_nn import AQT_NN
from pfhedge.nn import Hedger
from pfhedge.nn import ExpectedShortfall
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from cost_functions import RelativeCostFunction 


n_qubits = 5
n_layers = 5
in_features = 3
maturity = 0.4


underlier = BrownianStock(cost=RelativeCostFunction(cost=3.0e-4))
derivative = EuropeanOption(underlier, strike=1.1, maturity=maturity)


# TODO: Find way to read in fit parameters from jax trained network
backup_data = torch.load('model_weights.pth')
backup_data['quantum.params'] = backup_data.pop('quantum.weights')


criterion = ExpectedShortfall(0.1)
model = AQT_NN(n_qubits, n_layers, in_features)
# model.load_state_dict(backup_data)
hedger = Hedger(model, ["log_moneyness", "time_to_maturity", "volatility"], criterion)


pnl = criterion(hedger.compute_pnl(derivative, n_paths=10)).item()
print(pnl)

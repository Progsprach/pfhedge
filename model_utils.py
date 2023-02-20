from typing import Sequence, List
from math import sqrt,log10
from torch.nn import Linear, Sequential
from pfhedge.nn import MultiLayerPerceptron
from models import ConstantLayer, MultiLayerHybrid, PreprocessingCircuit, NoPreprocessingCircuit
from jaxlayer import JaxLayer
from quantum_circuits import SimpleQuantumCircuit, ReuploadingQuantumCircuit
def classical_model_params(n_parameters: int, in_features: int, out_features: int = 1):
    if n_parameters <= 30:
        return [int((n_parameters-out_features)/(in_features+out_features+1))]
    if n_parameters <= 90:
        n_layers = 2
    elif n_parameters <= 200:
        n_layers = 3
    if n_parameters > 200:
        n_layers = int(log10(n_parameters/2))+1
    sol = solve_for_units(n_parameters,n_layers, in_features)
    return optimize_params(n_parameters,n_layers,sol, in_features)

def solve_for_units(n_parameters,n_layers,in_features=4, out_features=1):
    a = n_layers
    b = in_features+n_layers+out_features+1
    c = out_features-n_parameters
    return int((sqrt(b**2-(4*a*c))-b)/float(2*a))
def optimize_params(n_parameters, n_layers, n_units, in_features=4, out_features=1):
    seq = [n_units] * n_layers
    for i in range(n_layers):
        seq[i] += 1
        if(calc_num_params(seq, in_features, out_features)) > n_parameters:
            seq[i] -= 1
            break
    return seq
def calc_num_params(n_units: List[int], in_features=4, out_features=1)-> int:
    params = (in_features+1)*n_units[0]
    for i in range(len(n_units)-1):
        params += n_units[i]*n_units[i+1]
        params += n_units[i+1]
    params += n_units[-1]*out_features+1
    return params
def make_classical_model(n_parameters: int, n_features: int):
    if n_parameters == 1:
        return ConstantLayer()
    if n_parameters < (n_features+1):
        return Linear(in_features=n_parameters-1,out_features=1)
    if n_parameters < (n_features+3):
        return Linear(in_features=n_features, out_features=1)
    seq = classical_model_params(n_parameters, n_features)
    return MultiLayerPerceptron(n_features,1,len(seq),seq)
def make_quantum_model(n_parameters:int, n_features: int):
    if n_parameters < n_features:
        circuit = SimpleQuantumCircuit(n_parameters, n_layers=1, n_measurements=1)
        return JaxLayer(circuit)
    if n_parameters < 3*n_features:
        circuit = SimpleQuantumCircuit(n_features,n_layers=int(n_parameters/n_features),n_measurements=1)
        return JaxLayer(circuit)
    if n_parameters <= 60:
        circuit = SimpleQuantumCircuit(n_features,n_layers=int((n_parameters-1)/n_features-1),n_measurements=0)
        return NoPreprocessingCircuit(circuit)
    if n_parameters <= 90:
        quantum_parameters = n_parameters - 5*(n_features+1) - n_features- 1
        circuit = SimpleQuantumCircuit(5,n_layers=int(quantum_parameters/5),n_measurements=0)
        return PreprocessingCircuit(circuit,n_features)
    classical_params = n_parameters - 41
    seq = classical_model_params(classical_params,n_features,5)
    circuit = ReuploadingQuantumCircuit(5,3,2)
    return MultiLayerHybrid(circuit,n_features,1,len(seq),seq)
        
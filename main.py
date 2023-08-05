import seaborn
from InputReader import InputReader
from dotenv import load_dotenv
if __name__ == "__main__":

    import yaml
    with open('./config_qiskit.yaml', 'r') as file:
        config = yaml.safe_load(file)

    n_qubits = config['model']['n_qubits']
    n_layers = config['model']['n_layers']
    n_paths = config['training']['n_paths']
    maturity = config['derivative']['maturity']
    n_time = round(250*maturity)+1
    n_runs = n_paths*n_time*(2*n_layers*n_qubits+1)
    print(n_runs)

    print(n_time)
    # exit(0)
    load_dotenv()
    seaborn.set_style("whitegrid")
    reader = InputReader("config_qiskit.yaml")
    handler = reader.load_config()
    handler.full_process()

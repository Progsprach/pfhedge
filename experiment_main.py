import os
import json
import seaborn
from InputReader import InputReader
if __name__ == "__main__":
        
    n_qubits = 5
    n_layers = 5
    repeats = 2
    
    configurations = {
        'All x': [['x']*n_qubits]*n_layers,
        'All y': [['y']*n_qubits]*n_layers,
        'Alternating Rows': [['x', 'x', 'x', 'x', 'x'],
                             ['y', 'y', 'y', 'y', 'y'],
                             ['x', 'x', 'x', 'x', 'x'],
                             ['y', 'y', 'y', 'y', 'y'],
                             ['x', 'x', 'x', 'x', 'x']],
        'Alternating in Rows': [['x', 'y', 'x', 'y', 'x'],
                                ['y', 'x', 'y', 'x', 'y'],
                                ['x', 'y', 'x', 'y', 'x'],
                                ['y', 'x', 'y', 'x', 'y'],
                                ['x', 'y', 'x', 'y', 'x']],
        'Random': [['x', 'y', 'y', 'x', 'x'],
                   ['y', 'y', 'x', 'y', 'y'],
                   ['y', 'x', 'x', 'x', 'y'],
                   ['y', 'x', 'x', 'y', 'x'],
                   ['x', 'x', 'x', 'y', 'y']]
    }

    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    # results = open('results.txt', 'w')
    results_dict = {}

    folder = 'Experimental_Results'
    if not os.path.isdir(folder):
        os.mkdir(folder)

    for key, item in configurations.items():

        path1 = os.path.join(folder, key)
        if not os.path.isdir(path1):
            os.mkdir(path1)

        results_dict[key] = []
        reader.config['model']['circuit']['rotation_matrix'] = item
        print(f'Running configuration mode {key}:')
        print(reader.config['model']['circuit']['rotation_matrix'])
        handler = reader.load_config()

        # results.write(f'{key} ')
        
        for iteration in range(repeats):
            plots_folder = f'Run{iteration}'
            path2 = os.path.join(path1, plots_folder)
            if not os.path.isdir(path2):
                os.mkdir(path2)
            print(f'Starting iteration {iteration}')
            loss = handler.full_process(path2)
            handler.stock_diagrams(5, path2)
            results_dict[key].append(loss)

        print()
        
        # results.write('\n')

    filename = 'results.json'

    with open(os.path.join(folder, filename), 'w') as json_file:
        json.dump(results_dict, json_file)

    print('Data stored.') 

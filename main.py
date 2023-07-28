import os
import subprocess
import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    handler.full_process()
    
    store = True
    if store:
        foldername = input('Please enter foldername: ')
        os.mkdir(f'./Results/{foldername}')
        subprocess.run(['mv', 'output.txt', f'./Results/{foldername}'], check=True)
        for file in os.listdir('.'):
            if file.endswith('.png'):
                subprocess.run(['mv', file, f'./Results/{foldername}'], check=True)
        subprocess.run(['cp', 'config.yaml', f'./Results/{foldername}'], check=True)

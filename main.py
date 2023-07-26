import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    backup = input('Do you want to fall back to backup? (y/N)\n')
    if backup.lower() == 'y':
        handler.full_process(backup=True)
    else:
        handler.full_process(backup=False)

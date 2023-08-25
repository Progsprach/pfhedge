import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    import time
    start = time.time()
    handler.full_process()
    duration = time.time()-start
    print(f'Duration: {duration}')

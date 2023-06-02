import seaborn
from InputReader import InputReader
from dotenv import load_dotenv
if __name__ == "__main__":
    load_dotenv()
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    handler.full_process()

import time
import os
import shutil
import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")

    n_repeats = 10

    import numpy as np
    params_ar = np.unique(np.geomspace(10, 40, 10, dtype=int))
    print(params_ar)

    folder = './Benchmark'
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)

    file = open(os.path.join(folder, 'results.txt'), 'w')

    file.write('# n_repeats\truntime\tloss\ttime_std\tloss_std\n')

    try:

        for run, el in enumerate(params_ar):
            print(run)
            print(el)
            subfolder = os.path.join(folder, f'{el}_params')
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)
            
            print(f'Starting run with {el} parameters')
            runtime = 0
            runtime2 = 0
            loss = 0
            loss2 = 0
            for k in range(n_repeats):
                print(k)
                filename = f'loss_{el}_params.png'
                path = os.path.join(subfolder, filename)
                reader.config['model']['n_params'] = el
                handler = reader.load_config()
                start = time.time()
                loss_tmp = handler.full_process(path)
                del handler
                duration = time.time()-start
                loss += loss_tmp
                runtime += duration
                loss2 += loss_tmp**2
                runtime2 += duration**2

            runtime /= n_repeats
            loss /= n_repeats
            runtime2 /= n_repeats
            loss2 /= n_repeats
            std_time = np.sqrt(runtime2-runtime**2)
            std_loss = np.sqrt(loss2-loss**2)

            file.write(f'{el}\t{runtime}\t{loss}\t{std_time}\t{std_loss}\n')
            print()

    except Exception as e:
        print(e)

    finally:
        file.close()
        print('Completed')
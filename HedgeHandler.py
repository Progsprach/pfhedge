import os
from typing import List, Optional
import matplotlib.pyplot as plt
from pfhedge.instruments import BaseDerivative, BaseInstrument
from pfhedge.nn import Hedger, HedgeLoss, WhalleyWilmott, BlackScholes, Naked
from plotting_library import make_training_diagram, make_pl_diagram, make_stock_diagram
class HedgeHandler:
    def __init__(self, hedger: Hedger, derivative: BaseDerivative, hedge: List[BaseInstrument], fit_params: dict, profit_params: dict, criterion: HedgeLoss, benchmark_params: dict, backup_file: Optional[str] = None):
        self.hedger = hedger
        self.derivative = derivative
        self.hedge = hedge
        self.fit_params = fit_params
        self.profit_params = profit_params
        self.criterion = criterion
        self.benchmark_params = benchmark_params
        self.backup_file = backup_file
    def fit(self):
        return self.hedger.fit(self.derivative,self.hedge,**self.fit_params)
    def profit(self):
        return self.hedger.compute_pnl(self.derivative,self.hedge,**self.profit_params)
    def benchmark(self):
        output = {}
        if self.benchmark_params.get('WhalleyWilmott',False):
            compmodel = WhalleyWilmott(self.derivative)
            comphedger = Hedger(compmodel, inputs=compmodel.inputs())
            comp = comphedger.compute_pnl(self.derivative, **self.profit_params)
            output['Whalley-Wilmott'] = comp
        if self.benchmark_params.get('BlackScholes', False):
            compmodel = BlackScholes(self.derivative)
            comphedger = Hedger(compmodel, inputs=compmodel.inputs())
            comp = comphedger.compute_pnl(self.derivative, **self.profit_params)
            output['Black-Scholes'] = comp
        if self.benchmark_params.get('NoHedge',False):
            nohedger = Hedger(Naked(),inputs=["empty"])
            nohedge = nohedger.compute_pnl(self.derivative,**self.profit_params)
            output['No Hedge'] = nohedge
        return output
    def eval(self,pnl):
        return self.criterion(pnl).item()
    def dict_eval(self,dictionary):
        output = {}
        for key in dictionary.keys():
            output[key] = self.eval(dictionary[key])
        return output
    def full_process(self):
        if self.backup_file:
            print('Loading backup')
            folder = './Backup_Params'
            path = os.path.join(folder, self.backup_file)
            self.hedger.load_backup(path)
        else:
            history = self.fit()
            training_fig = make_training_diagram(history)
            training_fig.savefig('trainingdiagram.png')
            plt.close(training_fig)
        pnl = self.profit()
        bench = self.benchmark()
        print(self.eval(pnl))
        print(self.dict_eval(bench))
        pnl_fig = make_pl_diagram(pnl)
        pnl_fig.savefig('pldiagram.png')
        plt.close(pnl_fig)
        for key, value in bench.items():
            fig = make_pl_diagram(value)
            fig.savefig(f'pl{(key[0:2].lower())}')
            plt.close(fig)
    def stock_diagrams(self, number):
        self.derivative.simulate(number)
        prices = self.derivative.underlier.spot
        for i in range(number):
            fig = make_stock_diagram(prices[i,...])
            fig.savefig(f'stockdiagram{i}')
            plt.close(fig)

        
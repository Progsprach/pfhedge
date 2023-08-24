import yaml
from config_utils import make_underlier, make_derivative, make_hedge, make_model, make_criterion
from pfhedge.nn import Hedger
from utils import prepare_features
from model_utils import make_classical_model, make_quantum_model
from HedgeHandler import HedgeHandler
from MultiHandler import MultiHandler
class InputReader:
    def __init__(self, filename):
        stream = open(filename, mode="r", encoding="utf-8")
        self.config = yaml.safe_load(stream)
        stream.close()
    def is_multi(self) -> bool:
        return ('parameters' in self.config.keys())
    def load_config(self) -> HedgeHandler:
        underlier = make_underlier(self.config.get('underlier',{}))
        derivative = make_derivative(self.config.get('derivative',{}), underlier)
        hedge = make_hedge(self.config.get('hedge',{}),underlier)
        model = make_model(self.config.get('model',{}),len(hedge),derivative)
        loss = make_criterion(self.config.get('loss',{}))
        features = self.config.get('features', [])
        if features == []:
            NTB = self.config.get('model',{}).get('NTB',False)
            features = prepare_features(derivative,NTB)
        hedger = Hedger(model,features,loss)
        fit_params = self.config.get('training',{})
        profit_params = self.config.get('profit',{})
        criterion = make_criterion(self.config.get('criterion',{}))
        benchmark_params = self.config.get('benchmark',{})
        handler_params = self.config.get('handler', {})
        return HedgeHandler(hedger,derivative,hedge,fit_params,profit_params,criterion,benchmark_params, **handler_params)
    def load_multi_config(self) -> MultiHandler:
        underlier = make_underlier(self.config.get('underlier',{}))
        derivative = make_derivative(self.config.get('derivative',{}), underlier)
        hedge = make_hedge(self.config.get('hedge',{}),underlier)
        loss = make_criterion(self.config.get('loss',{}))
        features = self.config.get('features', [])
        quantum = self.config.get('model',{}).get('quantum',False)
        if features == []:
            features = prepare_features(derivative,False)
        param_nums = self.config.get('parameters',[1])
        def reduce_features(features, n, quantum=False):
            if n == 1 and not quantum:
                return ['empty']
            if n < len(features)+1 and not quantum:
                return features[:n-1]
            if n < len(features):
                return features[:n]
            return features
        mult = self.config.get('multiply',1)
        param_nums *= mult
        if not quantum:
            hedgers = [Hedger(make_classical_model(n,len(features)),reduce_features(features,n),loss)for n in param_nums]
        else:
            hedgers =  [Hedger(make_quantum_model(n,len(features)),reduce_features(features,n,True),loss)for n in param_nums]
        fit_params = self.config.get('training',{})
        profit_params = self.config.get('profit',{})
        criterion = make_criterion(self.config.get('criterion',{}))
        benchmark_params = self.config.get('benchmark',{})
        handlers = [HedgeHandler(hedger,derivative,hedge,fit_params,profit_params,criterion,benchmark_params) for hedger in hedgers]
        return MultiHandler(handlers, param_nums)

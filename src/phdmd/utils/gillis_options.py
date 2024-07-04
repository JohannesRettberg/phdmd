import numpy as np

class GillisOptions():
    def __init__(self,options=None):
        
        default_options = {
        "standard": 0,
        "maxiter": float("inf"),
        "timemax": 10,
        "posdef": 0,
        "PHform": {},
        "init": 1,
        "alpha0": 0.5,
        "lsparam": 1.5,
        "lsitermax": 20,
        "gradient": 0,
        "weight": np.ones(5),
        "display": 1,
    }
        self.gillis_options = default_options
        if options:
            self.gillis_options.update(options)

    def get_options(self):
        return self.gillis_options
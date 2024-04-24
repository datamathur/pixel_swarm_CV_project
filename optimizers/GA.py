from typing import Dict, List, Callable, Type, Iterable, Optional
import torch # type: ignore
from torch.optim import Optimizer # type: ignore
from .utils import _initialize_param_groups, clone_param_group, clone_param_groups

class GA(Optimizer):
    def __init__(self, 
                 params:Iterable[torch.nn.Parameter],
                 max_param_value: float = 10.,
                 min_param_value: float = -10.,
                 popsize: int = 8):
        self.popsize = popsize
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        defaults = {}
        super.__init__(params, defaults)
        self.population = [_initialize_param_groups(params, self.max_param_value, self.min_param_value) for _ in range(self.popsize)]
        self.scores = torch.rand(popsize)
        self.best_known_global_param_groups = clone_param_groups(self.param_groups)
        self.best_known_global_loss_value = torch.inf

    def crossoverPopulation(self, keep):
        def _pairSelection(population, popsize=self.popsize):
            p1 = population[torch.randint[0,popsize]].copy()
            p2 = population[torch.randint[0,popsize]].copy()
            return p1, p2

        def _crossover(crossoverLength, p1, p2):
            crossover_point = torch.randint(0, crossoverLength-1)
            off1 = None
            off2 = None
            return off1, off2
        newPopulation = self.population
        for i in range(keep, self.popsize, 2):
            p1, p2 = _pairSelection(self.population, self.scores, self.popsize)
            off1, off2 = _crossover()


    def mutatePolulation(self):
        def _mutation():
            pass
        pass

    def clearDups(self):
        pass

    def calculateScores(self, closure: Callable[[], torch.Tensor]):
        
        pass

    def sortPopulation(self, ):
        pass

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]):
        return closure()
    
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

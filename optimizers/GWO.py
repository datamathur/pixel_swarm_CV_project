from typing import Dict, List, Callable, Type, Iterable, Optional
import torch # type: ignore
from torch.optim import Optimizer # type: ignore
from .utils import _initialize_param_groups, clone_param_group, clone_param_groups
from random import randrange

class GWO(Optimizer):
    def __init__(self, 
                 params:Iterable[torch.nn.Parameter],
                 num_workers: int = 10,
                 max_param_value: float = 1.,
                 min_param_value: float = 0.,
                 lr: float = 0.002):
        self.num_workers = num_workers
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        super().__init__(params, defaults={})
        self.workers = [_initialize_param_groups(self.param_groups, max_param_value, min_param_value) for _ in range(num_workers)]
        self.alpha = _initialize_param_groups(self.param_groups, max_param_value, min_param_value)
        self.beta = _initialize_param_groups(self.param_groups, max_param_value, min_param_value)
        self.delta = _initialize_param_groups(self.param_groups, max_param_value, min_param_value)
        self.a_score, self.b_score, self.d_score = torch.inf, torch.inf, torch.inf
        self.scores = [torch.inf for _ in range(num_workers)]
        self.a = 2
        self.lr = lr
    
    def update(self, i: int):
        def _update(worker, feat):
            r1, r2 = randrange(0,1), randrange(0,1)
            a1, c1 = 2 * self.a * randrange(0,1) - self.a, 2*randrange(0,1)
            a2, c2 = 2 * self.a * randrange(0,1) - self.a, 2*randrange(0,1)
            a3, c3 = 2 * self.a * randrange(0,1) - self.a, 2*randrange(0,1)
            x = clone_param_group(worker[feat])['params']
            for f in range(len(worker[feat]['params'])):
                D1 = c1*self.alpha[feat]['params'][f] - worker[feat]['params'][f]
                D2 = c2*self.beta[feat]['params'][f] - worker[feat]['params'][f]
                D3 = c3*self.delta[feat]['params'][f] - worker[feat]['params'][f]
                x[f] = (a1*D1 + a2*D2 + a3*D3)/3
            return x
        
        new_pos = clone_param_groups(self.workers[i])

        for _ in range(len(self.workers[i])):
            x = _update(self.workers[i], feat=_)
            new_pos[_]['params'] = x

        return new_pos

    @torch.no_grad()
    def step(self, closure:Callable[[], torch.Tensor]):
        for i in range(self.num_workers):
            fitness = closure(self.workers[i])
            
            if fitness<self.a_score:
                self.d_score = self.b_score
                self.delta = self.beta.copy()
                self.b_score = self.a_score
                self.beta = self.alpha.copy()
                self.a_score = fitness
                self.alpha = self.workers[i]
            elif self.a_score<=fitness<self.b_score:
                self.d_score = self.b_score
                self.delta = self.beta.copy()
                self.b_score = fitness
                self.beta = self.workers[i]
            elif self.b_score<=fitness<self.d_score:
                self.d_score = fitness
                self.delta = self.workers[i]
            else:
                continue

        for i in range(self.num_workers):
            self.workers[i] = self.update(i)

        self.a -= self.lr
        return closure(self.alpha)
    
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
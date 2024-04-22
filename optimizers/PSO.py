from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, List, Callable, Type, Iterable, Optional
import torch
from torch.optim import Optimizer

def clone_param_group(param_group: Dict) -> Dict:
    """
    Clone each param in a param_group and return a new dict containing the clones
    :param param_group: Dict containing param_groups
    :return: cloned param_group dict
    """
    new_group = {key: value for key, value in param_group.items() if key != 'params'}
    new_group['params'] = [param.detach().clone() for param in param_group['params']]
    return new_group

def clone_param_groups(param_groups: List[Dict]) -> List[Dict]:
    """
    Make a list of clones for each param_group in a param_groups list.
    :param param_groups: List of dicts containing param_groups
    :return: cloned list of param_groups
    """
    return [clone_param_group(param_group) for param_group in param_groups]


def _initialize_param_groups(param_groups: List[Dict], max_param_value, min_param_value) -> List[Dict]:
    """
    Take a list of param_groups, clone it, and then randomly initialize its parameters with values between
    max_param_value and min_param_value.

    :param param_groups: List of dicts containing param_groups
    :param max_param_value: Maximum value of the parameters in the search space
    :param min_param_value: Minimum value of the parameters in the search space
    :return the new, initialized param_groups
    """
    magnitude = max_param_value - min_param_value
    mean_value = (max_param_value + min_param_value) / 2

    def _initialize_params(param):
        return magnitude * torch.rand_like(param) - magnitude / 2 + mean_value

    # Make sure we get a clone, so we don't overwrite the original params in the module
    param_groups = clone_param_groups(param_groups)
    for group in param_groups:
        group['params'] = [_initialize_params(p) for p in group['params']]

    return param_groups

class Particle:
    def __init__(self,
                 param_groups: List[Dict],
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.):
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        magnitude = abs(max_param_value - min_param_value)
        self.param_groups = param_groups
        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)
        self.velocity = _initialize_param_groups(param_groups, magnitude, -magnitude)
        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

    def step(self,  closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]):
        for position_group, velocity_group, personal_best, global_best, master in zip(self.position, 
                                                                                      self.velocity,
                                                                                      self.best_known_position,
                                                                                      global_best_param_groups,
                                                                                      self.param_groups):
            position_group_params = position_group['params']
            velocity_group_params = velocity_group['params']
            personal_best_params = personal_best['params']
            global_best_params = global_best['params']
            master_params = master['params']

            new_position_params = []
            new_velocity_params = []
            for p, v, pb, gb, m in zip(position_group_params, velocity_group_params, personal_best_params,
                                       global_best_params, master_params):
                rand_personal = torch.rand_like(v)
                rand_group = torch.rand_like(v)
                new_velocity = (self.inertial_weight * v
                                + self.cognitive_coefficient * rand_personal * (pb - p)
                                + self.social_coefficient * rand_group * (gb - p)
                                )
                new_velocity_params.append(new_velocity)
                new_position = p + new_velocity
                new_position_params.append(new_position)
                m.data = new_position.data 
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params
            for i in range(len(self.position)):
                for j in range(len(self.param_groups[i]['params'])):
                    self.param_groups[i]['params'][j].data = self.param_groups[i]['params'][j].data

        # Calculate new loss after moving and update the best known position if we're in a better spot
        new_loss = closure()
        if new_loss < self.best_known_loss_value:
            self.best_known_position = clone_param_groups(self.position)
            self.best_known_loss_value = new_loss
        return new_loss

class PSO(Optimizer):

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.):
        self.num_particles = num_particles
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        defaults = {}
        super().__init__(params, defaults)
        self.particles = [Particle(self.param_groups) for _ in range(num_particles)]
        self.best_known_global_param_groups = clone_param_groups(self.param_groups)
        self.best_known_global_loss_value = torch.inf

    @torch.no_grad()
    def step(self):
        pass
    pass
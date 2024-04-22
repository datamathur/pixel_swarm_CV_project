from typing import Dict, List, Callable, Type, Iterable, Optional
import torch # type: ignore
from torch.optim import Optimizer # type: ignore
from .utils import _initialize_param_groups, clone_param_group, clone_param_groups

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
    def step(self, closure: Callable[[], torch.Tensor]):
        # if particle_step_kwargs is None:
        #     particle_step_kwargs = {}
        for particle in self.particles:
            particle_loss = particle.step(closure, self.best_known_global_param_groups)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

        return closure()  # loss = closure()

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
    
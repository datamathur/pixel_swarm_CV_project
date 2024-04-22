from typing import Dict, List
import torch # type: ignore

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
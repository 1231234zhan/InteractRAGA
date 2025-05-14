#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

def load_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as file:
        opt = yaml.load(file, yaml.Loader)
    return opt

def save_yaml(yaml_path, data):
    import yaml
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def set_param_attribute(args_dict):
    group = GroupParams()
    for key in args_dict:
        value = args_dict[key]
        if isinstance(value, dict):
            value = set_param_attribute(value)
        setattr(group, key, value)
    return group

class GroupParams:
    pass

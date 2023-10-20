import yaml


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config

import yaml


def load_config(yaml_path):
    yaml_path = "/Users/jongbeomkim/Desktop/workspace/CLIP/config.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config

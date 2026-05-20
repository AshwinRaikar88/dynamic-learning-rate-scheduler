import yaml

def load_configuration(config_file_path="./configs/training_config.yaml"):
    with open(config_file_path, 'r') as config_file:
        _config = yaml.safe_load(config_file)

    training_config = _config['train']
    physics_config = _config['physics']
    model = _config['model']
    paths = _config['paths']

    print("Training configuration values")
    for key, value in training_config.items():
        print(f"{key} = {value}")
    print("\n")

    print("Physics configuration values")
    for key, value in physics_config.items():
        print(f"{key} = {value}")
    print("\n")

    print("Model configuration values")
    for key, value in model.items():
        print(f"{key} = {value}")
    print("\n")

    print("Paths")
    for key, value in paths.items():
        print(f"{key} = {value}")
    print("\n")

    return training_config, physics_config, model, paths


if __name__ == "__main__":
    tr_cfg, phy_cfg, _, _ = load_configuration(config_file_path="../configs/config_dirichlet.yaml")
    # print(config)

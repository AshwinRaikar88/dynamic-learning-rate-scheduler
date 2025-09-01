import yaml
import struct
import importlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_configuration(config_file_path="./configs/training_config.yaml", boundary='dirichlet'):
    with open(config_file_path, 'r') as config_file:
        _config = yaml.safe_load(config_file)

    return _config['physics'], _config['train'], _config['model'], _config['paths']


def get_last_epoch_params(log_dir, tag=''):
    # Initialize an empty dictionary to store parameter values
    last_epoch_params = {}

    # Create an EventAccumulator to read event data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()  # Reload the data from the log directory

    # Extract the available scalars
    tensors = event_acc.Tensors(tag)

    for tensor in tensors:
        # Access the data and properties of each tensor
        # tensor_data = tensor.tensor_proto.tensor_content
        # tensor_step = tensor.step
        last_epoch_params.update({tensor.step: tensor.tensor_proto.tensor_content
        })
    index = sorted(last_epoch_params, reverse=True)[0]
    
    return index, round(struct.unpack('f', last_epoch_params[index])[0], 8)


def get_pinn_module(boundary, model_type, **kwargs):
    w_name = ''
    class_name = ''

    if boundary == 'dirichlet_boundary' and model_type == 'gnn':
        w_name = 'pinn_acoustic'
        class_name = "PINN"
    elif (boundary == 'complex_boundary' or boundary == 'complex_medium') and model_type == 'complex':
        w_name = 'complex.ComplexNN'
        class_name = "PINN_CMPLX"

    try:
        module = importlib.import_module(f"pinn.{w_name}")
        model_class = getattr(module, class_name)
        return model_class(**kwargs)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Model {w_name} not found. Ensure the model is defined correctly.") from e


if __name__ == '__main__':
    physics_cfg, training_cfg, model_cfg, paths_config = load_configuration(
        config_file_path="../configs/config_dirichlet.yaml")
        # config_file_path="../configs/config_complex_boundary.yaml")

    PINN = get_pinn_module(physics_cfg['boundary_type'], model_cfg['class_name'])
    # PINN = get_pinn_module(physics_cfg['boundary_type'], model_cfg['class_name'], cw=physics_cfg['cw'])

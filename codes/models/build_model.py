import yaml
import copy


from codes.utils.utils import load_model, setting_device
from codes.models.cnn import CNN


def create_model(class_name, arch_name, arch_params, device):
    model = None

    if class_name == "CNN":
        model = CNN(
            arch_name,
            arch_params,
            device
        )
    else:
        raise ValueError(f"Unsupported model: {class_name}")


    return model
    

def build_model_from_file(model_params):
    device = setting_device(model_params.get('device'))
    class_name = model_params.get('class')
    arch_name = model_params.get('name')
    model_path = model_params.get('model')
    trained_weights = model_params.get('save_trained_weights')

    with open(model_path, 'r') as file_option:
        arch_params = yaml.safe_load(file_option).get('arch').get(arch_name)

    skeleton_model = create_model(
        class_name,
        arch_name,
        arch_params,
        device
    )

    load_model(skeleton_model.model, trained_weights)

    return skeleton_model
import os
import logging
import torch


def format_size(size):
    # 对总参数量做格式优化
    K, M, B = 1e3, 1e6, 1e9
    if size == 0:
        return '0'
    elif size < M:
        return f"{size / K:.1f}K"
    elif size < B:
        return f"{size / M:.1f}M"
    else:
        return f"{size / B:.1f}B"


def print_model_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Shape: {param.size()}")
        print(f"Type: {param.dtype}")
        print(f"Trainable: {param.requires_grad}")
        print("----------")


def get_pytorch_model_info(model: torch.nn.Module) -> (dict, list):
    """
    输入一个PyTorch Model对象，返回模型的总参数量（格式化为易读格式）以及每一层的名称、尺寸、精度、参数量、是否可训练和层的类别。

    :param model: PyTorch Model
    :return: (总参数量信息, 参数列表[包括每层的名称、尺寸、数据类型、参数量、是否可训练和层的类别])
    """
    params_list = []
    total_params = 0
    total_params_non_trainable = 0

    for name, param in model.named_parameters():
        # 获取参数所属层的名称
        layer_name = name.split('.')[0]
        # 获取层的对象
        layer = dict(model.named_modules())[layer_name]
        # 获取层的类名
        layer_class = layer.__class__.__name__

        params_count = param.numel()
        trainable = param.requires_grad
        params_list.append({
            'tensor': name,
            'layer_class': layer_class,
            'shape': str(list(param.size())),
            'precision': str(param.dtype).split('.')[-1],
            'params_count': str(params_count),
            'trainable': str(trainable),
        })
        total_params += params_count
        if not trainable:
            total_params_non_trainable += params_count

    total_params_trainable = total_params - total_params_non_trainable

    total_params_info = {
        'total_params': format_size(total_params),
        'total_params_trainable': format_size(total_params_trainable),
        'total_params_non_trainable': format_size(total_params_non_trainable)
    }

    return total_params_info, params_list


def get_next_log_filename(log_folder):
    i = 0
    while True:
        log_filename = os.path.join(log_folder, f"log{i}.txt")
        if not os.path.exists(log_filename):
            return log_filename
        i += 1


def configure_logging(log_filename):
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

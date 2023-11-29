import torch


from random import shuffle, seed as rand_seed

from numpy.random import seed as np_seed
from torch import manual_seed
from torch.cuda import manual_seed_all
from torch.mps import manual_seed as mps_manual_seed

from csv import writer as wr

def shuffle_data(data: list):
    """
    Функция shuffle_data переставляет случайным образом элементы входного списка data.

    Аргументы:
    - data (list): список, содержащий элементы, которые нужно переставить случайным образом.

    Возвращаемое значение:
    - Функция не возвращает значения, но напрямую изменяет порядок элементов в списке data.
    """
    shuffle(data)


def build_csv(data: list, path: str, type: str):
    """
    Функция _create_csv создает файл CSV и записывает в него данные.

    Аргументы:
    - data: список, содержащий данные, которые нужно записать в файл CSV.
    - path: строка, представляющая путь к каталогу, в котором будет создан файл CSV.
    - type: строка, представляющая имя файла CSV.

    Функция открывает файл с заданным путем и именем в режиме записи ('w') и создает объект writer, используя модуль csv. Затем функция записывает данные в файл, используя метод writerows объекта 
    """
    with open(f'{path}/{type}.csv', 'w', newline='') as file:
        writer = wr(file)
        writer.writerows(data)


def split_data(data: list, data_param: dict):
    """
    Функция для разбиения данных на обучающий, проверочный и тестовый наборы.

    Параметры:
    - data: Список данных, которые необходимо разделить.
    - data_param: словарь, содержащий параметры для разбиения данных.

    Возвращает:
    - test_data: Список, содержащий тестовые данные.
    - val_data: Список, содержащий данные для проверки.
    - train_data: Список, содержащий обучающие данные.
    """
    size = len(data)

    test_size = int(size * data_param.get('test_size'))
    val_size = int(size * data_param.get('val_size'))

    return data[:test_size], data[test_size: test_size + val_size], data[test_size + val_size:]


def setting_seed(seed: int):
    """
    Функция setting_seed используется для установления начального значения генератора случайных чисел.
    Это позволяет получать одинаковые случайные числа при каждом запуске программы, если используется один и тот же seed.

    Аргументы:
    - seed (int): начальное состояние генератора.
    """
    rand_seed(seed)
    np_seed(seed)
    manual_seed(seed)
    manual_seed_all(seed)
    mps_manual_seed(seed)


def setting_device(device_name: str):
    """
    Функция setting_device используется для определения устройства, на котором будет выполняться вычисление. 

    Аргументы:
    - device_name: название девайса.
        
    Возвращаемое значение:
    - torch.device: объект torch.device, представляющий заданное устройство.
    """

    if device_name == 'cuda':
        if not torch.cuda.is_available():
            print(f"CUDA is not available. Setting device to cpu.")
            device_name = 'cpu'
    elif device_name == 'mps':
        if not torch.backends.mps.is_available():
            print(f"MPS is not available. Setting device to cpu.")
            device_name = 'cpu'
    elif device_name == 'cpu':
        ...
    else:
        print(f"Unsupported device: {device_name}. Setting device to cpu.")
        device_name = 'cpu'
        
    return torch.device(device_name)
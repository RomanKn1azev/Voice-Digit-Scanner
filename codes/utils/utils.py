import torch
import matplotlib.pyplot as plt
import librosa
import numpy as np
import yaml


from random import shuffle, seed as rand_seed, choices, random, sample as rn_sample

from numpy.random import seed as np_seed
from torch import manual_seed
from torch.cuda import manual_seed_all
from torch.mps import manual_seed as mps_manual_seed

from csv import writer as wr
from tqdm import tqdm

from codes.data.csv_reader import CsvReader


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


def random_choice(list: list):
    """
    Функция 'random_choice' принимает список и возвращает случайно выбранный элемент из этого списка.

    Аргументы:
    - list: list - обязательный аргумент, представляющий список элементов.

    Возвращаемое значение:
    Функция возвращает случайно выбранный элемент из списка.
    """
    return choices(list, k=1)[0]


def dict_to_str_csv(param: dict):
    """
    Функция dict_to_str_csv принимает в качестве параметра словарь param 
    и выполняет преобразование словаря в строку в формате CSV.
    Каждая пара ключ-значение словаря представляется в виде ключ=значение.

    Параметры:
    - param: словарь, который нужно преобразовать в строку формата CSV.

    Возвращаемое значение:
    - Функция возвращает строку, полученную путем преобразования словаря param в формат CSV.
    """
    return ";".join(f"{key}={value}" for key, value in param.items())


def str_csv_to_dict(param: str):
    """
    Функция str_csv_to_dict преобразует строку в формате CSV в словарь.

    Аргумент:
    - param (строка): Строка в формате CSV, где каждая пара ключ=значение разделена точкой с запятой (;).

    Возвращаемое значение:
    - Функция возвращает словарь, полученную путем преобразования строки param.
    """
    return dict(map(lambda x: x.split("="), param.split(";")))


def evaluate(model, dl, device, loss_func=torch.nn.CrossEntropyLoss()): 
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        
        with torch.no_grad():
            for X, y in tqdm(dl):
                X, y = X.to(device), y.to(device)

                outputs = model(X)

                loss = loss_func(outputs, y.long())

                running_loss += loss.item()

                _, prediction = torch.max(outputs, 1)
                correct_prediction += (prediction == y).sum().item()
                total_prediction += prediction.shape[0]

        num_batches = len(dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction

        return acc, avg_loss


def predict(model, data, device):
    with torch.no_grad():
        for X in data:
            X = X.to(device)
            outputs = model(X)

            _, prediction = torch.max(outputs, 1)
    
    return prediction.numpy()[-1]



def save_model(model_dict, path):
    torch.save(model_dict, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def plot_schedule_losses(train_losses, val_lossses, path):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, range(len(val_lossses)), val_lossses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(path)


def plot_schedule_accuracy(train_accuracy, val_accuracy, path):
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, range(len(val_accuracy)), val_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.savefig(path)


def build_data_csv(path, label, probability, classes, types_aug):   
    if int(label) in classes and random() < probability:
        choice_aug = random_choice(types_aug)

        return [
            path, 
            label,
            choice_aug.get('name'),
            dict_to_str_csv(choice_aug.get('param'))
        ]

    return [path, label, None, None]


def save_melspectrograms_dB_settings(file="melspectrograms_dB_settings/1.npy"):
    files = ["my_data/2/2_01.wav", "my_data/8/8_03.wav", "data/51/1_51_46.wav"]

    melspectrograms_dB = []
    for f in files:
        y, sr = librosa.load(f, sr=48_000)
        segment_pad = np.zeros((48_000,))
        segment_pad[:len(y)] = y

        mel_spectrogram_dB = build_melspectrogram_dB(segment_pad, sr)
        melspectrograms_dB.append(mel_spectrogram_dB)

    np.save(file, melspectrograms_dB)


def load_melspectrograms_dB_settings(file="melspectrograms_dB_settings/1.npy"):
    return np.load(file)


def build_melspectrogram_dB(audio, sample_rate, n_mels=128, fmax=4096, ref=np.max):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmax=fmax)

    return librosa.power_to_db(mel_spectrogram, ref=ref)


def build_mel_from_file(file, sample_rate=48_000):
    y, sr = librosa.load(file, sr=sample_rate)
    mel_spectrogram_dB = None

    if len(y) < sr:
        segment_pad = np.zeros((48_000,))
        segment_pad[:len(y)] = y

        mel_spectrogram_dB = build_melspectrogram_dB(segment_pad, sample_rate)
        
    else:
        acceleration = len(y) / sr
        audio_stretched = librosa.effects.time_stretch(y, rate=acceleration)  

        mel_spectrogram_dB = build_melspectrogram_dB(audio_stretched, sample_rate)

    return [mel_spectrogram_dB]


def plot_spectra(refer, num, paths):
    df = CsvReader(paths).df

    unique_labels = df['Label'].unique()

    for label in unique_labels:
        label_data = df[df['Label'] == label]

        random_indices = rn_sample(range(len(label_data)), num)
        random_samples = label_data.iloc[random_indices]

        i = 1
        for _, sample in random_samples.iterrows():
            audio_path = sample['Path']
            audio, sr = librosa.load(audio_path, sr=48_000)

            plt.figure(figsize=(10, 4))
            spectrogram = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
            librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram ({label})')
            plt.tight_layout()
            plt.savefig(f"{refer}/{label}/{label}_{i}.png")

            i += 1
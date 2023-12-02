import os
import csv
import random


from typing import List
from codes.utils.utils import shuffle_data, build_csv, split_data, random_choice, dict_to_str_csv


class DataToCsv:
    def __init__(self, datas_param: List[dict]):
        self.datas_param = datas_param

    def _create_dirs_params(self):
        """
        Функция перебирает список datas_param и проверяет, равно ли значение ключа 'type' в каждом словаре значению "dir". 
        
        Аргументы:
        - self (object): объект, к которому принадлежит функция. 
        
        Возвращает: 
        - dirs_params: список параметров всех данных представленых в виде директорий с данными.
        """
        dirs_params = []

        for data_param in self.datas_param:
            if data_param.get('type') == "dir":
                dirs_params.append(data_param)

        return dirs_params
    
    def create_csv(self):
        """
        Резюме:
        Этот код подготавливает данные для машинного обучения, 
        создавая CSV-файлы, содержащий пути к файлам и соответствующие метки и аугментации.

        Метод создает CSV-файлы, рекурсивно перебирая все файлы и подкаталоги в указанных каталогах.
        Для каждого файла, если он имеет расширение .wav, 
        в список добавляется путь к файлу и метка (полученная из имени файла), и случайная аугментация для определенных классов.

        Аргументы:
        - self (object): объект, к которому принадлежит функция. 
        
        Данные записываются в CSV-файлы.
        """
        dirs_params = self._create_dirs_params()
        
        # Проходимся рекурсивно по всем файлам и поддиректориям в указанной директории
        for dir_params in dirs_params:
            curr_root = dir_params.get('path')
            augmentation = dir_params.get('augmentation')
            classes = augmentation.get('classes')
            type_aug = augmentation.get('augmentation')

            data = []
            for root, dirs, files in os.walk(curr_root):
                for file_name in files:
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(root, file_name)
                        
                        # Разделяем название файла на метку и остаток
                        label, _ = os.path.splitext(file_name)
                        label_rest = label.split('_')[0]
                        
                        # Добавляем путь, метку, аугментация в список данных
                        data.append([file_path, label_rest, None, None])

                        for label_rest in classes:
                            choice_aug = random_choice(type_aug)
                            data.append(
                                [
                                    file_path,
                                    label_rest,
                                    choice_aug.get('name'),
                                    dict_to_str_csv(choice_aug.get('param'))
                                ]
                            )
                        
            
            if dir_params.get('shuffle'):
                shuffle_data(data)
            
            path_to_csv = dir_params.get("path_to_csv")

            if dir_params.get('split'):
                test_data, val_data, train_data = split_data(data, dir_params)
                build_csv(data=train_data, path=path_to_csv, type="train")
                build_csv(data=val_data, path=path_to_csv, type="val")
                build_csv(data=test_data, path=path_to_csv, type="test")
            else:
                build_csv(data=data, path=path_to_csv, type="train")
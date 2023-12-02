import pandas as pd


from typing import List


class CsvReader:
    def __init__(self, paths: List[str]) -> None:
        self.df = self._read_paths(paths)
    
    def _read_paths(self, paths) -> pd.DataFrame:
        """
        Функция _read_paths представляет собой метод класса,
          который считывает данные из CSV файлов, указанных в paths, и возвращает их в виде объекта pd.DataFrame.

        Аргументы:
        - paths (list) - список путей к CSV файлам, откуда необходимо считать данные.

        Возвращаемое значение:
        - объект pd.DataFrame, содержащий данные из всех указанных CSV файлов.
        """
        df = pd.DataFrame()

        for path in paths:
            curr_data = pd.read_csv(
                path,
                header=None,
                names=['Path', 'Label', 'Name_Augmentation', 'Param_Augmentation'],
                delimiter=','
                )
            df = pd.concat([df, curr_data])
        
        return df
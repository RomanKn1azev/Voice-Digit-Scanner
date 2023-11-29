from codes.data.data_preparation import DataToCsv
from codes.utils.utils import setting_seed, setting_device


class Setup:
    def __init__(self, config: dict):
        self.config = config
        setting_seed(self.config.get('random_seed', 0))
        self.device = setting_device(self.config.get('device', 'cpu'))

    def run_tasks(self):
        tasks = self.config.get('tasks')

        if tasks.get('train'):
            DataToCsv(tasks.get('train').get('data')).create_csv()

        elif tasks.get('test'):
            ...
        elif tasks.get('predict'):
            ...
        else:
            raise ValueError("Unsupported tasks")

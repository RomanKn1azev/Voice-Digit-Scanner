import yaml


from codes.data.data_preparation import DataToCsv
from codes.data.dataset import BuildDataLoader
from codes.utils.utils import setting_seed, setting_device, save_model, plot_schedule_accuracy, plot_schedule_losses
from codes.models.cnn_train import CNN_Trainer


class Setup:
    def __init__(self, config: dict):
        self.config = config
        setting_seed(self.config.get('random_seed', 0))
        self.device = setting_device(self.config.get('device', 'cpu'))

    def run_tasks(self):
        tasks = self.config.get('tasks')

        if tasks.get('train'):
            if tasks.get('train').get('data', "don't exist") != "don't exist":
                DataToCsv(tasks.get('train').get('data')).create_csv()

            paths_to_dataset = tasks.get('train').get('dataset').get('paths')
            dataloader_params = tasks.get('train').get('dataloader')

            train_dataloader = BuildDataLoader(paths_to_dataset.get('train'), dataloader_params).dataloader
            val_dataloader = BuildDataLoader(paths_to_dataset.get('val'), dataloader_params).dataloader
            test_dataloader = BuildDataLoader(paths_to_dataset.get('test'), dataloader_params).dataloader
            
            with open(tasks.get('train').get('model'), 'r') as file_option:
                model_params = yaml.safe_load(file_option).get('arch')
            
            cnn_trainer = CNN_Trainer('lite', model_params.get('lite'), self.device, int(len(train_dataloader)))
            losses, val_lossses, accuracy, val_accuracy = cnn_trainer.train(
                train_dataloader, val_dataloader
            )

            save_model(cnn_trainer.model.state_dict(), tasks.get('train').get('results').get('model_object'))

            path_plots = tasks.get('train').get('results').get('plots')

            plot_schedule_losses(losses, val_lossses, path_plots.get('loss'))
            plot_schedule_accuracy(accuracy, val_accuracy, path_plots.get('accuracy'))
            
        elif tasks.get('test'):
            ...
        elif tasks.get('predict'):
            ...
        else:
            raise ValueError("Unsupported tasks")

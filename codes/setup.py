import yaml


from codes.data.data_preparation import DataToCsv
from codes.data.dataset import BuildDataLoader
from codes.utils.utils import setting_seed, plot_spectra, setting_device, save_model, load_model, plot_schedule_accuracy, plot_schedule_losses, evaluate
from codes.models.cnn import CNN


class Setup:
    def __init__(self, config: dict):
        self.config = config
        setting_seed(self.config.get('random_seed', 0))
        self.device = setting_device(self.config.get('device', 'cpu'))

    def run_tasks(self):
        tasks = self.config.get('tasks')

        if tasks.get('data_to_csv'):
            data_to_csv = tasks.get('data_to_csv')
            
            if data_to_csv.get('data', "don't exist") != "don't exist":
                DataToCsv(data_to_csv.get('data')).create_csv()

        if tasks.get('train'):
            train = tasks.get('train')

            paths_to_dataset = train.get('dataset').get('paths')
            dataloader_params = train.get('dataloader')

            train_dataloader = BuildDataLoader(paths_to_dataset.get('train'), dataloader_params).dataloader
            val_dataloader = BuildDataLoader(paths_to_dataset.get('val'), dataloader_params).dataloader
            test_dataloader = BuildDataLoader(paths_to_dataset.get('test'), dataloader_params).dataloader

            with open(train.get('model'), 'r') as file_option:
                model_params = yaml.safe_load(file_option).get('arch')

            name_arch = train.get('name_arch')
            
            cnn = CNN(name_arch, model_params.get(name_arch), self.device, int(len(train_dataloader)))
            losses, val_lossses, accuracy, val_accuracy = cnn.train(
                train_dataloader, val_dataloader
            )

            save_model(cnn.model.state_dict(), train.get('results').get('model_object'))

            plots = train.get('plots')

            plot_schedule_losses(losses, val_lossses, plots.get('loss'))
            plot_schedule_accuracy(accuracy, val_accuracy, plots.get('accuracy'))

            print("Test accuacy: %.2f, Test loss: %.3f" % (evaluate(cnn.model, test_dataloader, self.device)))
            
            
        if tasks.get('test'):
            test = tasks.get('test')
            paths_to_dataset = test.get('dataset').get('paths')
            dataloader_params = test.get('dataloader')

            test_dataloader = BuildDataLoader(paths_to_dataset, dataloader_params).dataloader

            arch = test.get('arch')
            name_arch = arch.get('name')
            
            with open(arch.get('refer'), 'r') as file_option:
                model_params = yaml.safe_load(file_option).get('arch').get(name_arch)

            cnn = CNN(name_arch, model_params, self.device)

            path_to_save_model = test.get('save_model')

            load_model(cnn.model, path_to_save_model)

            print("Test accuacy: %.2f, Test loss: %.3f" % (evaluate(cnn.model, test_dataloader, self.device)))

        if tasks.get('plot_spectra'):
            plot_spectra_params = tasks.get('plot_spectra')
            refer = plot_spectra_params.get('refer')
            num = plot_spectra_params.get('num')
            paths = plot_spectra_params.get('paths')

            plot_spectra(refer, num, paths)
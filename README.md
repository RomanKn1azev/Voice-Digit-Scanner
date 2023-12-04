# Voice-Digit-Scanner

[![Python Version](https://img.shields.io/badge/python-3-informational?style=flat)](https://python.org)


Voice-Digit-Scanner это приложение с открытым исходным кодом для идентификации произнесенных цифр, основанный на PyTorch.

Код находится в состоянии постоянного изменения, поэтому , поэтому если вы обнаружили проблему или ошибку, пожалуйста, откройте [вопрос](https://github.com/RomanKn1azev/Voice-Digit-Scanner/issues).


## Оглавление
1. [Зависимости](#зависимости)
2. [Использование](#использование)
3. [Наборы данных](#наборы-данных)


### Зависимости

-   Python 3.9 (Рекомендуем к использованию [Anaconda](https://www.anaconda.com/download/#linux))
-   [PyTorch >= 0.4.0](https://pytorch.org/). PyTorch >= 1.7.0 необходимые для включения определенных функций, и также [torchvision](https://pytorch.org/vision/stable/index.html).
-   NVIDIA CPU | MPS | GPU + [CUDA] (https://developer.nvidia.com/cuda-downloads)
- Файлы `JSON` могут использоваться для файлов опций конфигурации, но для использования `YAML` необходимо также установить зависимость от python-пакета `PyYAML`: [`pip install PyYAML`](https://pyyaml.org/)
- Библиотека обработки аудио: `pip install librosa`
- Остальные зависимости с текущими версиями находятся в файле environment.yml


### Использование
С помощью sh sh/main.sh запускается скрипт для работы с данными, обучения, тестирования. Для работы с режимами надо использовать config файл. Базовая структура изложена в config/config_cnn.yml.

При использовании sh sh/app.sh запускается скрипт для работы с приложением.


## Наборы данных
На этой странице представлен некоторый стандарт [наборы-данных](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist), используемый для обучения моделей.

Данные дополнены еще одиним спикером. По 20 записей для каждой цифры.
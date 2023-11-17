## Tensorboard Logger (tb_logger)

Помимо протоколирования результатов в файлы, [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) является дополнительным инструментом визуализации для визуализации/сравнения потерь при обучении, метрик валидации и даже изображений валидации.

Включить/выключить его можно в файле опций с помощью ключа: `use_tb_logger`.

### Установка

Если опция установлена, то автоматически будет работать либо официальный tensorboard (`pip install tensorboard`), либо tensorboardX (`pip install tensorboardX`).

### Запуск
1. В терминале откройте tensorboard, направив его в директорию с выводами вашего эксперимента в этой директории ([tb_logger]()), как: `tensorboard --logdir xxx/xxx`.
2. Откройте в браузере пользовательский интерфейс tensorboard по адресу http://localhost:6006.
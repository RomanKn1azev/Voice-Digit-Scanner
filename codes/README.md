# Code

Общая структура кода представлена на следующем рисунке. В основном он состоит из частей - [Config], [Data],[DataOps], [Model] и [Network].

<p align="center">
   <img src="" height="450">
</p>

Например, после вызова команды [train.py] (т.е. `python train.py -opt options/sr/train_sr.yaml`) последует последовательность действий:

1.  [Config] - считывает конфигурацию из файла [/options/sr/train_sr.yaml], который представляет собой файл [YAML]. Затем передает значения конфигурации из него вниз через следующие шаги.
2.  [Data] - Создает загрузчики данных обучения и проверки.
3.  [DataOps] - содержит большинство операций с данными, связанных с различными вариантами дополнения и функциями, используемыми в загрузчиках данных, потерях и других компонентах.
4.  [Model] - Создает выбранную модель.
5.  [Сеть] - Создает выбранную сеть.
6.  Наконец, [train.py] - запускает обучение модели. В процессе обучения выполняются и другие действия, такие как протоколирование, сохранение промежуточных моделей, валидация, обновление скорости обучения и т.д.

Кроме того, для выполнения различных операций, например, настройки набора данных, доступны [Utilities](#utils) и [Useful script](#scripts).

[train.py]: https://github.com/RomanKn1azev/Voice-Digit-Scanner/blob/main/codes/train.py

[/options/sr/train_sr.yaml]: https://github.com/RomanKn1azev/Voice-Digit-Scanner/blob/main/codes/options/sr/train_sr.yaml

[/options]: https://github.com/RomanKn1azev/Voice-Digit-Scanner/blob/main/codes/options

[Config]: #config

[Data]: #data

[DataOps]: #dataops

[Model]: #model

[Network]: #network

[YAML]: https://yaml.org
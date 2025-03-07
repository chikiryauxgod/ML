# ИПР-22-1Б Семенов Лев

## Описание задачи
Поставил задачу дообучить модель YOLO на заданном датасете для определения животных (кошек, собак, кроликов). В процессе работы:

- Использовал несколько датасетов, но столкнулся с проблемами перевода описаний некоторых из них в формат YOLO.
- В итоге остановился на одном датасете, ссылка на него приведена в конце описания.
- Аннотации решил составлять самостоятельно.

## Процесс работы
Для составления описаний изображений:
1. Использовал сервер **CVAT**.
2. Поднял локальный сервер с помощью **Docker-Compose**.
3. Процесс разметки оказался трудоемким, так как задача была учебной и не предполагала высокой точности.

### Разделение датасета
- Было решено отказаться от стандартного разделения датасета (например, 20/80 или 33.3/66.6).
- Использовал лишь 1/3 от объема исходного датасета.

## Валидация
После обучения модели провел валидацию на:
- Фото.
- Видео (разделил на кадры).

### Результаты
- Модель не совершенна, что связано с большим разнообразием пород собак, которые могли быть не представлены в датасете.
- Некоторые собаки определяются некорректно или вообще не определяются.

## Полезные ссылки
- [Использованный датасет](https://huggingface.co/datasets/rokmr/pets)
- [Скрипт для конвертации CVAT в YOLO](https://github.com/ankhafizov/CVAT2YOLO)

### Примечание
Конвертация из формата CVAT в YOLO выполнена через скрипт **CVAT2YOLO**, используя окружение **conda**.

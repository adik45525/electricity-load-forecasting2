# Источник и распространение данных

Источник: [Nicholas Jhana / Kolasniwash — Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

На странице Kaggle и в официальных метаданных `/api/v1/datasets/view/nicholasjhana/energy-consumption-generation-prices-and-weather` указана **CC0: Public Domain**. Проверено 11 октября 2026 года. [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) допускает копирование, изменение и распространение. Ссылка на источник сохранена; проект не подразумевает одобрение авторами или операторами сети.

Публикуются предоставленные CSV `data/raw/energy_dataset.csv`, `data/raw/weather_features.csv`, их обработанная версия `data/processed/hourly.csv` и результаты исследования. Данные имеют национальный/городской уровень агрегирования и не содержат персональных записей.

Автор датасета указывает ENTSO-E и REE как источники энергии, OpenWeather — как источник погоды. Мы распространяем именно версию датасета, помеченную CC0 на Kaggle, а не заявляем общую лицензию на все данные этих организаций.

Контрольные суммы исходных файлов находятся в `reports/data_quality.json`. Полная карточка Kaggle не включается в репозиторий; публикуется только компактная запись подтверждения лицензии `reports/dataset_license.json`.

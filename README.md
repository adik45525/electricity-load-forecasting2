# GridScope — Electricity Load Forecasting

Прогноз почасовой нагрузки электросистемы Испании на 24 часа. Реальные данные, временная валидация, сравнение Seasonal Naive / Ridge / Random Forest / CatBoost, ablation и интерактивный dashboard на русском языке.

## Публикация

Целевой GitHub: [adik45525/electricity-load-forecasting2](https://github.com/adik45525/electricity-load-forecasting2).

Публичная ссылка Streamlit ещё не получена: загрузка и облачная проверка ожидают входа владельца репозитория. Этот статус не означает успешного развёртывания. Для Community Cloud выберите `main`, точку входа `app/streamlit_app.py` и **Python 3.12** в Advanced settings. Секреты не нужны.

Зависимости облачного приложения зафиксированы в корневом `requirements.txt`. Обучение при открытии сайта не запускается. Веб-приложение использует `models/best_model.cbm` + `best_model.json`, обработанную историю и сохранённые результаты. CatBoost-модель экспортирована из исходного обученного объекта, без изменения параметров или метрик.

Для подготовки пакета: `python scripts/prepare_publication.py`. Он создаёт `publication/` с проверенным набором файлов и ZIP-архивом; `.venv`, кеши, секреты и остальные тяжёлые pickle-модели туда не входят. Файлы данных распространяются согласно [DATA_LICENSE.md](DATA_LICENSE.md), лицензия Kaggle — CC0.

Инструкция и состояние развёртывания: [reports/deployment.md](reports/deployment.md).

## Быстрый запуск

Нужен Python 3.12 (проверенная локальная версия). В текущем проекте окружение `.venv` уже установлено, модели обучены и артефакты сохранены.

```bash
cd "/Users/anel/Desktop/Electricity Load Forecasting"
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Откройте http://localhost:8501. Если порт занят, добавьте `--server.port 8502`.

## Воспроизведение с нуля

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
python -m src.train
pytest -q
streamlit run app/streamlit_app.py
```

`requirements.lock.txt` содержит точные версии проверенного окружения. `requirements.txt` — точные версии для облачного inference; `requirements-training.txt` добавляет пакеты обучения и тестов; для воспроизведения сохранённых pickle-моделей используйте lock. Обучение фиксирует seed=42, выполняется на CPU в 4 потоках. Точные результаты могут незначительно зависеть от платформы. Повторное обучение перезаписывает артефакты в `models/`, `data/processed/` и `reports/`.

## Данные

Исходные файлы уже скопированы в `data/raw/`: `energy_dataset.csv`, `weather_features.csv`. Источник — [Nicholas Jhana / Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). Период 2015–2018, почасовая национальная нагрузка в МВт и погода Barcelona, Bilbao, Madrid, Seville, Valencia. SHA-256 источников и аудит находятся в `reports/data_quality.json`.

Пропущенные цели не заполняются. История и погода заполняются только из прошлого максимум на 24 часа. Физически недопустимые погодные значения удаляются, повторы city/time усредняются. UTC используется для хранения и блоков, Europe/Madrid — для календаря. Погода городов агрегируется равными весами.

## Методология

Origin 00:00 UTC, один прогноз на 24 следующих часа. Лаги t−24/t−48/t−168/t−336 известны для всего блока. Rolling-статистики фиксируются до origin. Прошлая погода t−24 также известна. Фактическая будущая погода используется только в явно помеченном `weather_oracle` и не допускается к выбору основной модели.

Train 2015–2016 → validation 2017 → test 2018. Два набора параметров для каждого алгоритма/сценария подбираются по validation MAE. Победитель среди non-oracle моделей также выбирается по validation. После этого параметры фиксируются и модели переобучаются на train+validation. Случайное разделение и признаки цены/генерации/внешнего прогноза не используются.

Сценарии: calendar; calendar+history; calendar+history+past weather; calendar+history+oracle weather. Seasonal Naive повторяет предыдущую неделю. Все сравнения используют одинаковые полные наблюдаемые блоки. MAE/RMSE в МВт, MAPE/sMAPE в процентах.

## Что открывать на защите

- **Обзор:** нагрузка, сезонный период, профиль недели, качество прогноза.
- **Прогноз:** выберите историческую дату test, загруженная модель рассчитает 24 часа; доступен CSV.
- **Аналитика данных:** погода, температурные остатки календарной модели, распределения, корреляции.
- **Качество моделей:** все алгоритмы и сценарии, test/validation, ошибки по часу, месяцу и горизонту.
- **Интерпретация:** permutation importance, CatBoost SHAP, локальные вклады.
- **О проекте:** бизнес-задача, методология, качество, ограничения, отчёт.

## Структура

```text
app/streamlit_app.py       интерфейс и навигация
app/components/style.html дизайн
src/data_loader.py        проверка и загрузка CSV
src/preprocessing.py      очистка и объединение
src/feature_engineering.py календарь, лаги, frozen rolling
src/train.py              валидация, refit, сохранение экспериментов
src/evaluate.py           метрики, bootstrap и ошибки
src/predict.py            загрузка модели и исторический inference
src/analysis.py           EDA, графики и генерация отчёта
notebooks/                исследовательский notebook
models/                   переносимая best_model.cbm + метаданные; остальные модели хранятся локально
reports/                  метрики, прогнозы, SHAP, аудит, отчёты
reports/figures/           графики PNG для защиты
tests/                    leakage, DST, метрики, модели и страницы
```

## Проверенный результат

Выбрана по validation CatBoost / calendar + history + past weather. На 8 688 test-часах (362 блока) MAE 1 533.54 МВт, RMSE 2 293.79 МВт, MAPE 5.247%, sMAPE 5.256%. Seasonal Naive MAE 2 647.97 МВт: улучшение 42.1%. Погода не дала устойчивого дополнительного выигрыша к истории: test-разница MAE +4.27 МВт, 95% парный bootstrap [-28.43; 40.15].

Проверено: 29 pytest-тестов, отсутствие leakage/DST-ошибок, повторная загрузка модели, все шесть страниц и основные взаимодействия Streamlit. Браузерная проверка проведена при ширине 1440 и 390 пикселей; горизонтального переполнения документа нет. Подробнее: `reports/verification.md`.

## Основные артефакты

Результат выбора и итоговые метрики — `reports/summary.json`. Полная таблица — `metrics.csv`; подбор — `tuning.json`; прогнозы — `predictions.csv`; ошибки — `error_breakdown.csv`; пики — `daily_peaks.csv`. Исследовательский отчёт — `final_report.md`; сценарий защиты — `presentation_notes.md`; промежуточный материал — `interim_report.md`; соответствие критериям — `criteria_checklist.md`. Презентация со слайдами по запросу не создавалась.

## Notebook

Основной код исследования — src/analysis.py. Для интерактивного notebook при необходимости установите `pip install jupyterlab` и откройте `notebooks/research.ipynb` через Jupyter в окружении проекта. Jupyter не нужен для обучения и dashboard.

## Docker

```bash
docker build -t gridscope .
docker run --rm -p 8501:8501 gridscope
```

Dockerfile подготовлен; Docker-запуск отдельно не проверен. Для сохранённых моделей контейнер устанавливает точные версии `requirements.lock.txt`. Raw CSV и модели включаются в образ.

## Ограничения

Это историческая демонстрация, не текущий прогноз. Нет свежей нагрузки, архива прогнозов погоды, моделирования задержек публикации, региональных праздников и весов населения. Данные заканчиваются в 2018 году; нужны новые данные и rolling backtests. Только полные блоки входят в оценку; исключение пропусков может смещать её. Bootstrap по дням не полностью учитывает автокорреляцию. SHAP/permutation рассчитаны на систематической выборке одного часа каждого UTC-дня и не доказывают причинность. При production-развитии нужны прогнозные интервалы, контроль drift и стоимость решений. Облачный статус и проверенная ссылка будут добавлены после авторизации и фактического запуска.

Облачная версия загружает переносимую CatBoost-модель в формате CBM. Локальные pickle-файлы экспериментов не публикуются и не нужны для dashboard.

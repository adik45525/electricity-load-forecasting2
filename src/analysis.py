import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.data_loader import ROOT
from src.feature_engineering import SCENARIO_NAMES

plt.rcParams.update({"figure.facecolor": "#0b111b", "axes.facecolor": "#111b29",
                     "text.color": "#e8eff7", "axes.labelcolor": "#c0ccda",
                     "xtick.color": "#a3b4c8", "ytick.color": "#a3b4c8",
                     "axes.edgecolor": "#273548", "grid.color": "#273548",
                     "font.size": 11, "savefig.dpi": 170})


def save_figure(name, title, ylabel=None):
    plt.title(title, loc="left", pad=16, fontweight="bold")
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures" / f"{name}.png", bbox_inches="tight")
    plt.close()


def daily_bootstrap_difference(daily, column):
    a = daily.loc[daily[column] == 1, "load"].dropna().to_numpy()
    b = daily.loc[daily[column] == 0, "load"].dropna().to_numpy()
    rng = np.random.default_rng(42)
    differences = [rng.choice(a, len(a), replace=True).mean()
                   - rng.choice(b, len(b), replace=True).mean() for _ in range(2000)]
    return {"difference_mw": float(a.mean() - b.mean()),
            "lower_95": float(np.quantile(differences, .025)),
            "upper_95": float(np.quantile(differences, .975)),
            "group_days": [len(a), len(b)]}


def create_analysis(hourly, frame, performance, predictions, summary, audit):
    data = hourly.copy()
    local = data.index.tz_convert("Europe/Madrid")
    data["hour"] = local.hour
    data["weekday"] = local.dayofweek
    data["month"] = local.month
    data["date"] = local.date
    data["weekend"] = (local.dayofweek >= 5).astype(int)
    data["holiday"] = frame.holiday.reindex(data.index)
    profiles = data.groupby(["hour", "weekend"]).load.mean().unstack()
    profiles.to_csv(ROOT / "reports/hourly_profile.csv")
    seasonal = data.groupby("month").agg(load=("load", "mean"), temp=("temp", "mean"))
    seasonal.to_csv(ROOT / "reports/seasonality.csv")
    temperature = data.dropna(subset=["load", "temp"]).copy()
    temperature["temp_bin"] = (temperature.temp / 2).round() * 2
    temp_bins = temperature.groupby("temp_bin").agg(load=("load", "mean"), hours=("load", "size"))
    temp_bins.to_csv(ROOT / "reports/temperature_bins.csv")
    correlations = data[["load", "temp", "pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]].corr()
    correlations.to_csv(ROOT / "reports/correlations.csv")
    daily = data.groupby("date").agg(load=("load", "mean"), weekend=("weekend", "max"), holiday=("holiday", "max"))
    hypotheses = {"weekend": daily_bootstrap_difference(daily, "weekend"),
                  "holiday": daily_bootstrap_difference(daily, "holiday"),
                  "temp_load_pearson": float(data[["load", "temp"]].corr().iloc[0, 1])}
    # Residualize temperature relationship using a calendar-only model fitted on train.
    calendar = [c for c in frame if c not in ["target", "origin"] and not c.startswith(("load_", "past_", "oracle_"))]
    train = frame.origin < pd.Timestamp("2017-01-01", tz="UTC")
    regression = make_pipeline(StandardScaler(), Ridge(alpha=1000))
    regression.fit(frame.loc[train, calendar], frame.loc[train, "target"])
    residuals = frame.loc[train, "target"] - regression.predict(frame.loc[train, calendar])
    residual_temp = pd.DataFrame({"residual": residuals, "temp": hourly.temp.reindex(residuals.index)})
    residual_temp["temp_bin"] = (residual_temp.temp / 2).round() * 2
    controlled = residual_temp.groupby("temp_bin").agg(residual_mw=("residual", "mean"), hours=("residual", "size"))
    controlled.to_csv(ROOT / "reports/temperature_residuals.csv")
    hypotheses["temperature_note"] = "Calendar-adjusted residuals on train only; descriptive, not causal"
    (ROOT / "reports/hypotheses.json").write_text(json.dumps(hypotheses, indent=2), encoding="utf-8")
    plt.figure(figsize=(12, 4))
    data.load.resample("D").mean().plot(color="#36d6cb", lw=1)
    for date in ["2017-01-01", "2018-01-01"]:
        plt.axvline(pd.Timestamp(date, tz="UTC"), color="#6a9fff", ls="--", alpha=.8)
    save_figure("history", "Историческая нагрузка и временное разделение", "МВт, среднее за сутки")
    plt.figure(figsize=(10, 4))
    for col, label, color in [(0, "Будни", "#36d6cb"), (1, "Выходные", "#6a9fff")]:
        plt.plot(profiles.index, profiles[col], label=label, color=color, lw=2)
    plt.legend()
    plt.xlabel("Час, Europe/Madrid")
    save_figure("hourly_profile", "Суточный профиль: будни и выходные", "Нагрузка, МВт")
    plt.figure(figsize=(10, 4))
    plt.bar(seasonal.index, seasonal.load, color="#6a9fff")
    plt.xticks(range(1, 13))
    plt.xlabel("Месяц")
    save_figure("seasonality", "Сезонность нагрузки", "Средняя нагрузка, МВт")
    plt.figure(figsize=(10, 4))
    plt.plot(temp_bins.index, temp_bins.load, "o-", color="#36d6cb")
    plt.xlabel("Средняя температура пяти городов, °C")
    save_figure("temperature", "Температура и нагрузка: описательная связь", "Нагрузка, МВт")
    plt.figure(figsize=(10, 4))
    plt.bar(controlled.index, controlled.residual_mw, color="#36d6cb")
    plt.xlabel("Температура, °C")
    save_figure("temperature_residuals", "Остаток после календарной модели, train", "Средний остаток, МВт")
    plt.figure(figsize=(8, 6))
    plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(correlations)), correlations.columns, rotation=40)
    plt.yticks(range(len(correlations)), correlations.columns)
    plt.colorbar()
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            plt.text(j, i, f"{correlations.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
    save_figure("correlations", "Корреляции наблюдаемых показателей")
    best = predictions[(predictions.model == summary["best_model"]) & (predictions.scenario == summary["best_scenario"])]
    sample = best.iloc[:168]
    plt.figure(figsize=(12, 4))
    plt.plot(pd.to_datetime(sample.time), sample.actual, color="#6a9fff", label="Факт", lw=1.5)
    plt.plot(pd.to_datetime(sample.time), sample.predicted, color="#36d6cb", label="Прогноз", lw=1.5)
    plt.legend()
    save_figure("forecast_week", "Первая неделя теста: независимые блоки по 24 часа", "Нагрузка, МВт")
    test = performance[performance.split == "test"].copy()
    plt.figure(figsize=(12, 6))
    labels = test.model + " / " + test.scenario
    plt.barh(labels.iloc[::-1], test.MAE.iloc[::-1], color=["#6a9fff" if x == "weather_oracle" else "#36d6cb" for x in test.scenario.iloc[::-1]])
    save_figure("model_comparison", "Сравнение всех экспериментов на тесте", "")
    plt.figure(figsize=(10, 4))
    plt.hist(best.actual - best.predicted, bins=60, color="#36d6cb", alpha=.8)
    plt.xlabel("Ошибка: факт − прогноз, МВт")
    save_figure("errors", "Распределение ошибок выбранной модели", "Число часов")
    importance = pd.read_csv(ROOT / "reports/feature_importance.csv").head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(importance.feature.iloc[::-1], importance.importance_mw.iloc[::-1], color="#36d6cb")
    save_figure("feature_importance", "Permutation importance выбранной модели", "")
    shap = pd.read_csv(ROOT / "reports/shap_importance.csv").head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(shap.feature.iloc[::-1], shap.mean_abs_shap_mw.iloc[::-1], color="#6a9fff")
    save_figure("shap", f"SHAP: CatBoost / {summary['shap_scenario']}", "")

    effects = []
    for model in ["Ridge", "Random Forest", "CatBoost"]:
        subset = predictions[predictions.model == model].copy()
        subset["absolute_error"] = (subset.actual - subset.predicted).abs()
        daily_errors = subset.groupby(["origin", "scenario"]).absolute_error.mean().unstack()
        for scenario in ["weather_past", "weather_oracle"]:
            differences = (daily_errors[scenario] - daily_errors["history"]).dropna().to_numpy()
            rng = np.random.default_rng(42)
            draws = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
            effects.append({"model": model, "scenario": scenario,
                            "mae_difference_mw": float(differences.mean()),
                            "lower_95": float(np.quantile(draws, .025)),
                            "upper_95": float(np.quantile(draws, .975)),
                            "days": len(differences)})
    pd.DataFrame(effects).to_csv(ROOT / "reports/weather_effect.csv", index=False)

    write_reports(summary, audit, hypotheses, performance, best)


def markdown_table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_reports(summary, audit, hypotheses, performance, predictions):
    test = performance[performance.split == "test"]
    baseline = test[test.model == "Seasonal Naive"].iloc[0]
    gain = 100 * (1 - summary["test"]["MAE"] / baseline.MAE)
    weekend = hypotheses["weekend"]
    holiday = hypotheses["holiday"]
    interval = summary["bootstrap"]
    cat = test[test.model == "CatBoost"].set_index("scenario")
    cat_val = performance[(performance.model == "CatBoost") & (performance.split == "validation")].set_index("scenario")
    val_weather_gain = cat_val.loc["history", "MAE"] - cat_val.loc["weather_past", "MAE"]
    test_weather_delta = cat.loc["weather_past", "MAE"] - cat.loc["history", "MAE"]
    weather_effect = pd.read_csv(ROOT / "reports/weather_effect.csv")
    cat_effect = weather_effect[(weather_effect.model == "CatBoost") & (weather_effect.scenario == "weather_past")].iloc[0]
    weather_conclusion = ("Интервал включает ноль: устойчивого выигрыша прошлой погоды не подтверждено."
                          if cat_effect.lower_95 <= 0 <= cat_effect.upper_95 else
                          "Интервал не включает ноль; направление разницы следует оценивать по его знаку.")
    weather_comparisons = []
    for model in ["Ridge", "Random Forest", "CatBoost"]:
        records = test[test.model == model].set_index("scenario")
        weather_comparisons.append(f"- {model}: история MAE {records.loc['history','MAE']:.1f} МВт; прошлая погода {records.loc['weather_past','MAE']:.1f} МВт; oracle {records.loc['weather_oracle','MAE']:.1f} МВт.")
    report = f"""# Прогнозирование нагрузки электросистемы Испании

## Бизнес-проблема и критерий успеха
Оператору сети нужен почасовой прогноз на следующие 24 часа для планирования генерации, резервов и закупок. Недооценка пика осложняет обеспечение баланса, переоценка ведёт к избыточному резервированию. Прототип помогает видеть суточный профиль и ожидаемый пиковый час. Это исследовательская система, не инструмент диспетчерского управления.

До эксперимента определены критерии: снизить MAE относительно недельного Seasonal Naive минимум на 10%, оценить ошибку мощности и времени пика, обеспечить воспроизводимость без временных утечек. Получено снижение MAE на {gain:.2f}%; критерий {'выполнен' if gain >= 10 else 'не выполнен'}. Денежная экономия не рассчитывается: нет стоимости резервов, штрафов и решений оператора. Нагрузка измеряется в МВт; сумма почасовых значений при интервале 1 час даёт энергию в МВт·ч.

## Источники и существующие решения
Датасет: [Nicholas Jhana, Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). Использованы предоставленные пользователем CSV, контрольные суммы сохранены в `data_quality.json`. [Red Eléctrica](https://www.ree.es/en/operation/electricity-system/demand-and-production-curves-in-real-time) публикует фактическую, прогнозную и запланированную нагрузку; это реальный контекст задачи. Обзор [Hong & Fan (2016)](https://www.sciencedirect.com/science/article/pii/S0169207015001508) рассматривает вероятностное прогнозирование нагрузки и его применение в энергетическом планировании. Наш проект ограничивается точечным прогнозом; вероятностные интервалы нагрузки остаются развитием, bootstrap ниже оценивает разницу ошибок, а не неопределённость будущей нагрузки.

## Данные и качество
Период UTC: {audit['start_utc']} — {audit['end_utc']}. {audit['energy_rows']:,} строк энергии; {audit['weather_rows']:,} строк погоды. Города: {', '.join(audit['cities'])}. Энергетический источник содержит {audit['missing_load']} пропущенных значений нагрузки и {audit['energy_duplicate_timestamps']} повторных временных меток. Погодных повторов city/time: {audit['weather_duplicate_keys']}. Аномалии погоды по фиксированным физическим границам: {audit['weather_invalid']}.

Ключи времени разбираются с UTC, затем календарь строится в Europe/Madrid. Так сохраняются различия смещений и переходы DST. Повторные погодные ключи усредняются, затем города объединяются равными весами. Температура переведена из Кельвинов в °C. Недопустимые давление/влажность/ветер и другие значения становятся пропусками; заполнение только вперёд, максимум 24 часа. Для истории нагрузки применяется аналогичное заполнение, но целевая переменная НЕ заполняется. Физически допустимые пики не обрезаются по квантилям. Никакие цены, генерация или внешний total load forecast не используются как признаки.

Для сравнения берутся одинаковые полные UTC-блоки по 24 наблюдаемых цели с доступными признаками; первые 336 часов и неполные блоки исключаются. Всего исключено {summary['excluded_hours']} часов. Это ограничивает выводы: качество на неполных/аварийных периодах не оценено. После forward fill агрегат погоды может использовать менее пяти городов; нет весов населения и метеостанций за пределами городов.

## EDA и гипотезы
Графики: `reports/figures`; численные сводки: `hourly_profile.csv`, `seasonality.csv`, `temperature_bins.csv`, `correlations.csv`, `hypotheses.json`.

1. **В выходные средняя нагрузка ниже.** Разница выходные − будни: {weekend['difference_mw']:.1f} МВт, 95% bootstrap по суткам [{weekend['lower_95']:.1f}; {weekend['upper_95']:.1f}]. Это описательная ассоциация; bootstrap не полностью учитывает межсуточную автокорреляцию.
2. **Праздничные дни отличаются от остальных.** Разница: {holiday['difference_mw']:.1f} МВт, интервал [{holiday['lower_95']:.1f}; {holiday['upper_95']:.1f}]. Использованы общенациональные праздники Spain, без региональных календарей. Группы не сбалансированы по сезонам и дням недели, поэтому это не причинная оценка.
3. **Связь температуры с нагрузкой нелинейна.** Pearson r = {hypotheses['temp_load_pearson']:.3f} сам по себе недостаточен. График 2°C-бинов дополняется температурным профилем остатков Ridge после учёта календаря, построенным только на train. Признаки heating/cooling используют заранее заданные пороги 18°C/22°C, не подобранные на test.
4. **Погода добавляет информацию сверх истории.** Проверяется одинаковой ablation study, результаты ниже. Возможное отсутствие улучшения — полноценный результат, а не повод выбирать метрики по тесту.

## Протокол прогноза и защита от leakage
Origin — 00:00 UTC. Прогноз выпускается один раз на 24 следующих последовательных часа (horizon 1–24); это не прогноз каждого часа с обновлением фактической нагрузки внутри дня. В Испании начало блока приходится на 01:00/02:00 местного времени. Блок из 24 UTC-часов не всегда совпадает с местными календарными сутками из-за DST. Это осознанное упрощение demo.

Лаги 24/48/168/336 часов для любого горизонта строго раньше origin. Rolling mean/std/min/max за 24/168 часов вычисляются после shift(1) и фиксируются на origin для всего блока. Прошлая погода берётся t−24, тоже строго раньше origin. Только `weather_oracle` использует фактическую погоду t, поэтому это исследовательский upper-information scenario, а не прогноз погоды. Oracle не допускается к выбору основной модели.

Train: UTC 2015–2016 ({summary['split_hours']['train']} часов), validation: 2017 ({summary['split_hours']['validation']} часов), test: 2018 ({summary['split_hours']['test']} часов, {summary['split_days']['test']} полных блоков). Это один хронологический holdout; устойчивость по нескольким rolling folds не проверялась. Все scaler обучены внутри train. Для каждого алгоритма/сценария два набора гиперпараметров оцениваются по validation MAE. После выбора параметры фиксируются и модель переобучается на train+validation. Test не используется для выбора модели/параметров. Все test-прогнозы — дообучения внутри 2018 нет, фактическая прошлая история 2018 доступна при каждом origin.

## Модели и эксперименты
Seasonal Naive повторяет нагрузку t−168. Ridge со StandardScaler проверяет линейную регуляризованную зависимость. Random Forest моделирует нелинейные взаимодействия. CatBoost — градиентный бустинг по деревьям. Календарь включает локальные час, день недели, месяц, годовой цикл, выходные, праздники, UTC offset, рабочие часы и one-hot категории. Полный перебор параметров сохранён в `tuning.json`.

MAE и RMSE — МВт; MAPE и sMAPE — %. MAE основной критерий, RMSE выделяет большие ошибки, относительные метрики облегчают сопоставление часов. MAPE приемлема здесь из-за строго положительной национальной нагрузки, но не подходит для рядов около нуля.

{markdown_table(test[['model', 'scenario', 'MAE', 'RMSE', 'MAPE', 'sMAPE']])}

Выбор по validation: **{summary['best_model']} / {SCENARIO_NAMES[summary['best_scenario']]}**, validation MAE {summary['validation_mae']:.2f} МВт. После refit test MAE {summary['test']['MAE']:.2f}, RMSE {summary['test']['RMSE']:.2f} МВт; MAPE {summary['test']['MAPE']:.2f}%, sMAPE {summary['test']['sMAPE']:.2f}%.

### Что даёт погода
{chr(10).join(weather_comparisons)}

Для CatBoost разница validation MAE (история − прошлая погода): {val_weather_gain:.2f} МВт, {100 * val_weather_gain / cat_val.loc['history', 'MAE']:.2f}%. Разница test MAE (погода − история): {test_weather_delta:.2f} МВт; отрицательное значение означает улучшение от погоды. Парный bootstrap по test-дням: [{cat_effect.lower_95:.2f}; {cat_effect.upper_95:.2f}] МВт. {weather_conclusion} Аналогичные интервалы всех алгоритмов и oracle сохранены в `weather_effect.csv`. Выбранная по validation модель сохранена; менять её на модель с лучшим test было бы использованием теста для выбора.

Это сравнение моделей с отдельным подбором параметров в каждом сценарии, не изолированная причинная оценка признака. Будущая наблюдаемая погода показывает ценность дополнительной информации; в production нужен архив прогнозов погоды с временем выпуска, чтобы повторить доступность признаков.

## Ошибки и практический смысл
Средняя ошибка мощности дневного пика: {summary['peak_mae_mw']:.1f} МВт. Средняя абсолютная разница часов пика в UTC-блоке: {summary['peak_hour_mae']:.2f} часа. Для плато нагрузки сравнение одного argmax может быть нестабильно. Разбивка MAE по местному часу, месяцу, выходным и горизонту находится в `error_breakdown.csv`, все пики — `daily_peaks.csv`.

Парный bootstrap разницы суточных MAE выбранная модель − Seasonal Naive: {interval['mae_difference_mw']:.1f} МВт, 95% интервал [{interval['lower_95']:.1f}; {interval['upper_95']:.1f}], {interval['days']} суток, 2000 выборок seed=42. Отрицательная разница означает улучшение. Дни переотбираются независимо, так что интервал не полностью учитывает зависимость соседних дней.

## Интерпретация
Permutation importance выбранной модели: прирост MAE при перестановке признака, 3 повтора на систематической выборке 1 часа каждого UTC-дня test. Это ограниченная диагностическая выборка, не полная оценка всех горизонтов; коррелированные лаги делят важность. SHAP рассчитан встроенным TreeSHAP CatBoost для {SCENARIO_NAMES[summary['shap_scenario']]}, средние абсолютные вклады в МВт и значения отдельных наблюдений сохранены. Если основной алгоритм иной, SHAP относится именно к указанной CatBoost-модели. Ни importance, ни SHAP не доказывают причинность.

## Demo и воспроизводимость
Шесть страниц Streamlit: обзор, прогноз, аналитика данных, качество моделей, интерпретация и о проекте. Все показатели читаются из сохранённых CSV/JSON. Forecast выполняет inference загруженной лучшей моделью для выбранного блока test и сравнивает его с реальными целями. Свежих данных после 2018 нет; прогноз на текущую дату не предлагается.

Команды установки, обучения, тестов и запуска приведены в README. Все модели и точные версии библиотек сохранены локально. Публикация в GitHub/облаке не выполнена; Dockerfile подготовлен, контейнерный запуск отдельно не проверен.

## Ограничения и дальнейшая работа
Данные 2015–2018 не описывают текущую систему. Требуются актуальная история, контроль времени публикации нагрузки и прогнозов погоды, региональные праздники, региональные веса погодных станций, backtesting по нескольким годам, мониторинг drift и вероятностный прогноз. Предполагается мгновенная доступность нагрузки до origin; задержки телеметрии не моделируются. Нет причинного вывода или рассчитанной финансовой экономии. Промежуточный отчёт подготовлен как материал, но сроки сдачи из задания не могут быть исполнены задним числом.
"""
    (ROOT / "reports/final_report.md").write_text(report, encoding="utf-8")
    notes = f"""# Выступление на 10 минут

0:00–1:00 — Проблема оператора: заранее знать суточный профиль и пик, чтобы планировать резерв. Не обещать денежную экономию без модели затрат.

1:00–2:00 — Источник, пять городов Испании, четыре года, {audit['hours']} часов. Показать страницу «О проекте» и качество данных: пропуски, дубликаты, физические аномалии, UTC и DST.

2:00–3:30 — На «Аналитике данных» показать профиль выходных/будней, сезонность и температурные остатки. Выходные отличаются на {weekend['difference_mw']:.0f} МВт. Объяснить, почему линейная корреляция не описывает нелинейную связь и почему это не причинный эффект.

3:30–5:00 — Протокол: train 2015–2016, validation 2017, test 2018. В origin 00:00 UTC выпускаются все 24 часа. Объяснить t−24 и замороженные rolling-признаки на примере последнего горизонта. Нельзя использовать load.shift(1) для всех будущих часов.

5:00–6:30 — Четыре алгоритма и ablation. Открыть «Качество моделей». Основная модель {summary['best_model']}, выбрана до test. MAE {summary['test']['MAE']:.0f} МВт, MAPE {summary['test']['MAPE']:.2f}%, улучшение baseline {gain:.1f}%. Oracle явно отделён от доступной прошлой погоды.

6:30–8:00 — Demo страницы «Прогноз»: выбрать исторический день из 2018, показать фактическую кривую, прогноз, пик, экспорт CSV. Сказать, что это inference модели, а не заранее придуманные числа. Даты 2026 не прогнозируются без свежей истории.

8:00–9:00 — «Интерпретация»: permutation importance и CatBoost SHAP. Назвать основные признаки по реальному графику; подчеркнуть зависимость лагов. Показать ошибки по часу/месяцу и ограничения пикового часа.

9:00–10:00 — Выводы и развитие: реальные прогнозы погоды, новые данные, rolling backtests и прогнозные интервалы. Проект легко запускается локально; публикация ещё не выполнена.

## Ответы на вопросы
- Почему не random split? Он перемешивает прошлое и будущее и переоценивает качество.
- Почему не lag1 для каждого часа? После первого горизонта lag1 внутри блока ещё неизвестен.
- Почему не текущий прогноз? Нет свежей истории после 2018 года.
- Почему oracle? Проверяем ценность погоды, сохраняя честное обозначение доступности.
- Почему MAE? Понятная средняя ошибка в МВт; RMSE дополнительно штрафует большие промахи.
- Что оригинального? Сопоставление доступной прошлой и будущей oracle-погоды, календарно скорректированный температурный профиль и анализ пиков в едином честном 24-часовом протоколе.
- Можно ли приписать экономию? Нет, отсутствуют данные о резервировании и его стоимости.
- Есть ли утечки? Проверки меняют будущие цели/погоду и подтверждают неизменность operational-признаков; transformer обучен только на train.
"""
    (ROOT / "reports/presentation_notes.md").write_text(notes, encoding="utf-8")
    (ROOT / "reports/interim_report.md").write_text("# Материал промежуточного отчёта\n\nТема: прогноз почасовой нагрузки Испании на 24 часа. Датасет Nicholas Jhana (Kaggle), энергия и погода пяти городов, 2015–2018. Бизнес-задача: поддержка планирования генерации и резервов. План: аудит качества → EDA → временные признаки → Seasonal Naive, Ridge, Random Forest, CatBoost → временная валидация и ablation → анализ ошибок и SHAP → Streamlit demo. Реализация и результаты отражены в final_report.md. Это подготовленный документ, а не подтверждение сдачи в установленный срок.\n", encoding="utf-8")
    (ROOT / "reports/criteria_checklist.md").write_text("""# Соответствие критериям Capstone

| Критерий | Вес | Материал |
| --- | --- | --- |
| Business understanding / EDA | 25% | final_report.md, hypotheses.json, figures, Data Insights |
| Deployment | 10% | README, requirements.lock.txt, Dockerfile, локальный запуск Streamlit; облачная публикация не выполнена |
| ML modeling | 25% | src/train.py, tuning.json, metrics.csv, тесты leakage |
| Оригинальность / сравнение | 20% | ablation: календарь, история, past weather, oracle; пики и температурные остатки |
| Промежуточный отчёт | 5% | interim_report.md; сдача в срок пользователем не подтверждена |
| Demo / выступление | 15% | шесть страниц приложения, presentation_notes.md на 10 минут |

Слайды не создавались согласно запросу. Материал готовится для индивидуальной защиты; данные о составе команды не предоставлены.\n""", encoding="utf-8")

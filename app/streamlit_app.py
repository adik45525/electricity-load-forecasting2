import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import forecast_historical, load_model
from src.evaluate import metrics
from src.feature_engineering import SCENARIO_NAMES

st.set_page_config(page_title="GridScope | Аналитика энергосистемы", page_icon="⚡", layout="wide")
st.markdown((ROOT / "app/components/style.html").read_text(), unsafe_allow_html=True)
TEAL, BLUE, AMBER = "#36d6cb", "#78a5ff", "#e8b96e"
PAGES = ["Обзор", "Прогноз", "Аналитика данных", "Качество моделей", "Интерпретация", "О проекте"]


@st.cache_data(max_entries=1, show_spinner="Загрузка сохранённых результатов…")
def read_results():
    hourly = pd.read_csv(ROOT / "data/processed/hourly.csv", index_col="time", parse_dates=["time"])
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    summary = json.loads((ROOT / "reports/summary.json").read_text())
    selected = []
    for chunk in pd.read_csv(ROOT / "reports/predictions.csv", chunksize=12000):
        selected.append(chunk[(chunk.model == summary["best_model"])
                              & (chunk.scenario == summary["best_scenario"])])
    prediction = pd.concat(selected, ignore_index=True)
    prediction["time"] = pd.to_datetime(prediction.time, utc=True)
    prediction["origin"] = pd.to_datetime(prediction.origin, utc=True)
    quality = json.loads((ROOT / "reports/data_quality.json").read_text())
    performance = pd.read_csv(ROOT / "reports/metrics.csv")
    local = hourly.index.tz_convert("Europe/Madrid")
    hourly["hour"] = local.hour
    hourly["weekday"] = local.dayofweek
    hourly["month"] = local.month
    hourly["day_type"] = np.where(local.dayofweek >= 5, "Выходные", "Будни")
    return hourly, prediction, summary, quality, performance


@st.cache_resource(max_entries=1)
def model_bundle():
    return load_model()


@st.cache_data(max_entries=64, show_spinner="Расчёт исторического прогноза…")
def cached_forecast(origin):
    hourly, *_ = read_results()
    return forecast_historical(hourly, origin, model_bundle())


def title(name, description, eyebrow="ИСПАНИЯ / ENERGY INTELLIGENCE"):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(name)
    st.markdown(f'<div class="subhead">{description}</div>', unsafe_allow_html=True)


def kpis(items):
    columns = st.columns(len(items))
    for column, (label, value, unit, detail) in zip(columns, items):
        with column:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
                        f'<div class="kpi-value">{html.escape(str(value))}<small>{html.escape(unit)}</small></div>'
                        f'<div class="kpi-detail">{html.escape(detail)}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{html.escape(text)}</div>', unsafe_allow_html=True)


def chart(fig, height=350, legend_below=False):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="#111c2b", plot_bgcolor="#111c2b",
                      margin=dict(l=20, r=20, t=24, b=28),
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#b4c4d7", size=12),
                      hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                      colorway=[TEAL, BLUE, AMBER],
                      xaxis=dict(gridcolor="#203044", zeroline=False),
                      yaxis=dict(gridcolor="#203044", zeroline=False))
    if legend_below:
        fig.update_layout(margin=dict(l=20, r=20, t=24, b=115),
                          legend=dict(orientation="h", y=-.22, x=0, yanchor="top", font=dict(size=10)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def panel_title(label, caption=None):
    st.subheader(label)
    if caption:
        st.caption(caption)


def number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", " ")


if not (ROOT / "reports/summary.json").exists():
    st.error("Результаты экспериментов отсутствуют. Выполните: python -m src.train")
    st.stop()

hourly, predictions, summary, quality, performance = read_results()
best_predictions = predictions[(predictions.model == summary["best_model"])
                               & (predictions.scenario == summary["best_scenario"])]
baseline_mae = performance[(performance.split == "test") & (performance.model == "Seasonal Naive")].MAE.iloc[0]
gain = 100 * (1 - summary["test"]["MAE"] / baseline_mae)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brandmark">ϟ</div><div class="brandname">GridScope'
                '<span>ENERGY INTELLIGENCE</span></div></div>', unsafe_allow_html=True)
    st.caption("РАБОЧЕЕ ПРОСТРАНСТВО")
    page = st.radio("Навигация", PAGES, label_visibility="collapsed", key="page")
    st.markdown('<div class="side-info"><strong>Историческая демонстрация</strong><br>'
                'Электросистема Испании · 2015–2018<br>Данные заканчиваются в 2018 году<br>Горизонт прогноза · 24 часа<br>'
                'Календарь · Europe/Madrid<br><br>CAPSTONE / DATA SCIENCE</div>', unsafe_allow_html=True)
    st.markdown('<span class="status">РЕАЛЬНЫЕ ДАННЫЕ · ИСТОРИЧЕСКОЕ DEMO</span>', unsafe_allow_html=True)

if page == "Обзор":
    title("Нагрузка под контролем", "История энергосистемы, суточные закономерности и качество прогноза на одном экране.")
    kpis([("Средняя нагрузка", number(hourly.load.mean()), "МВт", "За весь наблюдаемый период"),
          ("Исторический максимум", number(hourly.load.max()), "МВт", "Пиковая почасовая мощность"),
          ("Ошибка прогноза", number(summary["test"]["MAPE"], 2), "% MAPE", "Независимый тест · 2018"),
          ("Улучшение к baseline", number(gain, 1), "% MAE", "Недельный Seasonal Naive")])
    with st.container(border=True):
        a, b = st.columns([3, 1])
        with a:
            panel_title("Динамика энергопотребления", "Средняя суточная мощность · UTC. Наведите курсор для точного значения.")
        with b:
            year = st.selectbox("Период", ["Все годы", "2015", "2016", "2017", "2018"], key="overview_year")
        visible = hourly if year == "Все годы" else hourly[hourly.index.tz_convert("Europe/Madrid").year == int(year)]
        daily = visible.load.resample("D").mean()
        fig = go.Figure(go.Scatter(x=daily.index, y=daily, name="Нагрузка", mode="lines",
                                  line=dict(color=TEAL, width=1.8), fill="tozeroy", fillcolor="rgba(54,214,203,.07)"))
        fig.update_yaxes(title="МВт", range=[daily.min() * .95, daily.max() * 1.04])
        chart(fig, 340)
    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        panel_title("Ритм недели", "Средний часовой профиль · местное время Испании")
        profile = hourly.groupby(["hour", "day_type"]).load.mean().reset_index()
        fig = px.line(profile, x="hour", y="load", color="day_type",
                      color_discrete_map={"Будни": TEAL, "Выходные": BLUE}, labels={"hour": "Час", "load": "МВт", "day_type": ""})
        chart(fig, 290)
    with right, st.container(border=True):
        panel_title("Проверка прогноза", f"{summary['best_model']} · первые 72 часа теста")
        sample = best_predictions.iloc[:72]
        fig = go.Figure()
        fig.add_scatter(x=sample.time, y=sample.actual, name="Факт", line=dict(color=BLUE, width=1.7))
        fig.add_scatter(x=sample.time, y=sample.predicted, name="Прогноз", line=dict(color=TEAL, width=1.7))
        fig.update_yaxes(title="МВт")
        chart(fig, 290)
    st.caption(f"Источник: Kaggle / Nicholas Jhana. {quality['hours']:,} часов. Значения нагрузки — мощность в МВт; это не энергия в МВт·ч.")

elif page == "Прогноз":
    title("Следующие 24 часа", "Выберите начало исторического блока. Прогноз рассчитывается сохранённой моделью и сопоставляется с фактом.")
    note("Историческая демонстрация на test 2018. Начало блока — 00:00 UTC (01:00/02:00 в Испании). Свежей истории для текущей даты нет.")
    origins = sorted(best_predictions.origin.dt.date.unique())
    controls, details = st.columns([1, 2])
    with controls:
        date = st.select_slider("Начало прогноза · UTC", options=origins, value=origins[len(origins) // 2], key="forecast_date")
    with details:
        st.caption(f"Модель: {summary['best_model']} · {SCENARIO_NAMES[summary['best_scenario']]}\n\nВыбрана по validation MAE, до оценки на тесте.")
    origin = pd.Timestamp(date, tz="UTC")
    forecast = cached_forecast(origin.isoformat())
    observed = forecast.actual.notna()
    scores = metrics(forecast.loc[observed, "actual"], forecast.loc[observed, "predicted"])
    peak = forecast.predicted.idxmax()
    local_peak = peak.tz_convert("Europe/Madrid")
    kpis([("Ожидаемый пик", number(forecast.predicted.max()), "МВт", f"{local_peak:%d.%m · %H:%M} Europe/Madrid"),
          ("Средняя нагрузка", number(forecast.predicted.mean()), "МВт", "Прогноз выбранного блока"),
          ("Ошибка блока", number(scores["MAE"]), "МВт MAE", f"MAPE {scores['MAPE']:.2f}%"),
          ("Горизонт", "24", "часа", "Один origin · без обновления внутри блока")])
    with st.container(border=True):
        panel_title("Прогноз и фактическая нагрузка", "Часы на графике указаны в Europe/Madrid с UTC-смещением.")
        labels = [t.strftime("%d.%m %H:%M %z") for t in forecast.index.tz_convert("Europe/Madrid")]
        fig = go.Figure()
        fig.add_scatter(x=labels, y=forecast.actual, name="Факт", line=dict(color=BLUE, width=2.3))
        fig.add_scatter(x=labels, y=forecast.predicted, name="Прогноз", line=dict(color=TEAL, width=2.3),
                        fill=None)
        peak_label = peak.tz_convert("Europe/Madrid").strftime("%d.%m %H:%M %z")
        fig.add_scatter(x=[peak_label], y=[forecast.predicted.max()], name="Пик прогноза", mode="markers",
                       marker=dict(color=AMBER, size=11, line=dict(color="#111c2b", width=2)))
        fig.update_yaxes(title="МВт")
        fig.update_xaxes(nticks=8)
        chart(fig, 410)
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        panel_title("Почасовая ошибка", "Положительное значение: модель недооценила нагрузку.")
        fig = go.Figure(go.Bar(x=labels, y=forecast.actual - forecast.predicted, marker_color=TEAL, name="Факт − прогноз"))
        fig.update_yaxes(title="МВт")
        fig.update_xaxes(nticks=6)
        chart(fig, 220)
    with right, st.container(border=True):
        panel_title("Сохранить прогноз")
        st.write("Полный блок: временные метки UTC, фактическая и прогнозируемая нагрузка.")
        export = forecast.copy()
        export.index.name = "time_utc"
        st.download_button("Скачать CSV", export.to_csv().encode("utf-8"), f"forecast_{date}.csv", "text/csv", width="stretch")
        st.caption("Информационный прогноз. Погрешность пика и почасовые ошибки доступны на странице качества моделей.")
    with st.expander("Почасовые значения"):
        table = forecast.copy()
        table.index = labels
        st.dataframe(table.rename(columns={"actual": "Факт, МВт", "predicted": "Прогноз, МВт"}).round(1), width="stretch")

elif page == "Аналитика данных":
    title("Что движет нагрузкой", "Температура, рабочий календарь и сезонность. Описательные связи дополняются экспериментами с признаками.")
    hypotheses = json.loads((ROOT / "reports/hypotheses.json").read_text())
    kpis([("Температура и нагрузка", number(hypotheses["temp_load_pearson"], 3), "r", "Pearson · линейная связь"),
          ("Выходные − будни", number(hypotheses["weekend"]["difference_mw"]), "МВт", "Разница средних суточных значений"),
          ("Праздники − остальные", number(hypotheses["holiday"]["difference_mw"]), "МВт", "Общенациональные праздники"),
          ("Погодные города", str(len(quality["cities"])), "городов", "Равные веса при агрегации")])
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Температура и потребление", "Средние по температурным бинам 2°C · все наблюдаемые часы")
        bins = pd.read_csv(ROOT / "reports/temperature_bins.csv")
        fig = px.line(bins, x="temp_bin", y="load", markers=True, labels={"temp_bin": "Средняя температура, °C", "load": "МВт"}, hover_data=["hours"])
        chart(fig, 310)
    with b, st.container(border=True):
        panel_title("После учёта календаря", "Остатки календарной Ridge · только train 2015–2016")
        controlled = pd.read_csv(ROOT / "reports/temperature_residuals.csv")
        fig = px.bar(controlled, x="temp_bin", y="residual_mw", labels={"temp_bin": "Температура, °C", "residual_mw": "Средний остаток, МВт"}, hover_data=["hours"])
        chart(fig, 310)
    note("Температурные профили показывают ассоциацию, а не причинный эффект. Для оценки добавочной прогнозной ценности погоды смотрите ablation study на странице качества моделей.")
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Сезонная нагрузка", "Средняя нагрузка по месяцам за четыре года")
        season = pd.read_csv(ROOT / "reports/seasonality.csv")
        fig = px.bar(season, x="month", y="load", labels={"month": "Месяц", "load": "МВт"}, hover_data=["temp"])
        fig.update_xaxes(dtick=1)
        chart(fig, 280)
    with b, st.container(border=True):
        panel_title("Неделя × час", "Средняя мощность, МВт · местное время")
        matrix = hourly.pivot_table(index="weekday", columns="hour", values="load", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns,
                                  y=["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"],
                                  colorscale=[[0, "#183049"], [.5, "#246d83"], [1, "#4be0cc"]],
                                  colorbar=dict(title="МВт", thickness=10)))
        chart(fig, 280)
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Распределение нагрузки")
        fig = px.histogram(hourly, x="load", nbins=60, labels={"load": "МВт"})
        fig.update_yaxes(title="Число часов")
        chart(fig, 280)
    with b, st.container(border=True):
        panel_title("Корреляции", "Наблюдаемая нагрузка и агрегированная погода")
        corr = pd.read_csv(ROOT / "reports/correlations.csv", index_col=0)
        labels = ["Нагрузка", "Темп.", "Давление", "Влажность", "Ветер", "Дождь", "Облака"]
        fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
                                  colorscale="RdBu", text=np.round(corr.to_numpy(), 2), texttemplate="%{text}",
                                  colorbar=dict(thickness=10)))
        chart(fig, 280)
    with st.expander("Проверка гипотез и качество данных"):
        for key, label in [("weekend", "Выходные"), ("holiday", "Праздники")]:
            values = hypotheses[key]
            st.write(f"{label}: разница {values['difference_mw']:.1f} МВт; 95% bootstrap [{values['lower_95']:.1f}; {values['upper_95']:.1f}] МВт.")
        st.caption("Bootstrap по отдельным суткам, 2000 повторов. Соседние дни могут зависеть друг от друга; интервалы приблизительны. Группы не сбалансированы по сезону.")
        st.json(quality)

elif page == "Качество моделей":
    title("Результаты, которые можно проверить", "Хронологический holdout, общий набор часов и четыре сценария признаков. Модель выбрана по validation, не по test.")
    kpis([("Выбранная модель", summary["best_model"], "", "Минимум validation MAE · без oracle"),
          ("Test MAE", number(summary["test"]["MAE"]), "МВт", f"RMSE {summary['test']['RMSE']:.0f} МВт"),
          ("Ошибка мощности пика", number(summary["peak_mae_mw"]), "МВт", "Среднее по полным test-блокам"),
          ("Ошибка часа пика", number(summary["peak_hour_mae"], 2), "часа", "Разница argmax в UTC-блоке")])
    note("Train: 2015–2016 → validation: 2017 → test: 2018. После подбора параметров модели переобучены на train + validation. Oracle использует будущую фактическую погоду и исключён из выбора основной модели.")
    split = st.radio("Период оценки", ["test", "validation"], horizontal=True,
                     format_func=lambda s: "Тест · 2018" if s == "test" else "Валидация · 2017", key="performance_split")
    table = performance[performance.split == split].copy()
    with st.container(border=True):
        panel_title("Ablation study", "Одинаковые блоки для всех алгоритмов. Меньшее MAE лучше.")
        fig = px.bar(table[table.model != "Seasonal Naive"], x="model", y="MAE", color="scenario", barmode="group",
                     color_discrete_map={"calendar": "#516681", "history": BLUE, "weather_past": TEAL, "weather_oracle": AMBER},
                     labels={"model": "", "MAE": "MAE, МВт", "scenario": ""})
        for trace in fig.data:
            trace.name = {"calendar": "Календарь", "history": "+ история", "weather_past": "+ прошлая погода", "weather_oracle": "+ будущая погода (oracle)"}[trace.name]
        baseline = table[table.model == "Seasonal Naive"].MAE.iloc[0]
        fig.add_hline(y=baseline, line_dash="dot", line_color="#acb8c7",
                      annotation_text="Seasonal Naive", annotation_position="top left")
        chart(fig, 430, legend_below=True)
    if (ROOT / "reports/weather_effect.csv").exists():
        effects = pd.read_csv(ROOT / "reports/weather_effect.csv")
        effect = effects[(effects.model == "CatBoost") & (effects.scenario == "weather_past")].iloc[0]
        note(f"Добавление прошлой погоды к CatBoost на test: MAE меняется на {effect.mae_difference_mw:+.2f} МВт; 95% bootstrap [{effect.lower_95:.2f}; {effect.upper_95:.2f}]. Отрицательная разница лучше. Это сравнение на тесте не меняет выбор по валидации.")
    display = table[["model", "scenario", "MAE", "RMSE", "MAPE", "sMAPE", "hours"]].copy()
    display.scenario = display.scenario.map(SCENARIO_NAMES)
    st.dataframe(display.rename(columns={"model": "Алгоритм", "scenario": "Признаки", "MAE": "MAE, МВт", "RMSE": "RMSE, МВт", "MAPE": "MAPE, %", "sMAPE": "sMAPE, %", "hours": "Часов"}).round(2), hide_index=True, width="stretch")
    breakdown = pd.read_csv(ROOT / "reports/error_breakdown.csv")
    breakdown = breakdown[(breakdown.model == summary["best_model"]) & (breakdown.scenario == summary["best_scenario"])]
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Где модель ошибается", "Выбранная модель · test MAE по местному часу")
        values = breakdown[breakdown.group == "hour"].copy()
        values["value"] = pd.to_numeric(values.value)
        fig = px.bar(values.sort_values("value"), x="value", y="MAE", labels={"value": "Час Europe/Madrid", "MAE": "МВт"})
        chart(fig, 270)
    with b, st.container(border=True):
        panel_title("Ошибка по горизонту", "1–24 часа от origin · test")
        values = breakdown[breakdown.group == "horizon"].copy()
        values["value"] = pd.to_numeric(values.value)
        fig = px.line(values.sort_values("value"), x="value", y="MAE", markers=True, labels={"value": "Горизонт, час", "MAE": "МВт"})
        chart(fig, 270)
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Сезонные ошибки", "MAE по месяцу теста")
        values = breakdown[breakdown.group == "month"].copy()
        values["value"] = pd.to_numeric(values.value)
        chart(px.bar(values.sort_values("value"), x="value", y="MAE", labels={"value": "Месяц", "MAE": "МВт"}), 250)
    with b, st.container(border=True):
        panel_title("Распределение ошибок", "Факт − прогноз, выбранная модель")
        errors = best_predictions.actual - best_predictions.predicted
        fig = px.histogram(x=errors, nbins=50, labels={"x": "Ошибка, МВт"})
        fig.update_yaxes(title="Число часов")
        chart(fig, 250)
    interval = summary["bootstrap"]
    st.caption(f"Парный bootstrap суточных MAE: модель − baseline = {interval['mae_difference_mw']:.1f} МВт; 95% интервал [{interval['lower_95']:.1f}; {interval['upper_95']:.1f}]. Отрицательная разница означает улучшение. Независимый bootstrap по дням не полностью учитывает автокорреляцию.")
    st.download_button("Скачать все метрики", (ROOT / "reports/metrics.csv").read_bytes(), "metrics.csv", "text/csv")

elif page == "Интерпретация":
    title("Почему модель так прогнозирует", "Вклад истории, календаря и погодных факторов. Объяснения помогают понимать модель, но не доказывают причинность.")
    importance = pd.read_csv(ROOT / "reports/feature_importance.csv")
    shap = pd.read_csv(ROOT / "reports/shap_importance.csv")
    note(f"Permutation importance относится к выбранной модели {summary['best_model']} / {SCENARIO_NAMES[summary['best_scenario']]}. SHAP рассчитан отдельно для CatBoost / {SCENARIO_NAMES[summary['shap_scenario']]}.")
    a, b = st.columns(2)
    with a, st.container(border=True):
        panel_title("Важность признаков", "Прирост MAE при перестановке · 3 повтора")
        values = importance.head(15).iloc[::-1]
        fig = px.bar(values, x="importance_mw", y="feature", orientation="h", error_x="std_mw",
                     labels={"importance_mw": "Прирост MAE, МВт", "feature": ""})
        chart(fig, 470)
    with b, st.container(border=True):
        panel_title("SHAP · CatBoost", "Средний абсолютный вклад в прогноз, МВт")
        values = shap.head(15).iloc[::-1]
        fig = px.bar(values, x="mean_abs_shap_mw", y="feature", orientation="h",
                     labels={"mean_abs_shap_mw": "Средний |SHAP|, МВт", "feature": ""})
        fig.update_traces(marker_color=BLUE)
        chart(fig, 470)
    def group_name(feature):
        if feature.startswith("load_"):
            return "История нагрузки"
        if feature.startswith(("past_", "oracle_")):
            return "Погодные факторы"
        return "Календарь"
    shap["group"] = shap.feature.map(group_name)
    grouped = shap.groupby("group").mean_abs_shap_mw.sum().reset_index()
    with st.container(border=True):
        panel_title("Группы факторов", "Сумма средних абсолютных вкладов SHAP · не доля причинного эффекта")
        chart(px.bar(grouped, x="group", y="mean_abs_shap_mw", labels={"group": "", "mean_abs_shap_mw": "МВт"}), 240)
    with st.expander("Локальное объяснение SHAP"):
        values = np.load(ROOT / "reports/shap_values.npz")
        sample = st.slider("Номер наблюдения в диагностической выборке", 0, len(values["values"]) - 1, 0, key="shap_sample")
        local = pd.DataFrame({"Признак": values["names"], "Значение": values["features"][sample],
                              "SHAP, МВт": values["values"][sample, :-1]})
        local["magnitude"] = local["SHAP, МВт"].abs()
        st.dataframe(local.sort_values("magnitude", ascending=False).drop(columns="magnitude").head(15).round(2), hide_index=True, width="stretch")
        base_value = values["values"][sample, -1]
        st.caption(f"Базовый прогноз {base_value:.1f} МВт + сумма вкладов = {values['values'][sample].sum():.1f} МВт. Выборка — один час каждого UTC-дня test.")
    st.caption("Перестановка и SHAP рассчитаны на систематической выборке: один час каждого UTC-дня test. Коррелированные лаги могут делить важность. Это диагностика с ограниченным покрытием горизонтов.")

else:
    title("От данных к решению", "Data Science Capstone: прогноз нагрузки на электросеть с учётом погоды и календарных факторов.")
    a, b = st.columns([1.5, 1])
    with a, st.container(border=True):
        panel_title("Практическая задача")
        st.write("Почасовой прогноз помогает оператору энергосистемы планировать генерацию и резервы, оценивать ожидаемый пик и сравнивать модели до их применения в работе.")
        st.write("Исследование проверяет, что добавляют календарь, история нагрузки и погода. Основной результат — честное сравнение в едином протоколе выпуска 24-часового прогноза.")
        st.markdown("**Критерий успеха:** снизить test MAE минимум на 10% относительно недельного Seasonal Naive. Денежный эффект без данных о стоимости резервов не оценивается.")
        st.markdown(f"**Полученный результат:** снижение MAE на {gain:.1f}%. Основная модель — {summary['best_model']}.")
    with b, st.container(border=True):
        panel_title("Паспорт исследования")
        st.write(f"Период: 2015–2018 · {quality['hours']:,} часов")
        st.write("Города: " + ", ".join(quality["cities"]))
        st.write("Цель: total load actual, МВт")
        st.write("Горизонт: 24 часа от 00:00 UTC")
        st.write(f"Полных test-блоков: {summary['split_days']['test']}")
    tabs = st.tabs(["Методология", "Качество данных", "Ограничения", "Источники и материалы"])
    with tabs[0]:
        st.markdown("""1. Проверка CSV, UTC-ключей, повторов и физически недопустимых значений.
2. Погода пяти городов усредняется равными весами; температура переводится в °C.
3. Train: 2015–2016, validation: 2017, test: 2018. Параметры выбираются по validation MAE.
4. Лаги 24/48/168/336 часов; rolling-статистики фиксируются на origin. Все доступные входы строго раньше прогноза.
5. Seasonal Naive, Ridge, Random Forest и CatBoost; MAE, RMSE, MAPE и sMAPE.
6. Отдельный oracle-сценарий использует фактическую будущую погоду; он не входит в выбор основной модели.
7. Сохранение моделей, прогнозов, ошибок, permutation importance и CatBoost SHAP.
""")
    with tabs[1]:
        st.json(quality)
    with tabs[2]:
        st.markdown("""- Данных после 2018 года нет; текущий прогноз без свежей истории недоступен.
- Протокол использует блоки от 00:00 UTC, а не местные календарные сутки.
- Предполагается, что нагрузка до origin опубликована без задержки.
- Погода агрегируется равными весами; региональные праздники не учитываются.
- Один временной holdout; несколько rolling folds и производственная устойчивость ещё не исследованы.
- Оценены только полные блоки с наблюдаемой целью. Пропуски могут создавать смещение выборки.
- Bootstrap и importance имеют ограничения выборки; причинные эффекты и финансовая экономия не установлены.
- Dockerfile подготовлен; контейнерный запуск и облачная публикация отдельно не проверены.
""")
    with tabs[3]:
        st.markdown("[Датасет на Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) · [Red Eléctrica](https://www.ree.es/en/operation/electricity-system/demand-and-production-curves-in-real-time) · [Hong & Fan, 2016](https://www.sciencedirect.com/science/article/pii/S0169207015001508)")
        st.download_button("Исследовательский отчёт", (ROOT / "reports/final_report.md").read_bytes(), "final_report.md", "text/markdown")
        st.download_button("Материал выступления · 10 минут", (ROOT / "reports/presentation_notes.md").read_bytes(), "presentation_notes.md", "text/markdown")
    with st.expander("Полный исследовательский отчёт"):
        st.markdown((ROOT / "reports/final_report.md").read_text())

st.markdown('<div style="margin-top:24px;padding-top:18px;border-top:1px solid #213044;color:#7489a2;font-size:11px;">'
            'GridScope · Electricity Load Forecasting · Историческая демонстрация на реальных данных Испании</div>', unsafe_allow_html=True)

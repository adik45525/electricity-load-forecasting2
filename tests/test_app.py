import pytest
from streamlit.testing.v1 import AppTest
from src.data_loader import ROOT


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60)
    instance.run()
    assert not instance.exception
    return instance


@pytest.mark.parametrize("page", ["Обзор", "Прогноз", "Аналитика данных", "Качество моделей", "Интерпретация", "О проекте"])
def test_every_page_renders(app, page):
    app.radio(key="page").set_value(page).run()
    assert not app.exception, list(app.exception)
    assert len(app.title) == 1


def test_forecast_date_interaction(app):
    app.radio(key="page").set_value("Прогноз").run()
    selector = app.select_slider(key="forecast_date")
    selector.set_value(selector.options[0]).run()
    assert not app.exception
    selector = app.select_slider(key="forecast_date")
    selector.set_value(selector.options[-1]).run()
    assert not app.exception


def test_validation_interaction(app):
    app.radio(key="page").set_value("Качество моделей").run()
    app.radio(key="performance_split").set_value("validation").run()
    assert not app.exception


def test_year_interaction(app):
    app.radio(key="page").set_value("Обзор").run()
    app.selectbox(key="overview_year").set_value("2018").run()
    assert not app.exception


def test_local_shap_interaction(app):
    app.radio(key="page").set_value("Интерпретация").run()
    app.slider(key="shap_sample").set_value(10).run()
    assert not app.exception

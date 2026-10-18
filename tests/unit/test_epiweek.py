"""Testes unitários das semanas epidemiológicas (src/core/epiweek.py)."""
from datetime import date

import pytest

from src.core import epiweek


def test_semana_comeca_no_domingo():
    inicio = epiweek.week_start(2019, 30)
    assert inicio.weekday() == 6  # domingo


def test_semana_1_de_2019_comeca_em_30_de_dezembro_de_2018():
    # 1º/jan/2019 foi uma terça: a semana 1 começa no domingo anterior.
    assert epiweek.week_start(2019, 1) == date(2018, 12, 30)


def test_inicio_de_janeiro_pode_pertencer_ao_ano_anterior():
    # 1º/jan/2016 foi uma sexta: a semana tem só 2 dias em 2016,
    # então pertence à última semana de 2015 (a 52ª).
    assert epiweek.from_date(date(2016, 1, 1)) == (2015, 52)
    assert epiweek.from_date(date(2016, 1, 3)) == (2016, 1)


@pytest.mark.parametrize("ano, semanas", [(2014, 53), (2015, 52), (2019, 52), (2020, 53)])
def test_quantidade_de_semanas_no_ano(ano, semanas):
    assert epiweek.weeks_in_year(ano) == semanas


def test_add_weeks_atravessa_a_virada_do_ano():
    assert epiweek.add_weeks(2019, 52, 1) == (2020, 1)
    assert epiweek.add_weeks(2020, 1, -1) == (2019, 52)


def test_parse_iso_label_valido():
    assert epiweek.parse_iso_label("2019-30") == (2019, 30)


@pytest.mark.parametrize("rotulo", ["2019-54", "2019/30", "abc", "2019-00"])
def test_parse_iso_label_invalido(rotulo):
    with pytest.raises(ValueError):
        epiweek.parse_iso_label(rotulo)


def test_parse_label_da_planilha():
    assert epiweek.parse_label("S201601") == (2016, 1)
    with pytest.raises(ValueError):
        epiweek.parse_label("S201699")

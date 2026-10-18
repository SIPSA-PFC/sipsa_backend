"""Testes unitários da ingestão de casos via CSV (src/ingestion/csv_source.py)."""
import pytest

from src.ingestion.csv_source import CsvCasesSource


def _fonte(tmp_path, conteudo):
    arquivo = tmp_path / "casos.csv"
    arquivo.write_text(conteudo, encoding="utf-8")
    return CsvCasesSource(path=arquivo, name="teste")


def test_csv_valido_gera_registros(tmp_path):
    fonte = _fonte(tmp_path, "neighborhood,epi_year,epi_week,cases\nPonta Negra,2019,30,4\nNeópolis,2019,30,\n")
    registros = list(fonte.fetch())
    assert len(registros) == 2
    assert registros[0].cases == 4 and registros[0].disease == "dengue"
    assert registros[1].cases is None  # célula vazia = desconhecido, não zero


def test_csv_sem_coluna_obrigatoria_e_rejeitado(tmp_path):
    fonte = _fonte(tmp_path, "neighborhood,epi_year,cases\nPonta Negra,2019,4\n")
    with pytest.raises(ValueError, match="missing columns"):
        list(fonte.fetch())


def test_csv_com_semana_impossivel_e_rejeitado(tmp_path):
    fonte = _fonte(tmp_path, "neighborhood,epi_year,epi_week,cases\nPonta Negra,2019,60,4\n")
    with pytest.raises(ValueError):
        list(fonte.fetch())


def test_csv_com_casos_negativos_e_rejeitado(tmp_path):
    fonte = _fonte(tmp_path, "neighborhood,epi_year,epi_week,cases\nPonta Negra,2019,30,-1\n")
    with pytest.raises(ValueError, match="negative"):
        list(fonte.fetch())

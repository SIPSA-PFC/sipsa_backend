"""Testes unitários das métricas de avaliação (src/ml/utils.py)."""
import numpy as np
import pytest

from src.ml.utils import persistence_baseline, rmse_and_r


def test_previsao_perfeita_tem_rmse_zero_e_correlacao_um():
    y = np.array([1.0, 3.0, 2.0, 5.0])
    rmse, r = rmse_and_r(y, y)
    assert rmse == 0
    assert r == pytest.approx(1.0)


def test_rmse_calculado_a_mao():
    # erros 1, -1, 1, 1  ->  RMSE = 1
    rmse, _ = rmse_and_r([2, 4, 2, 4], [3, 3, 3, 5])
    assert rmse == pytest.approx(1.0)


def test_baseline_de_persistencia_repete_o_valor_de_semanas_atras():
    serie = np.array([10, 20, 30, 40, 50])
    assert persistence_baseline(serie, targets=np.array([3, 4]), lead=2).tolist() == [20, 30]

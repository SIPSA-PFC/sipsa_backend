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


def test_erro_padrao_divide_por_n_menos_1_como_o_matlab():
    from src.ml.utils import compute_rmse
    y_test = np.array([1.0, 1.0])
    # RMSE por repetição: 1 e 3 (previsões 2 e 4, em log2: 1 e 2).
    y_pred_log2 = np.log2(np.array([[2.0, 4.0], [2.0, 4.0]]))
    media, erro_padrao = compute_rmse(y_pred_log2, y_test)
    assert media == pytest.approx(2.0)
    # std com N-1 de [1, 3] = sqrt(2); dividido por sqrt(2 repetições) = 1.
    assert erro_padrao == pytest.approx(1.0)

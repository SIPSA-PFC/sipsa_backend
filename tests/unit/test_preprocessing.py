"""Testes unitários do pré-processamento do modelo (src/ml/preprocessing.py)."""
import numpy as np
import pytest

from src.ml import preprocessing as pp


@pytest.mark.parametrize("lag, esperado", [(1, [1]), (3, [1, 2, 3]), (6, [4, 5, 6])])
def test_janelas_de_entrada(lag, esperado):
    assert pp.lag_offsets(lag) == esperado


def test_lag_nao_suportado_e_rejeitado():
    with pytest.raises(ValueError):
        pp.lag_offsets(2)


def test_horizonte_maximo_de_previsao():
    # Com lag 6 a entrada mais recente tem 4 semanas: dá para prever 4 semanas à frente.
    assert pp.max_horizon(6) == 4
    assert pp.max_horizon(1) == 1


def test_divisao_treino_teste_arredonda_como_o_matlab():
    # 0.5 * 5 = 2.5: o MATLAB arredonda para 3, o round() do Python daria 2.
    assert pp.n_train_samples(5, train_split=0.5) == 3
    assert pp.n_train_samples(208) == 166  # 80% das 208 semanas do estudo


def test_log2_rejeita_zero_e_valores_faltantes():
    with pytest.raises(pp.MissingDataError):
        pp.log2_transform(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(pp.MissingDataError):
        pp.log2_transform(np.array([1.0, np.nan]))


def test_log2_e_transformacao_inversa():
    x = np.array([1.0, 2.0, 8.0])
    assert np.allclose(pp.log2_transform(x), [0, 1, 3])
    assert np.allclose(pp.back_transform(pp.log2_transform(x)), x)


def test_matriz_de_entrada_usa_as_semanas_anteriores():
    serie = np.arange(10, dtype=float)       # valor = índice da semana
    entradas = pp.build_inputs(serie, np.array([6, 7]), lag=6)
    assert entradas.tolist() == [[2, 3], [1, 2], [0, 1]]  # semanas t-4, t-5, t-6


def test_semana_sem_janela_completa_e_rejeitada():
    with pytest.raises(ValueError):
        pp.build_inputs(np.arange(10.0), np.array([3]), lag=6)

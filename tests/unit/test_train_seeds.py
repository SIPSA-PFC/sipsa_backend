"""Testes unitários das sementes de treino (src/ml/train.py)."""
from src.ml import train


def test_semente_nao_depende_das_outras_configuracoes():
    # edi_lag6 tem a mesma semente treinada sozinha ou na execução completa.
    sozinha = train.parse_configs("edi:6")
    completa = train.parse_configs(None)
    assert sozinha == [("edi", 6)]
    assert ("edi", 6) in completa
    assert train.config_seed(1, "edi", 6, 0) == 109_000


def test_sementes_da_execucao_completa_nao_mudaram():
    # Antes: base * 100000 + posição na lista * 1000 + rep, com a lista padrão.
    for ci, (input_type, lag) in enumerate(train.parse_configs(None)):
        for rep in (0, 29):
            assert train.config_seed(1, input_type, lag, rep) == 100_000 + ci * 1_000 + rep


def test_sementes_sao_unicas():
    sementes = [train.config_seed(1, i, lag, rep) for i, lag in train.ALL_CONFIGS for rep in range(30)]
    assert len(sementes) == len(set(sementes))

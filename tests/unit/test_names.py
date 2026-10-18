"""Testes unitários da padronização de nomes de bairros (src/core/names.py)."""
import pytest

from src.core.names import name_key


@pytest.mark.parametrize("a, b", [
    ("Neópolis", "Neopolis"),
    ("Dix-sept Rosado", "Dixsept Rosado"),
    ("PONTA NEGRA", "Ponta Negra"),
    ("Filipe Camarão", "Felipe Camarão"),  # alias de grafia (IBGE 2010)
])
def test_grafias_diferentes_geram_a_mesma_chave(a, b):
    assert name_key(a) == name_key(b)


def test_bairros_diferentes_geram_chaves_diferentes():
    assert name_key("Lagoa Nova") != name_key("Lagoa Seca")

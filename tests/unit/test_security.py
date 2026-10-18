"""Testes unitários de senhas e tokens JWT (src/core/security.py)."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from src.core import security
from src.core.config import settings


def test_hash_e_verificacao_de_senha():
    h = security.hash_password("senha-segura-123")
    assert h != "senha-segura-123"  # nunca armazena a senha em texto puro
    assert security.verify_password("senha-segura-123", h)
    assert not security.verify_password("senha-errada", h)


def test_senha_curta_e_rejeitada():
    with pytest.raises(ValueError):
        security.hash_password("1234567")


def test_senha_acima_de_72_bytes_e_rejeitada():
    with pytest.raises(ValueError):
        security.hash_password("a" * 73)


def test_hash_malformado_nao_quebra_a_verificacao():
    assert security.verify_password("qualquer", "hash-invalido") is False


def test_token_gerado_contem_usuario_e_perfil():
    token = security.create_access_token("ana", "admin")
    payload = security.decode_access_token(token)
    assert payload["sub"] == "ana"
    assert payload["role"] == "admin"


def test_token_adulterado_e_rejeitado():
    token = security.create_access_token("ana", "user")
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(token[:-2] + "xx")


def test_token_assinado_com_outra_chave_e_rejeitado():
    falso = jwt.encode({"sub": "ana", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       "outra-chave-qualquer-com-32-caracteres!!", algorithm="HS256")
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(falso)


def test_token_expirado_e_rejeitado():
    expirado = jwt.encode({"sub": "ana", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
                          settings.secret_key, algorithm=settings.algorithm)
    with pytest.raises(security.InvalidToken):
        security.decode_access_token(expirado)

"""
Testes de integração da API: requisições HTTP reais contra o app FastAPI,
com banco SQLite temporário (configurado em tests/conftest.py).
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from src.api.models.db_models import District, Neighborhood, User
from src.core.database import AsyncSessionLocal, Base, engine
from src.core.security import hash_password
from src.main import app

SENHA = "senha-de-teste-123"


async def _preparar_banco():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as s:
        distrito = District(id=1, name="Sul")
        s.add_all([
            distrito,
            Neighborhood(id=1, code="001", name="Ponta Negra", district=distrito),
            User(username="analista", password_hash=hash_password(SENHA), role="user"),
            User(username="inativo", password_hash=hash_password(SENHA), role="user", is_active=False),
        ])
        await s.commit()


@pytest.fixture(scope="module")
def client():
    asyncio.run(_preparar_banco())
    with TestClient(app) as c:  # o "with" executa o lifespan (inicialização) do app
        yield c


def _login(client, usuario="analista", senha=SENHA):
    return client.post("/auth/token", data={"username": usuario, "password": senha})


@pytest.fixture(scope="module")
def auth(client):
    token = _login(client).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── Disponibilidade ──────────────────────────────────────────────────────────

def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ── Autenticação (RN07) ──────────────────────────────────────────────────────

def test_login_com_credenciais_validas_retorna_token(client):
    r = _login(client)
    assert r.status_code == 200
    assert r.json()["token_type"].lower() == "bearer"
    assert r.json()["access_token"]


def test_login_com_senha_errada_retorna_401(client):
    assert _login(client, senha="senha-errada").status_code == 401


def test_login_de_usuario_inexistente_retorna_401(client):
    assert _login(client, usuario="ninguem").status_code == 401


def test_usuario_desativado_nao_consegue_entrar(client):
    assert _login(client, usuario="inativo").status_code == 401


def test_auth_me_retorna_usuario_logado(client, auth):
    r = client.get("/auth/me", headers=auth)
    assert r.status_code == 200
    assert r.json() == {"username": "analista", "role": "user"}


# ── Controle de acesso ───────────────────────────────────────────────────────

def test_rota_protegida_sem_token_retorna_401(client):
    assert client.get("/bairros").status_code == 401


def test_rota_protegida_com_token_invalido_retorna_401(client):
    assert client.get("/bairros", headers={"Authorization": "Bearer token-falso"}).status_code == 401


# ── Consulta de dados ────────────────────────────────────────────────────────

def test_lista_de_bairros(client, auth):
    r = client.get("/bairros", headers=auth)
    assert r.status_code == 200
    bairro = r.json()[0]
    assert bairro["name"] == "Ponta Negra" and bairro["district_name"] == "Sul"


def test_lista_de_distritos(client, auth):
    r = client.get("/distritos", headers=auth)
    assert r.status_code == 200
    assert r.json()[0]["neighborhood_ids"] == [1]


def test_casos_de_bairro_inexistente_retorna_404(client, auth):
    assert client.get("/bairros/999/casos", headers=auth).status_code == 404


def test_semana_epidemiologica_invalida_retorna_422(client, auth):
    assert client.get("/bairros/1/casos?as_of=2019-60", headers=auth).status_code == 422


# ── Previsão ─────────────────────────────────────────────────────────────────

def test_previsao_sem_modelo_treinado_retorna_503(client, auth):
    # Nenhum modelo foi treinado no banco de teste: a API deve informar
    # indisponibilidade em vez de falhar com erro interno.
    assert client.get("/cidade/previsao", headers=auth).status_code == 503


def test_previsao_com_lag_invalido_retorna_422(client, auth):
    assert client.get("/cidade/previsao?lag=2", headers=auth).status_code == 422

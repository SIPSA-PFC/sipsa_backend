# Instalação e configuração

## Requisitos

- Python 3.10 ou mais novo
- Os arquivos de dados do artigo (abaixo)
- Opcional: Docker, para usar PostgreSQL em vez de SQLite

Não é preciso GPU: o treino completo leva cerca de 80 minutos em 2 núcleos de CPU.

## Ambiente virtual e dependências

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
source .venv/bin/activate           # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Arquivo `.env`

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Cole a chave gerada em `SECRET_KEY=` (sem aspas). Ela é **obrigatória**: a API, o ETL e o
Alembic não iniciam sem uma chave de pelo menos 32 caracteres, e textos de exemplo são
recusados. O `.env` não vai para o git; em outro computador, gere outra chave.

| Variável | Padrão | Uso |
|---|---|---|
| `DATABASE_URL` | SQLite em `data/sipsa_test.db` | Banco de dados (veja abaixo) |
| `SECRET_KEY` | — (obrigatória) | Assinatura dos tokens de login |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Validade do token |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Endereços do frontend que podem chamar a API |
| `RISK_PERCENTILES` | `[50, 90]` | Limites de risco moderado e alto (veja [API.md](API.md#nível-de-risco)) |
| `RISK_MIN_CASES` | `1` | Abaixo disso o risco é sempre baixo |

## Arquivos de dados

Coloque em `data/raw/` antes de rodar o ETL. Eles não vão para o git e vêm do repositório do
artigo (licença CC BY-NC-SA 4.0; detalhes em [DATA_SOURCES.md](DATA_SOURCES.md)).

| Arquivo | Usado por |
|---|---|
| `Dados_Modelagem.xlsx` | ETL e treino |
| `DADOS GERAIS_01.xlsx` | ETL |
| `Rearranged_Data.mat` | ETL e treino |
| `Models_1DLSTM_2022_02_04_EvalResults.mat` | Só a rota `/resultados` |

O ETL também lê `data/reference/neighborhoods.csv` (versionado), que liga cada bairro entre as
fontes: linha nos dados de ovitrampa, código municipal, distrito sanitário e coordenadas.
Veja [data/reference/README.md](../data/reference/README.md).

## Banco de dados

```bash
alembic upgrade head        # cria ou atualiza as tabelas
python -m src.etl.migrate   # carrega os dados de data/raw/
```

O ETL pode ser rodado de novo a qualquer momento: atualiza os dados no lugar, sem duplicar, e
apaga as previsões guardadas, tudo numa única transação.

### SQLite ou PostgreSQL

Só muda uma linha no `.env`; o código é o mesmo.

```bash
# SQLite (padrão, sem servidor)
DATABASE_URL=sqlite+aiosqlite:///./data/sipsa_test.db

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://sipsa:SENHA@localhost:5432/sipsa
```

Para subir um PostgreSQL local com Docker:

```bash
docker run --name sipsa-db -e POSTGRES_USER=sipsa -e POSTGRES_PASSWORD=SENHA \
  -e POSTGRES_DB=sipsa -p 5432:5432 -d postgres:15
```

Depois rode `alembic upgrade head` e `python -m src.etl.migrate` de novo.

### Mudanças no esquema

Depois de alterar `src/api/models/db_models.py`, gere uma migration e revise antes de commitar:

```bash
alembic revision --autogenerate -m "descreva a mudança"
```

Bancos criados antes do Alembic: apague o arquivo SQLite (ou as tabelas no PostgreSQL) e rode
`alembic upgrade head` e `python -m src.etl.migrate`.

## Contas de usuário

Há dois papéis, `admin` e `user`. Hoje têm o mesmo acesso; `require_role(...)` em
`src/api/controllers/auth.py` restringe uma rota quando isso mudar. As senhas são pedidas no
terminal e nunca ficam em arquivos.

```bash
python -m src.cli create-user NOME [--role admin|user]
python -m src.cli set-password NOME
python -m src.cli set-role NOME admin|user
python -m src.cli deactivate NOME        # os tokens existentes param de funcionar na hora
python -m src.cli activate NOME
python -m src.cli list-users
```

## Treino dos modelos

As rotas de previsão usam modelos treinados antes, fora da API. O treino salva os modelos em
`data/models/<run_id>/` (fora do git) e os registra na tabela `model_run`.

```bash
python -m src.ml.train --dev        # 3 repetições por configuração, ~8 min
python -m src.ml.train              # 30 repetições, como no artigo, ~80 min
python -m src.ml.evaluate --run data/models/<run_id>    # compara com o artigo e os baselines
```

- A API procura os modelos ao iniciar: reinicie-a depois de treinar.
- Para cada configuração, a API usa o treino com mais repetições e, entre esses, o mais
  recente. Um treino `--dev` (3 repetições) só é usado enquanto não houver um completo.
- Para usar um treino já feito num banco novo ou recriado:
  `python -m src.ml.train --register data/models/<run_id>`.
- Sem banco nem API, `python -m src.ml.evaluate` refaz as figuras do artigo a partir dos
  resultados MATLAB; elas ficam em `outputs/`.

Resultados e decisões do treino: [REPRODUCTION.md](REPRODUCTION.md).

## Rodar a API

```bash
uvicorn src.main:app --reload --port 8000
```

Documentação interativa em http://localhost:8000/docs. Na inicialização aparece a linha
`Trained models: ...` com os modelos encontrados.

## Arquivos que nunca devem ir para o git

Todos já estão no `.gitignore`.

| Arquivo ou pasta | Motivo | Como recriar |
|---|---|---|
| `.env` | Contém a chave secreta | Copiar `.env.example` |
| `data/raw/` | Arquivos de origem grandes | Baixar do repositório do artigo |
| `data/*.db` | Banco gerado | `alembic upgrade head` + `python -m src.etl.migrate` |
| `data/models/` | Modelos treinados | `python -m src.ml.train` |
| `outputs/` | Figuras e relatórios | `python -m src.ml.evaluate` |
| `.venv/` | Ambiente local | `pip install -r requirements.txt` |

## Problemas comuns

**`SECRET_KEY` / `Field required` ao rodar alembic, ETL ou API.**
Falta o `.env` ou a chave está vazia. Veja [Arquivo `.env`](#arquivo-env).

**`No module named 'greenlet'`.**
O SQLAlchemy precisa do extra assíncrono: `pip install "sqlalchemy[asyncio]"` (já está no
`requirements.txt`).

**Windows: `OSError` com "Long Path support" ao instalar o TensorFlow.**
O caminho passou de 260 caracteres. No PowerShell **como administrador**:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Feche os terminais, apague o `.venv` e instale de novo. Sem acesso de administrador, crie o
ambiente num caminho curto, por exemplo `python -m venv C:\venvs\sipsa`.

**Windows: "execução de scripts foi desabilitada" ao ativar o venv.**
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, uma vez.

**`KeyError: '0001'` no alembic.**
Falta um arquivo em `migrations/versions/` ou ele está sem a extensão `.py`.

**`/previsao` ou `/dashboard` responde 503.**
Não há modelo treinado e registrado para a configuração pedida. Treine (ou use `--register`)
e reinicie a API.

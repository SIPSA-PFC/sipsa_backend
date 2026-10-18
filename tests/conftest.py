"""
Configuração compartilhada dos testes.

As variáveis de ambiente são definidas ANTES de importar qualquer módulo de
`src`, porque `src.core.config.settings` e o engine do banco são criados no
momento da importação. Assim os testes usam um SQLite temporário e uma
SECRET_KEY própria, sem depender do .env nem do PostgreSQL.
"""
import os
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="sipsa_tests_"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DIR / 'test.db'}"
os.environ["SECRET_KEY"] = "chave-de-teste-com-mais-de-32-caracteres-0123456789"
os.environ["CORS_ORIGINS"] = '["http://localhost:3000"]'

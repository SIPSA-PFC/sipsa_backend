"""
api/controllers/resultados.py
------------------------------
GET /resultados

Returns pre-computed model evaluation results directly from the .mat file.
No database connection, no training, no authentication required.
Safe to call immediately after starting the API.
"""

from fastapi import APIRouter
from src.core.config import ROOT
from src.api.models.schemas import ResultadosOut
from src.api.services.resultados_service import ResultadosService

router = APIRouter(prefix="/resultados", tags=["resultados"])

# Path to the pre-computed MATLAB eval results file, resolved from the repo
# root so the API finds it regardless of the directory uvicorn is started from.
_MAT_PATH = ROOT / "data" / "raw" / "Models_1DLSTM_2022_02_04_EvalResults.mat"


@router.get("", response_model=ResultadosOut)
async def get_resultados() -> ResultadosOut:
    """
    Returns pre-computed LSTM model evaluation results.

    No training required — reads directly from the original MATLAB .mat file.

    Response includes:
    - **metrics**: RMSE and Pearson r for all 10 model configurations
    - **best_models**: full predicted vs actual time series for D→D and O→D best models
    - **best_r**: highest correlation coefficient achieved (0.899)
    - **best_rmse**: lowest RMSE achieved
    - **test_period_start / end**: date range of the test window (Mar–Dec 2019)
    - **n_reps**: number of independent training repetitions (30)
    """
    return ResultadosService(_MAT_PATH).get_resultados()
"""
api/services/validation_service.py
Validates requests before any DB query or ML call runs.
"""
from fastapi import HTTPException, status
from src.api.repositories.neighborhood_repository import NeighborhoodRepository
from src.api.repositories.dengue_repository import DengueRepository


class ValidationService:
    def __init__(self, neighborhood_repo: NeighborhoodRepository, dengue_repo: DengueRepository):
        self.neighborhood_repo = neighborhood_repo
        self.dengue_repo       = dengue_repo

    async def assert_exists(self, neighborhood_id: int) -> None:
        n = await self.neighborhood_repo.get_by_id(neighborhood_id)
        if n is None:
            raise HTTPException(status_code=404, detail=f"Neighborhood {neighborhood_id} not found.")

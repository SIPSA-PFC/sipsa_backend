"""forecast replaces model prediction

Stores forecasts per model run, area (city / district / neighborhood),
as-of week and target week (refactor/offline-inference). model_prediction
was only a cache of on-request training and is dropped.

Revision ID: 0004
Revises: 0003
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecast",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_run_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=12), nullable=False),
        sa.Column("district_id", sa.Integer(), nullable=True),
        sa.Column("neighborhood_id", sa.Integer(), nullable=True),
        sa.Column("as_of_week_id", sa.Integer(), nullable=False),
        sa.Column("target_week_id", sa.Integer(), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("predicted", sa.Float(), nullable=False),
        sa.Column("lower", sa.Float(), nullable=True),
        sa.Column("upper", sa.Float(), nullable=True),
        sa.Column("method", sa.String(length=10), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.CheckConstraint(
            "(scope = 'city' AND district_id IS NULL AND neighborhood_id IS NULL)"
            " OR (scope = 'district' AND district_id IS NOT NULL AND neighborhood_id IS NULL)"
            " OR (scope = 'neighborhood' AND neighborhood_id IS NOT NULL AND district_id IS NULL)",
            name=op.f("ck_forecast_area_matches_scope"),
        ),
        sa.CheckConstraint("method IN ('model', 'split')", name=op.f("ck_forecast_method")),
        sa.CheckConstraint("scope IN ('city', 'district', 'neighborhood')", name=op.f("ck_forecast_scope")),
        sa.CheckConstraint("horizon >= 1", name=op.f("ck_forecast_horizon_positive")),
        sa.ForeignKeyConstraint(["as_of_week_id"], ["epi_week.id"], name=op.f("fk_forecast_as_of_week_id_epi_week")),
        sa.ForeignKeyConstraint(["district_id"], ["district.id"], name=op.f("fk_forecast_district_id_district")),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_run.id"], name=op.f("fk_forecast_model_run_id_model_run"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["neighborhood_id"], ["neighborhood.id"], name=op.f("fk_forecast_neighborhood_id_neighborhood")),
        sa.ForeignKeyConstraint(["target_week_id"], ["epi_week.id"], name=op.f("fk_forecast_target_week_id_epi_week")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecast")),
    )
    op.create_index(op.f("ix_forecast_model_run_id"), "forecast", ["model_run_id"], unique=False)
    # Expression index (Alembic does not autogenerate these): the city has no
    # area id, and NULLs never collide in a unique index.
    op.create_index(
        "uq_forecast_key", "forecast",
        ["model_run_id", "scope", sa.text("coalesce(district_id, 0)"), sa.text("coalesce(neighborhood_id, 0)"),
         "as_of_week_id", "target_week_id"],
        unique=True,
    )
    op.drop_table("model_prediction")


def downgrade() -> None:
    op.create_table(
        "model_prediction",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("neighborhood_id", sa.Integer(), nullable=False),
        sa.Column("epi_year", sa.Integer(), nullable=False),
        sa.Column("epi_week", sa.Integer(), nullable=False),
        sa.Column("week_start", sa.Date(), nullable=False),
        sa.Column("predicted_cases", sa.Float(), nullable=False),
        sa.Column("lower_bound", sa.Float(), nullable=True),
        sa.Column("upper_bound", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("input_type", sa.String(length=10), nullable=False),
        sa.Column("lag", sa.Integer(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.CheckConstraint("input_type IN ('dengue', 'edi')", name=op.f("ck_model_prediction_input_type")),
        sa.ForeignKeyConstraint(["neighborhood_id"], ["neighborhood.id"], name=op.f("fk_model_prediction_neighborhood_id_neighborhood")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_prediction")),
        sa.UniqueConstraint("neighborhood_id", "epi_year", "epi_week", "model_version", "input_type", "lag",
                            name="uq_model_prediction_config"),
    )
    op.drop_index("uq_forecast_key", table_name="forecast")
    op.drop_index(op.f("ix_forecast_model_run_id"), table_name="forecast")
    op.drop_table("forecast")

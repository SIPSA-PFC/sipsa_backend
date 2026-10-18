"""model run

Registry of trained model configurations (feat/ml-paper-reproduction).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-16 08:43:47.250250
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('model_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=60), nullable=False),
    sa.Column('input_type', sa.String(length=10), nullable=False),
    sa.Column('lag', sa.Integer(), nullable=False),
    sa.Column('train_from_week_id', sa.Integer(), nullable=False),
    sa.Column('train_to_week_id', sa.Integer(), nullable=False),
    sa.Column('n_reps', sa.Integer(), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=True),
    sa.Column('artifact_path', sa.String(length=500), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    sa.CheckConstraint("input_type IN ('dengue', 'edi')", name=op.f('ck_model_run_input_type')),
    sa.ForeignKeyConstraint(['train_from_week_id'], ['epi_week.id'], name=op.f('fk_model_run_train_from_week_id_epi_week')),
    sa.ForeignKeyConstraint(['train_to_week_id'], ['epi_week.id'], name=op.f('fk_model_run_train_to_week_id_epi_week')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_run')),
    sa.UniqueConstraint('model_version', 'input_type', 'lag', name=op.f('uq_model_run_model_version_input_type_lag'))
    )


def downgrade() -> None:
    op.drop_table('model_run')

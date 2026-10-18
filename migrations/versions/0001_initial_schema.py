"""initial schema

Target schema from refactor/db-schema: provenance (data_source), health
districts, epidemiological-week calendar, and weekly dengue / ovitrap tables
keyed by epi_week. model_prediction is the interim forecast cache.

Revision ID: 0001
Revises: 
Create Date: 2026-10-16 08:32:44.473246
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('data_source',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('file_or_url', sa.String(length=500), nullable=True),
    sa.Column('license', sa.String(length=120), nullable=True),
    sa.Column('citation', sa.Text(), nullable=True),
    sa.Column('imported_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_data_source')),
    sa.UniqueConstraint('name', name=op.f('uq_data_source_name'))
    )
    op.create_table('district',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=60), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_district')),
    sa.UniqueConstraint('name', name=op.f('uq_district_name'))
    )
    op.create_table('epi_week',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('epi_year', sa.Integer(), nullable=False),
    sa.Column('epi_week', sa.Integer(), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=False),
    sa.CheckConstraint('epi_week BETWEEN 1 AND 53', name=op.f('ck_epi_week_week_range')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_epi_week')),
    sa.UniqueConstraint('epi_year', 'epi_week', name=op.f('uq_epi_week_epi_year_epi_week')),
    sa.UniqueConstraint('start_date', name=op.f('uq_epi_week_start_date'))
    )
    op.create_table('neighborhood',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code', sa.String(length=10), nullable=True),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('district_id', sa.Integer(), nullable=True),
    sa.Column('latitude', sa.Float(), nullable=True),
    sa.Column('longitude', sa.Float(), nullable=True),
    sa.Column('coord_source_id', sa.Integer(), nullable=True),
    sa.Column('coord_retrieved_at', sa.Date(), nullable=True),
    sa.ForeignKeyConstraint(['coord_source_id'], ['data_source.id'], name=op.f('fk_neighborhood_coord_source_id_data_source')),
    sa.ForeignKeyConstraint(['district_id'], ['district.id'], name=op.f('fk_neighborhood_district_id_district')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_neighborhood')),
    sa.UniqueConstraint('code', name=op.f('uq_neighborhood_code')),
    sa.UniqueConstraint('name', name=op.f('uq_neighborhood_name'))
    )
    op.create_index(op.f('ix_neighborhood_district_id'), 'neighborhood', ['district_id'], unique=False)

    op.create_table('dengue_case',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('neighborhood_id', sa.Integer(), nullable=False),
    sa.Column('epi_week_id', sa.Integer(), nullable=False),
    sa.Column('disease', sa.String(length=20), server_default='dengue', nullable=False),
    sa.Column('case_count', sa.Integer(), nullable=True),
    sa.Column('source_id', sa.Integer(), nullable=True),
    sa.CheckConstraint('case_count >= 0', name=op.f('ck_dengue_case_non_negative')),
    sa.ForeignKeyConstraint(['epi_week_id'], ['epi_week.id'], name=op.f('fk_dengue_case_epi_week_id_epi_week')),
    sa.ForeignKeyConstraint(['neighborhood_id'], ['neighborhood.id'], name=op.f('fk_dengue_case_neighborhood_id_neighborhood')),
    sa.ForeignKeyConstraint(['source_id'], ['data_source.id'], name=op.f('fk_dengue_case_source_id_data_source')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dengue_case')),
    sa.UniqueConstraint('neighborhood_id', 'epi_week_id', 'disease', name=op.f('uq_dengue_case_neighborhood_id_epi_week_id_disease'))
    )
    op.create_index(op.f('ix_dengue_case_epi_week_id'), 'dengue_case', ['epi_week_id'], unique=False)
    op.create_index(op.f('ix_dengue_case_neighborhood_id'), 'dengue_case', ['neighborhood_id'], unique=False)

    op.create_table('model_prediction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('neighborhood_id', sa.Integer(), nullable=False),
    sa.Column('epi_year', sa.Integer(), nullable=False),
    sa.Column('epi_week', sa.Integer(), nullable=False),
    sa.Column('week_start', sa.Date(), nullable=False),
    sa.Column('predicted_cases', sa.Float(), nullable=False),
    sa.Column('lower_bound', sa.Float(), nullable=True),
    sa.Column('upper_bound', sa.Float(), nullable=True),
    sa.Column('model_version', sa.String(length=50), nullable=False),
    sa.Column('input_type', sa.String(length=10), nullable=False),
    sa.Column('lag', sa.Integer(), nullable=False),
    sa.Column('generated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    sa.CheckConstraint("input_type IN ('dengue', 'edi')", name=op.f('ck_model_prediction_input_type')),
    sa.ForeignKeyConstraint(['neighborhood_id'], ['neighborhood.id'], name=op.f('fk_model_prediction_neighborhood_id_neighborhood')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_prediction')),
    sa.UniqueConstraint('neighborhood_id', 'epi_year', 'epi_week', 'model_version', 'input_type', 'lag', name='uq_model_prediction_config')
    )
    op.create_table('ovitrap_reading',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('neighborhood_id', sa.Integer(), nullable=False),
    sa.Column('epi_week_id', sa.Integer(), nullable=False),
    sa.Column('opi', sa.Float(), nullable=True),
    sa.Column('edi', sa.Float(), nullable=True),
    sa.Column('source_id', sa.Integer(), nullable=True),
    sa.CheckConstraint('opi IS NULL OR (opi >= 0 AND opi <= 1)', name=op.f('ck_ovitrap_reading_opi_range')),
    sa.ForeignKeyConstraint(['epi_week_id'], ['epi_week.id'], name=op.f('fk_ovitrap_reading_epi_week_id_epi_week')),
    sa.ForeignKeyConstraint(['neighborhood_id'], ['neighborhood.id'], name=op.f('fk_ovitrap_reading_neighborhood_id_neighborhood')),
    sa.ForeignKeyConstraint(['source_id'], ['data_source.id'], name=op.f('fk_ovitrap_reading_source_id_data_source')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ovitrap_reading')),
    sa.UniqueConstraint('neighborhood_id', 'epi_week_id', name=op.f('uq_ovitrap_reading_neighborhood_id_epi_week_id'))
    )
    op.create_index(op.f('ix_ovitrap_reading_epi_week_id'), 'ovitrap_reading', ['epi_week_id'], unique=False)
    op.create_index(op.f('ix_ovitrap_reading_neighborhood_id'), 'ovitrap_reading', ['neighborhood_id'], unique=False)

    op.create_table('socioeconomic',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('neighborhood_id', sa.Integer(), nullable=False),
    sa.Column('reference_year', sa.Integer(), nullable=False),
    sa.Column('population', sa.Integer(), nullable=True),
    sa.Column('total_income_brl', sa.Float(), nullable=True),
    sa.Column('income_per_capita_brl', sa.Float(), nullable=True),
    sa.Column('source_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['neighborhood_id'], ['neighborhood.id'], name=op.f('fk_socioeconomic_neighborhood_id_neighborhood')),
    sa.ForeignKeyConstraint(['source_id'], ['data_source.id'], name=op.f('fk_socioeconomic_source_id_data_source')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_socioeconomic')),
    sa.UniqueConstraint('neighborhood_id', 'reference_year', name=op.f('uq_socioeconomic_neighborhood_id_reference_year'))
    )


def downgrade() -> None:
    op.drop_table('socioeconomic')
    op.drop_index(op.f('ix_ovitrap_reading_neighborhood_id'), table_name='ovitrap_reading')
    op.drop_index(op.f('ix_ovitrap_reading_epi_week_id'), table_name='ovitrap_reading')

    op.drop_table('ovitrap_reading')
    op.drop_table('model_prediction')
    op.drop_index(op.f('ix_dengue_case_neighborhood_id'), table_name='dengue_case')
    op.drop_index(op.f('ix_dengue_case_epi_week_id'), table_name='dengue_case')

    op.drop_table('dengue_case')
    op.drop_index(op.f('ix_neighborhood_district_id'), table_name='neighborhood')

    op.drop_table('neighborhood')
    op.drop_table('epi_week')
    op.drop_table('district')
    op.drop_table('data_source')
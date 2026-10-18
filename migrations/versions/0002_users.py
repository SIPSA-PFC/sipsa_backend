"""users

API accounts for fix/auth: bcrypt password hash, role (admin | user), active flag.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16 08:36:44.694225
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=80), nullable=False),
    sa.Column('password_hash', sa.String(length=100), nullable=False),
    sa.Column('role', sa.String(length=10), server_default='user', nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.true(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    sa.CheckConstraint("role IN ('admin', 'user')", name=op.f('ck_users_role')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('username', name=op.f('uq_users_username'))
    )


def downgrade() -> None:
    op.drop_table('users')
"""
cli.py
------
Account management from the command line.

    python -m src.cli create-user ana --role admin     # prompts for the password
    python -m src.cli create-user bea                  # role defaults to 'user'
    python -m src.cli set-password ana
    python -m src.cli set-role bea admin
    python -m src.cli deactivate ana
    python -m src.cli activate ana
    python -m src.cli list-users

For scripts, pass --password-stdin and pipe the password in:
    printf '%s' "$PASSWORD" | python -m src.cli create-user ana --password-stdin
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from sqlalchemy import select

from src.api.models.db_models import USER_ROLES, User
from src.api.repositories.user_repository import UserRepository
from src.core.database import AsyncSessionLocal, engine
from src.core.security import hash_password, validate_new_password


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        password = sys.stdin.read().rstrip("\n")
    else:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat password: ") != password:
            raise SystemExit("Passwords do not match.")
    try:
        validate_new_password(password)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return password


async def _with_user(username: str, change) -> None:
    async with AsyncSessionLocal() as session:
        user = await UserRepository(session).get_by_username(username)
        if user is None:
            raise SystemExit(f"No user named {username!r}.")
        message = change(user)
        await session.commit()
    print(message)


async def create_user(args) -> None:
    password = _read_password(args.password_stdin)
    async with AsyncSessionLocal() as session:
        repo = UserRepository(session)
        if await repo.get_by_username(args.username):
            raise SystemExit(f"User {args.username!r} already exists.")
        await repo.add(User(username=args.username, password_hash=hash_password(password), role=args.role))
        await session.commit()
    print(f"Created {args.role} {args.username!r}.")


async def set_password(args) -> None:
    password = _read_password(args.password_stdin)
    new_hash = hash_password(password)

    def change(user: User) -> str:
        user.password_hash = new_hash
        return f"Password changed for {user.username!r}."
    await _with_user(args.username, change)


async def set_role(args) -> None:
    def change(user: User) -> str:
        user.role = args.role
        return f"{user.username!r} is now {args.role}."
    await _with_user(args.username, change)


async def set_active(args, active: bool) -> None:
    def change(user: User) -> str:
        user.is_active = active
        return f"{user.username!r} {'activated' if active else 'deactivated'}."
    await _with_user(args.username, change)


async def list_users(_args) -> None:
    async with AsyncSessionLocal() as session:
        users = (await session.execute(select(User).order_by(User.username))).scalars().all()
    if not users:
        print("No users yet. Create one with:  python -m src.cli create-user NAME --role admin")
    for u in users:
        print(f"{u.username:<24} {u.role:<6} {'active' if u.is_active else 'inactive'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="SIPSA account management")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-user", help="create an account")
    p.add_argument("username")
    p.add_argument("--role", choices=USER_ROLES, default="user")
    p.add_argument("--password-stdin", action="store_true", help="read the password from standard input")

    p = sub.add_parser("set-password", help="change an account's password")
    p.add_argument("username")
    p.add_argument("--password-stdin", action="store_true")

    p = sub.add_parser("set-role", help="change an account's role")
    p.add_argument("username")
    p.add_argument("role", choices=USER_ROLES)

    sub.add_parser("deactivate", help="block an account (existing tokens stop working)").add_argument("username")
    sub.add_parser("activate", help="unblock an account").add_argument("username")
    sub.add_parser("list-users", help="list accounts")

    args = parser.parse_args(argv)
    commands = {
        "create-user":  create_user,
        "set-password": set_password,
        "set-role":     set_role,
        "deactivate":   lambda a: set_active(a, False),
        "activate":     lambda a: set_active(a, True),
        "list-users":   list_users,
    }

    async def run() -> None:
        try:
            await commands[args.command](args)
        finally:
            await engine.dispose()
    asyncio.run(run())


if __name__ == "__main__":
    main()
"""Noxfile for performing various code checks."""

import nox  # type: ignore

locations = "pytket", "tests", "noxfile.py"

"""Nox sessions."""


@nox.session(python=["3.11", "3.10", "3.9", "3.8"])
def test(session: nox.sessions.Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["tests"]
    session.install(".")
    session.install("--upgrade", "--pre", "pytket~=1.0")
    session.install("--pre", "-r", "tests/test-requirements.txt")
    session.run("pytest", *args)


@nox.session(python=["3.11"])
def lint(session: nox.sessions.Session) -> None:
    """Lint using pylint and black"""
    args = session.posargs or locations
    session.install("-r", ".github/workflows/linting/lint-requirements.txt")
    session.run("black", *args)
    session.run("pylint", *args)


@nox.session(python=["3.11", "3.10", "3.9", "3.8"])
def type_check(session: nox.sessions.Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.install(".")
    session.install("--upgrade", "--pre", "pytket~=1.0")
    session.install("--pre", "-r", "tests/test-requirements.txt")
    session.install("mypy")
    session.run("mypy", *args)
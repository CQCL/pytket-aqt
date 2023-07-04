# Pytket Extensions

This repository contains the pytket-aqt extension, using Quantinuum's
[pytket](https://cqcl.github.io/tket/pytket/api/index.html) quantum SDK.

# pytket-aqt

[Pytket](https://cqcl.github.io/tket/pytket/api/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimisation compiler developed by Quantinuum.

`pytket-aqt` is an extension to `pytket` that allows `pytket` circuits to be
executed on AQT's quantum devices and simulators.

## Getting started

`pytket-aqt` is available for Python 3.10 and 3.11, on Linux, MacOS
and Windows. To install, run:

`pip install pytket-aqt`

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-aqt/issues).

There is also a Slack channel for discussion and support. Click [here](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#/shared-invite/email) to join.

## Development

This project uses [poetry](https://python-poetry.org/) for packaging and dependency management and
[Nox](https://nox.thea.codes/en/stable/) for task automation.

### Recommended development setup

Create a python virtual environment within the project root directory:

```shell
python -m venv .venv
```

Activate it:

```shell
#Unix systems
source .venv/bin/activate
```

```shell
#Windows
venv\Scripts\activate
```

Install development tools (e.g., Poetry, Nox):

```shell
pip install -r dev-tool-requirements.txt
```

### Local development with Nox (recommended)

[Nox](https://nox.thea.codes/en/stable/) can be used to automate various development tasks running in isolated python environments.
The following nox sessions are provided:

- `pre-commit`: run the configured pre-commit hooks within [.pre-commit-config.yaml](.pre-commit-config.yaml), this includes linting with black and pylint
- `mypy`: run type checks using mypy
- `tests`: run the unit tests
- `docs-build`: build the documentation

To run a session use:

```shell
nox -s <session_name>
```

To save time, reuse the session virtual environment using the `-r` option, i.e. `nox -rs <session_name>` (may cause errors after a dependency update).

[Pre-commit](https://pre-commit.com/) can be used to run the pre-commit hooks before each commit. This is recommended.
To set up the pre-commit hooks to run automatically on each commit run:

```shell
nox -s pre-commit -- install
```

Afterward, the [pre-configured hooks](.pre-commit-config.yaml) will run on all changed files in a commit and the commit will be
rejected if the hooks find errors. Some hooks will correct formatting issues automatically (but will still reject the commit, so that
the `git commit` command will need to be repeated).

### Local development without Nox

To install the local package, its dependencies and various development dependencies run:

```shell
poetry install --with tests,docs,mypy,pre-commit
```

from within the previously configured virtual environment.

- Run tests: `pytest tests`
- Run mypy: `mypy --explicit-package-bases pytket tests docs/conf.py docs/build-docs`
- Run pre-commit checks: `pre-commit run --all-files --show-diff-on-failure`
- Build docs: `./docs/build-docs`

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `.pylintrc`). You should run
`pylint` over any changed files before submitting a PR, to catch any issues.

If you have `nox` installed (see [Noxfile](#noxfile)) use `nox -rs lint` to run
`black` and `pylint`

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `modules/mypy-check`
(passing as a single argument the root directory of the module to test). The
script requires `mypy` 0.800 or above.

If you have `nox` installed (see [Noxfile](#noxfile)) use `nox -rs type_check` to run
`mypy`

### Tests

If you have `nox` installed (see [Noxfile](#noxfile)) use `nox -rs test` to run
all tests.

Otherwise, to run the tests for a module:

1. `cd` into that module's `tests` directory;
2. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
   the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.

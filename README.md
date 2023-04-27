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

To install an extension in editable mode, simply change to its subdirectory
within the `modules` directory, and run:

```shell
pip install -e .
```

### Noxfile

[Nox](https://nox.thea.codes/en/stable/) is used to automate various development tasks running in isolated python environments.
The following nox sessions are provided:

- `test`: run tests for all supported python versions
- `lint`: run `black` and `pylint` for all supported python versions
- `type_check`: run `mypy` for all supported python versions

To run a session, install the `nox` command (e.g., using `pipx`) and run:

```shell
nox -s <session_name>
```

To save time, reuse the session virtual environment using the `-r` option, i.e. `nox -rs <session_name>`.
The checks are designed to conform to

### Pre-commit

[Pre-commit](https://pre-commit.com/) can be used to run a configured set of git pre-commit hooks before each commit. This is recommended.
To set up the pre-commit hooks, install `pre-commit` (e.g., using `pipx`) and run:

```shell
pre-commit install
```

Afterwards the [pre-configured hooks](.pre-commit-config.yaml) will run automatically before each commit. The commit will be
rejected if the hooks find errors. Some hooks will correct formatting issues automatically (but will still reject the commit, so that
the `git commit` command will need to be repeated).

Note that some pre-commit hooks require that `nox` is installed (e.g., using `pipx`)

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

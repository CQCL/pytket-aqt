# pytket-aqt

[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

`pytket-aqt` is an extension to `pytket` that allows `pytket` circuits to be
executed on AQT's ([Alpine Quantum Technologies'](https://www.aqt.eu/)) quantum devices and simulators.

See [extension documentation](https://cqcl.github.io/pytket-aqt/api/index.html) for more.

## Getting started

`pytket-aqt` is available for Python 3.10, 3.11 and 3.12, on Linux, MacOS
and Windows. To install, run:

```shell
pip install pytket-aqt
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Available devices

`pytket-aqt` offers offline simulators for aqt devices that do not require special access.
Access to remote simulators and machines can be configured by providing an AQT access token.

To see which devices are available to you, use the `AQTBackend.print_device_table` method.
This method will prompt for an access token if none has been configured. Providing a token at the
prompt will store it in memory for further API use. Skip the prompt to see the available offline
simulators. It is also possible to store and use your access token across sessions using
`config.set_aqt_config`.

## Ion Shuttling

The `AQTMultiZoneBackend` supports routing of a circuit to a particular segmented ion-trap architecture before submission. This feature is experimental and not necessary for any currently available AQT devices. Using the graph partitioning based algorithms within this context requires the [mt-kahypar](https://github.com/kahypar/mt-kahypar) package.

On Windows based systems, `mtkahypar` is not installable via `pip` and will not be included with `pytket-aqt` as a dependency. It must be compiled and installed manually (see the instructions provided in the mt-kahypar repository for a manual
installation).

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-aqt/issues).

There is also a Slack channel for discussion and support. Click [here](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#/shared-invite/email) to join.

## Development

This project uses [Poetry](https://python-poetry.org/) for packaging and dependency management and
[Nox](https://nox.thea.codes/en/stable/) for task automation.

### Recommended development setup

Install development tools:

```shell
pip install -r dev-tool-requirements.txt
```

### Local development with Nox (recommended)

[Nox](https://nox.thea.codes/en/stable/) can be used to automate various development tasks running in isolated python environments.
The following Nox sessions are provided:

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

This will install the dependencies within an isolated virtual environment managed by Poetry. To activate that environment run:

```shell
poetry shell
```

Within this environment, the following commands can be used:

```shell
# run tests
pytest tests
# run mypy
mypy --explicit-package-bases pytket tests docs/conf.py docs/build-docs
# run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
# build documentation
./docs/build-docs
```

To exit the Poetry environment, run:

```shell
exit
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting and Linting

All code will be checked on the CI with [black](https://black.readthedocs.io/en/stable/) and [pylint](https://pypi.org/project/pylint/)
as configured within the `pre-commit` checks. These checks should be
run locally before any pull request submission using the corresponding `nox` session or `pre-commit` directly (see above).
The used versions of the formatting ad linting tools is specified in the [pyproject.toml](pyproject.toml).

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. This should be done using the method
described under Local development [with Nox](#local-development-with-nox-recommended) or [without Nox](#local-development-without-nox).

### Adding Tests

When adding a new feature, please add appropriate tests for it within the [tests](tests) directory. When fixing a bug, please
add a test that demonstrates the fix.

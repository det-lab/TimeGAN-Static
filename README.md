# TimeGAN

[![Release](https://img.shields.io/github/v/release/det-lab/TimeGAN-Static)](https://img.shields.io/github/v/release/det-lab/TimeGAN-Static)
[![Build status](https://img.shields.io/github/actions/workflow/status/det-lab/TimeGAN-Static/main.yml?branch=main)](https://github.com/det-lab/TimeGAN-Static/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/det-lab/TimeGAN-Static/branch/main/graph/badge.svg)](https://codecov.io/gh/det-lab/TimeGAN-Static)
[![Commit activity](https://img.shields.io/github/commit-activity/m/det-lab/TimeGAN-Static)](https://img.shields.io/github/commit-activity/m/det-lab/TimeGAN-Static)
[![License](https://img.shields.io/github/license/det-lab/timegan-static)](https://img.shields.io/github/license/det-lab/timegan-static)

A fork of https://github.com/jsyoon0823/TimeGAN that implements static features and snapshotting

- **Original Github repository**: <https://github.com/det-lab/TimeGAN-Static/>
- **Documentation** <https://det-lab.github.io/TimeGAN-Static/>

## Installing this software

You can install this software into a Python 3.9 - 3.10 environment with

```bash
pip install timegan
```
## Creating a singularity container
Creeating a singularity container is currently supported with apptainer

```bash
apptainer build 'envname'.sif timeGANenv.def
```

And can be tested with 
```bash
apptainer shell 'envname'.sif
pip list
```

## Developing this code

To finalize the set-up for publishing to PyPi or Artifactory, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/det-lab/TimeGAN-Static/settings/secrets/actions/new).
- Create a [new release](https://github.com/det-lab/TimeGAN-Static/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

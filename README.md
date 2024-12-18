# TimeGAN

[![Release](https://img.shields.io/github/v/release/det-lab/TimeGAN-Static)](https://img.shields.io/github/v/release/det-lab/TimeGAN-Static)
[![Build status](https://img.shields.io/github/actions/workflow/status/det-lab/TimeGAN-Static/main.yml?branch=main)](https://github.com/det-lab/TimeGAN-Static/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/det-lab/TimeGAN-Static/branch/main/graph/badge.svg)](https://codecov.io/gh/det-lab/TimeGAN-Static)
[![Commit activity](https://img.shields.io/github/commit-activity/m/det-lab/TimeGAN-Static)](https://img.shields.io/github/commit-activity/m/det-lab/TimeGAN-Static)
[![License](https://img.shields.io/github/license/det-lab/timegan-static)](https://img.shields.io/github/license/det-lab/timegan-static)

A fork of https://github.com/jsyoon0823/TimeGAN that implements static features and snapshotting

- **Original Github repository**: <https://github.com/det-lab/TimeGAN-Static/>
- **Documentation** <https://det-lab.github.io/TimeGAN-Static/>

## Installing this Software

This package is available for install via pip: [timegan · PyPI](https://pypi.org/project/timegan/).
You will need a Python 3.9 - 3.10 environment to properly match versions with certain dependencies.

```bash
pip install timegan
```

## Creating a Singularity Container

The timegan package is also equipped with a definition file that allows you to build a timegan training container with root.

After cloning the Repository, run the following command within the directory:

```bash
apptainer build 'envname'.sif env.def
```

You can test for a proper build to check the timegan version install. It should match the latest version on Github.

```bash
apptainer shell 'envname'.sif
pip list
```

This will also work if you are using singularity to build.

## (For Developers)

When uploading any updates, you will need to resolve any build issues before pushing your changes.

If you need to make any changes, you will need to do a but of set up.

First install [Poetry](https://python-poetry.org/docs/), I recommend doing so with pipx. Poetry will allow you to track and manage the dependencies, and git workflow:

```bash
sudo apt update -y
sudo apt install pipx
pipx install poetry

pipx ensurepath
```

You will need to set pipx locations to your PATH environment and restart your terminal session.
For Linux:

You will need to install pre-commit as well. You cant do this as an apt install, or via pip within a virtual environment. If you have issues with accessing the git Hooks, see: [How to Set Up Pre-Commit Hooks | Stefanie Molin](https://stefaniemolin.com/articles/devx/pre-commit/setup-guide/)

```bash
apt install pre-commit
pre-commit install
```

Before pushing any updates, run the command:

```bash
make check
```

to resolve workflow errors. Then re-commit those changes and push:

```bash
git commit
git push
```

Whenever making changes to the package's code, be sure to update the version number in the pyproject.toml file.

You can upload new versions to pypi via poetry, or through Github. I recommend the latter, but using Poetry gives you a little more control.

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

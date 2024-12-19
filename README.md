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

You will need a Python 3.9 - 3.11 environment to properly match versions with certain dependencies.

```bash
pip install timegan
```

## Creating a Singularity Container

The timegan package is also equipped with a definition file that allows you to build a timegan training container with root.

After cloning the Repository, run the following command within the directory:

```bash
apptainer build 'envname'.sif env.def
```

You can test the build by checking timegan version install within the container. It should match the latest version on Github.

```bash
apptainer shell 'envname'.sif
pip list
```

This will also work if you are using singularity to build.

## (For Developers)

Maintaining this repository will require a bit of setup.

- [Poetry](https://python-poetry.org/docs/)
  - [Pre-commit](https://pre-commit.com/)
- [Pyenv Virtual environment | K0nze](https://k0nze.dev/posts/install-pyenv-venv-vscode/)
  - You can also use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for this
- Python 3.9-3.10
- Additional... (perhaps)

<h5> Poetry </h5>
This repository was built using Poetry, its recommended for all of the management for this repository. You will need also this to commit changes to the repository.

- Functions like "poetry show" will list all the dependencies for running timegan training.

- More details about the use of Poetry for this repository can be found here (perhaps).

<h5>Pyenv</h5>
The main reason for using pyenv here is for setting up virtual environments.
Using a virtual environment is ideal for testing and other tasks. I recommending checking the install tutorial by K0nze linked above.

- Pyenv also allows for choosing which version of python you would like to use.

- If preferred, you can accomplish the same things using Conda

<h3>Commit Changes (Linux) </h3>

First install [Poetry](https://python-poetry.org/docs/), you can find more details for the install from this link.

```bash
sudo apt update -y
sudo apt install pipx
pipx install poetry

pipx ensurepath
```

You will need to set pipx locations to your PATH environment and restart your terminal session.

Committing changes without running a pre-check will result in build issues. To run a check, you will need pre-commit installed and able to access the Github hooks.

You cant do this as an apt install, or (recommended) via pip within a virtual environment. If you have issues with accessing the git Hooks, see: [How to Set Up Pre-Commit Hooks | Stefanie Molin](https://stefaniemolin.com/articles/devx/pre-commit/setup-guide/)

```bash
apt install pre-commit
pre-commit install
```

Or install through a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pre-commit
```

From here, the hooks are built into the repository and can be set up by running:

```bash
pre-commit install
```

<h3>Pushing Commits</h3>
start by staging files to be committed.

Before pushing, run the command to resolve workflow errors:

```bash
git status
git add 'filename'
make check
```

If changes were made to your files, they will drop back down to the un-staged section when running "git status".

Then re-stage those files and push a commit:

```bash
git status # If you'd like to double check
git add 'filename'
make check # If you'd like to double check
git status # If you'd like to double check
git commit
git push
```

Whenever making changes to the package's code, be sure to update the version number in the pyproject.toml file.

You can upload new releases to Pypi via [Poetry](https://python-poetry.org/docs/cli/#publish), or through [Github](https://github.com/det-lab/TimeGAN-Static/releases). With poetry, you will need to configure your credentials, so I recommend drafting new releases through Github for simplicity.

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

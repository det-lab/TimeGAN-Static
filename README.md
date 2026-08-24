# TimeGAN-Static

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

You will need a Python 3.10 environment to properly match versions with certain dependencies.

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

## Contributing

This repository is managed with [Poetry](https://python-poetry.org/docs/). See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up a development environment, running tests, and submitting a pull request.

### Releasing a New Version to PyPI

New releases are published to PyPI automatically by a GitHub Actions workflow that runs when a release is published on GitHub. To cut a new release:

1. Update the version number in `pyproject.toml` to match the release you're about to cut (the CI workflow also sets it from the tag when publishing, but keeping it in sync locally avoids confusion).

2. Tag the commit you want to release, using the version number as the tag name (e.g. `0.1.9`):

```bash
git tag 0.1.9
git push origin 0.1.9
```

3. On GitHub, [draft a new release](https://github.com/det-lab/TimeGAN-Static/releases) using the tag you just pushed, then publish it.

Publishing the release triggers the `release-main` workflow, which sets the package version to the tag name, builds the package, and publishes it to PyPI automatically.

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).

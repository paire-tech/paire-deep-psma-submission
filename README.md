<h3 align="center">PaIRe - DEEP PSMA Challenge Submission</h3>

[![python](https://img.shields.io/badge/python-3.11_%7C_3.12-red.svg?color=090422&labelColor=1E14B0&logo=python&logoColor=white)](https://www.python.org/)
[![docker](https://img.shields.io/badge/docker-build-red?color=090422&labelColor=1E14B0&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytorch](https://img.shields.io/badge/pytorch-2.0+-red.svg?color=090422&labelColor=1E14B0&logo=pytorch&logoColor=white)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/lightning-2.0+-792ee5?color=090422&labelColor=1E14B0&logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![mypy](https://img.shields.io/badge/mypy-typing-red.svg?color=090422&labelColor=1E14B0&logo=python&logoColor=white)](https://mypy-lang.org)
[![ruff](https://img.shields.io/badge/ruff-linter-red.svg?color=090422&labelColor=1E14B0&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff)

</div>

<p align="center">
  <i>
    Code and submission for the DeepPSMA Grand Challenge 🚀⚡🔥
  </i>
</p>

<br>

## 📌 About <a name="about" />

This repository contains the code for our submission to the **DEEP-PSMA** (Deep-learning Evaluation for Enhanced Prognostics - Prostate Specific Membrane Antigen) Grand Challenge, which focuses on the detection of lesions on PSMA PET/CT and FDG PET/CT.

The challenge is hosted by [Grand Challenge](https://deep-psma.grand-challenge.org/). You can find more documentation and details on [how to test and deploy your container](https://grand-challenge.org/documentation/test-and-deploy-your-container/), specifically [how to add the algorithm](https://grand-challenge.org/documentation/add-the-algorithm/) and [how to link a github repository to the algorithm](https://grand-challenge.org/documentation/linking-a-github-repository-to-your-algorithm/).

> [!IMPORTANT]
> This repository is linked to the DEEP PSMA Grand Challenge submission page to automatically update the algorithm when a new version is tagged. <br> **Do not add any sensitive information to this repository.**

<br>

## 🏁 Getting Started <a name="getting-started" />

The following section will guide you on how to set up the project locally, build and run the Docker container, and submit your algorithm to the DEEP PSMA Grand Challenge.

### Installation

To install the project locally, you can use the following command:

```bash
uv sync --dev
```

> [!NOTE]
> You can still install the package using `pip` but we recommend using `uv` instead for managing your environment. For more details on how to use `uv`, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/).

### Build and Run

To build the Docker container, you can use the following command:

```bash
make build
```

To run it, you can use:

```bash
make run
```

> [!NOTE]
> The model weights are not included in this repository. They should be made available at `/opt/ml/model/` inside the container.
> Contact us to obtain the model weights.

> [!TIP]
> Weights provided in the `weights/` directory will be mounted to `/opt/ml/model/` inside the container by default by the `make run` command.

### Submit Your Algorithm

This repository is linked to the DEEP PSMA Grand Challenge page. To submit a new algorithm, you should:

1. Add your changes / features to this repository through a pull request
2. Verify the tests are passing and the build is successful
3. Once the pull request is merged, create a new tag with the version number (e.g., `v0.1.0`)
4. The algorithm will be automatically updated on the Grand Challenge platform
5. Add the model weights in the DEEP PSMA Grand Challenge submission page.

> [!NOTE]
> For more information refer to the documentation on [how to link a github repository to the algorithm](https://grand-challenge.org/documentation/linking-a-github-repository-to-your-algorithm/).

<br>

## 🧑‍💻 For Developers <a name="for-developers" />

### Testing

### Make Commands

<br>

## ❓ FAQ <a name="faq" />

<details>
<summary><b>How to configure VSCode?</b></summary>

Add the following at the root of the repo, in `.vscode/settings.json`

```json
{
  "[markdown]": {
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true
  },
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "[toml]": {
    "editor.defaultFormatter": "tamasfe.even-better-toml"
  },
  "autoDocstring.docstringFormat": "google-notypes",
  "autoDocstring.guessTypes": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.rulers": [120],
  "markdownlint.config": {
    "MD013": false,
    "MD028": false,
    "MD033": false,
    "MD041": false,
    "MD046": false,
    "default": true
  },
  "mypy.configFile": "./pyproject.toml",
  "mypy.dmypyExecutable": "${config:python.pythonPath}/bin/dmypy",
  "mypy.runUsingActiveInterpreter": true,
  "python.testing.pytestArgs": ["tests"],
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "ruff.configuration": "./pyproject.toml"
}
```

</details>

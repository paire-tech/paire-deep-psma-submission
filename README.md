<h3 align="center">PaIRe DEEP-PSMA Submission</h3>

<div align="center">

[![python](https://img.shields.io/badge/python-3.11_%7C_3.12-red.svg?color=090422&labelColor=1E14B0&logo=python&logoColor=white)](https://www.python.org/)
[![docker](https://img.shields.io/badge/docker-build-red?color=090422&labelColor=1E14B0&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytorch](https://img.shields.io/badge/pytorch-2.0+-red.svg?color=090422&labelColor=1E14B0&logo=pytorch&logoColor=white)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/lightning-2.0+-792ee5?color=090422&labelColor=1E14B0&logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![mypy](https://img.shields.io/badge/mypy-typing-red.svg?color=090422&labelColor=1E14B0&logo=python&logoColor=white)](https://mypy-lang.org)
[![ruff](https://img.shields.io/badge/ruff-linter-red.svg?color=090422&labelColor=1E14B0&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff)

</div>

<p align="center">
  <i>
    Inference code and submission for the DEEP-PSMA Grand Challenge 🚀⚡🔥
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

This repository can be used in two ways:

1. **Local Development**: You can run the code locally on your machine for development purposes. This is useful for testing and debugging your code before submitting it. See the [For Developers](#for-developers) section for more details.

2. **Docker Container**: You can build and run the code inside a Docker container for a more isolated and reproducible environment. This is the recommended way to ensure the code runs as expected for the DEEP PSMA Grand Challenge.

### Installation

To build the docker container that will be used for the DEEP PSMA Grand Challenge, you need to have Docker installed on your machine. You can follow the [official Docker installation guide](https://docs.docker.com/get-docker/) to install Docker.

1. **Build the Docker container:** Use the following command to build the Docker container:

   ```bash
   make build
   ```

2. **Download the model weights:** The model weights are not included in this repository. You need to obtain them and place them in the `weights/` directory. You can find the weights on S3.

### How To Use

Once you have the Docker container on your machine, you can run it using the following command:

```bash
make run
```

The docker container also supports additional arguments and options to customize the inference process. You can specify them through environment variables (create a `.env` file in the root directory):

| Argument          | Description                                                                             | Default          |
| ----------------- | --------------------------------------------------------------------------------------- | ---------------- |
| `INPUT_FORMAT`    | The format of the input data. Can be `gc` (Grand Challenge format) or `csv` (CSV file). | `gc`             |
| `INPUT_DIR`       | The directory containing the input data.                                                | `/input`         |
| `OUTPUT_DIR`      | The directory where the output data will be saved.                                      | `/output`        |
| `WEIGHTS_DIR`     | The directory containing the model weights.                                             | `/opt/ml/model/` |
| `DEVICE`          | The device to use for inference. Can be `auto`, `cpu` or `cuda`.                        | `cuda`           |
| `MIXED_PRECISION` | Flag to enable mixed precision inference.                                               | `false`          |

> [!NOTE]
> **Grand Challenge format**: This is the default format used by the DEEP PSMA Grand Challenge. <br>
> **CSV format**: This format expects a CSV file with the following columns: `{psma|fdg}_pt_path`, `{psma|fdg}_ct_path`, `{psma|fdg}_organ_segmentation_path`, and `{psma|fdg}_suv_threshold`. The paths should point to the respective files relative to the input directory.

### Evaluation

You can quickly evaluate the performance of the model from the saved predictions mask using the `scripts/evaluate.py` script. This script will compute the Dice score and other metrics for the predictions:

```bash
python scripts/evaluate.py \
    --input-dir /data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA/ \
    --output-dir data/output/ \
    --input-csv /data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA/challenge_data.csv \
    --output-csv data/output/results.csv
```

### Submit Your Algorithm

To submit a new algorithm to the DEEP-PSMA Grand Challenge, follow these steps:

1. Add your changes / features to this repository through a pull request. Follow the [For Developers](#for-developers) section for more details on how to contribute.
2. Once the pull request is merged, create a new tag with the version number (e.g., `vx.y.z`) to track our submissions from the code.
3. Build the Docker container using

    ```bash
    make build
    ```

4. Export the Docker image to a tar file:

    ```bash
    make export
    ```

5. Add the container and model weights in the DEEP PSMA Grand Challenge submission page. For more details, refer to the [test and deploy your container](https://grand-challenge.org/documentation/test-and-deploy-your-container/) documentation.

<br>

## 🧑‍💻 For Developers <a name="for-developers" />

### Local Installation

To install the project locally, you can use the following command:

```bash
uv sync --dev
```

> [!NOTE]
> You can still install the package using `pip` but we recommend using `uv` instead for managing your environment. For more details on how to use `uv`, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/).

### Usage

This package comes with a CLI used as the main entrypoint for the inference (and from the Docker container). You can run the CLI using the following command:

```bash
paire-deep-psma-submission --help
```

The available options are:

| Option              | Description                                                                             | Default          |
| ------------------- | --------------------------------------------------------------------------------------- | ---------------- |
| `--input-format`    | The format of the input data. Can be `gc` (Grand Challenge format) or `csv` (CSV file). | `gc`             |
| `--input-dir`       | The directory containing the input data.                                                | `/input`         |
| `--output-dir`      | The directory where the output data will be saved.                                      | `/output`        |
| `--weights-dir`     | The directory containing the model weights.                                             | `/opt/ml/model/` |
| `--device`          | The device to use for inference. Can be `auto`, `cpu` or `cuda`.                        | `cuda`           |
| `--mixed-precision` | Flag to enable mixed precision inference.                                               | `false`          |

> [!NOTE]
> These options can also be set through environment variables.

### Testing

Before submitting your changes, make sure to run the tests to ensure everything is working as expected. You can run the tests using the following command:

```bash
make test
```

### Make Commands

We used `make` to define common commands for building, testing, and running the project. Here are the available commands:

| Command         | Description                                       |
| --------------- | ------------------------------------------------- |
| `make format`   | Format source code using Ruff and Black.          |
| `make lint`     | Lint source code using Ruff.                      |
| `make lint-fix` | Lint and fix source code using Ruff.              |
| `make isort`    | Sort imports using Ruff.                          |
| `make type`     | Check types in source code using Mypy.            |
| `make test`     | Run unit tests using Pytest.                      |
| `make build`    | Build a production Docker image.                  |
| `make run`      | Run the production Docker image.                  |
| `make export`   | Export the production Docker image to a tar file. |
| `make all`      | Run all formatting commands.                      |
| `make clean`    | Clear local caches and build artifacts.           |
| `make help`     | Show available commands.                          |

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

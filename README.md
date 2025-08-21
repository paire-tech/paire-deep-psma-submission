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

<br>

## ⚡ Quick Access <a name="quick-links" />

<details>
<summary><b>Default <code>splits_final.json</code> used</b></summary>

```json
[
  {
    "train": [
      "train_0001",
      "train_0002",
      "train_0003",
      "train_0004",
      "train_0006",
      "train_0007",
      "train_0008",
      "train_0009",
      "train_0011",
      "train_0012",
      "train_0013",
      "train_0015",
      "train_0016",
      "train_0017",
      "train_0018",
      "train_0019",
      "train_0020",
      "train_0022",
      "train_0023",
      "train_0024",
      "train_0025",
      "train_0027",
      "train_0028",
      "train_0030",
      "train_0031",
      "train_0032",
      "train_0033",
      "train_0035",
      "train_0037",
      "train_0038",
      "train_0039",
      "train_0040",
      "train_0042",
      "train_0044",
      "train_0045",
      "train_0046",
      "train_0047",
      "train_0048",
      "train_0049",
      "train_0051",
      "train_0053",
      "train_0054",
      "train_0055",
      "train_0056",
      "train_0057",
      "train_0058",
      "train_0059",
      "train_0060",
      "train_0062",
      "train_0063",
      "train_0065",
      "train_0067",
      "train_0068",
      "train_0069",
      "train_0071",
      "train_0072",
      "train_0073",
      "train_0074",
      "train_0075",
      "train_0077",
      "train_0078",
      "train_0080",
      "train_0081",
      "train_0082",
      "train_0083",
      "train_0084",
      "train_0085",
      "train_0086",
      "train_0088",
      "train_0090",
      "train_0091",
      "train_0092",
      "train_0093",
      "train_0094",
      "train_0095",
      "train_0096",
      "train_0097",
      "train_0098",
      "train_0099",
      "train_0100"
    ],
    "val": [
      "train_0005",
      "train_0010",
      "train_0014",
      "train_0021",
      "train_0026",
      "train_0029",
      "train_0034",
      "train_0036",
      "train_0041",
      "train_0043",
      "train_0050",
      "train_0052",
      "train_0061",
      "train_0064",
      "train_0066",
      "train_0070",
      "train_0076",
      "train_0079",
      "train_0087",
      "train_0089"
    ]
  },
  {
    "train": [
      "train_0002",
      "train_0004",
      "train_0005",
      "train_0006",
      "train_0008",
      "train_0010",
      "train_0011",
      "train_0012",
      "train_0013",
      "train_0014",
      "train_0015",
      "train_0016",
      "train_0017",
      "train_0019",
      "train_0020",
      "train_0021",
      "train_0022",
      "train_0023",
      "train_0024",
      "train_0025",
      "train_0026",
      "train_0028",
      "train_0029",
      "train_0030",
      "train_0032",
      "train_0033",
      "train_0034",
      "train_0035",
      "train_0036",
      "train_0037",
      "train_0039",
      "train_0041",
      "train_0042",
      "train_0043",
      "train_0044",
      "train_0045",
      "train_0049",
      "train_0050",
      "train_0052",
      "train_0054",
      "train_0056",
      "train_0057",
      "train_0059",
      "train_0060",
      "train_0061",
      "train_0062",
      "train_0063",
      "train_0064",
      "train_0065",
      "train_0066",
      "train_0067",
      "train_0068",
      "train_0069",
      "train_0070",
      "train_0071",
      "train_0073",
      "train_0074",
      "train_0075",
      "train_0076",
      "train_0078",
      "train_0079",
      "train_0080",
      "train_0081",
      "train_0082",
      "train_0083",
      "train_0084",
      "train_0085",
      "train_0086",
      "train_0087",
      "train_0089",
      "train_0090",
      "train_0091",
      "train_0092",
      "train_0094",
      "train_0095",
      "train_0096",
      "train_0097",
      "train_0098",
      "train_0099",
      "train_0100"
    ],
    "val": [
      "train_0001",
      "train_0003",
      "train_0007",
      "train_0009",
      "train_0018",
      "train_0027",
      "train_0031",
      "train_0038",
      "train_0040",
      "train_0046",
      "train_0047",
      "train_0048",
      "train_0051",
      "train_0053",
      "train_0055",
      "train_0058",
      "train_0072",
      "train_0077",
      "train_0088",
      "train_0093"
    ]
  },
  {
    "train": [
      "train_0001",
      "train_0002",
      "train_0003",
      "train_0004",
      "train_0005",
      "train_0006",
      "train_0007",
      "train_0008",
      "train_0009",
      "train_0010",
      "train_0011",
      "train_0012",
      "train_0014",
      "train_0015",
      "train_0016",
      "train_0018",
      "train_0021",
      "train_0024",
      "train_0026",
      "train_0027",
      "train_0029",
      "train_0030",
      "train_0031",
      "train_0032",
      "train_0034",
      "train_0035",
      "train_0036",
      "train_0037",
      "train_0038",
      "train_0039",
      "train_0040",
      "train_0041",
      "train_0042",
      "train_0043",
      "train_0044",
      "train_0045",
      "train_0046",
      "train_0047",
      "train_0048",
      "train_0050",
      "train_0051",
      "train_0052",
      "train_0053",
      "train_0054",
      "train_0055",
      "train_0056",
      "train_0057",
      "train_0058",
      "train_0060",
      "train_0061",
      "train_0062",
      "train_0064",
      "train_0065",
      "train_0066",
      "train_0067",
      "train_0068",
      "train_0070",
      "train_0071",
      "train_0072",
      "train_0073",
      "train_0074",
      "train_0076",
      "train_0077",
      "train_0078",
      "train_0079",
      "train_0081",
      "train_0082",
      "train_0083",
      "train_0087",
      "train_0088",
      "train_0089",
      "train_0091",
      "train_0092",
      "train_0093",
      "train_0094",
      "train_0095",
      "train_0097",
      "train_0098",
      "train_0099",
      "train_0100"
    ],
    "val": [
      "train_0013",
      "train_0017",
      "train_0019",
      "train_0020",
      "train_0022",
      "train_0023",
      "train_0025",
      "train_0028",
      "train_0033",
      "train_0049",
      "train_0059",
      "train_0063",
      "train_0069",
      "train_0075",
      "train_0080",
      "train_0084",
      "train_0085",
      "train_0086",
      "train_0090",
      "train_0096"
    ]
  },
  {
    "train": [
      "train_0001",
      "train_0002",
      "train_0003",
      "train_0005",
      "train_0007",
      "train_0008",
      "train_0009",
      "train_0010",
      "train_0011",
      "train_0012",
      "train_0013",
      "train_0014",
      "train_0015",
      "train_0017",
      "train_0018",
      "train_0019",
      "train_0020",
      "train_0021",
      "train_0022",
      "train_0023",
      "train_0025",
      "train_0026",
      "train_0027",
      "train_0028",
      "train_0029",
      "train_0030",
      "train_0031",
      "train_0033",
      "train_0034",
      "train_0035",
      "train_0036",
      "train_0037",
      "train_0038",
      "train_0039",
      "train_0040",
      "train_0041",
      "train_0042",
      "train_0043",
      "train_0044",
      "train_0046",
      "train_0047",
      "train_0048",
      "train_0049",
      "train_0050",
      "train_0051",
      "train_0052",
      "train_0053",
      "train_0055",
      "train_0058",
      "train_0059",
      "train_0060",
      "train_0061",
      "train_0063",
      "train_0064",
      "train_0066",
      "train_0069",
      "train_0070",
      "train_0072",
      "train_0074",
      "train_0075",
      "train_0076",
      "train_0077",
      "train_0078",
      "train_0079",
      "train_0080",
      "train_0081",
      "train_0082",
      "train_0083",
      "train_0084",
      "train_0085",
      "train_0086",
      "train_0087",
      "train_0088",
      "train_0089",
      "train_0090",
      "train_0093",
      "train_0096",
      "train_0098",
      "train_0099",
      "train_0100"
    ],
    "val": [
      "train_0004",
      "train_0006",
      "train_0016",
      "train_0024",
      "train_0032",
      "train_0045",
      "train_0054",
      "train_0056",
      "train_0057",
      "train_0062",
      "train_0065",
      "train_0067",
      "train_0068",
      "train_0071",
      "train_0073",
      "train_0091",
      "train_0092",
      "train_0094",
      "train_0095",
      "train_0097"
    ]
  },
  {
    "train": [
      "train_0001",
      "train_0003",
      "train_0004",
      "train_0005",
      "train_0006",
      "train_0007",
      "train_0009",
      "train_0010",
      "train_0013",
      "train_0014",
      "train_0016",
      "train_0017",
      "train_0018",
      "train_0019",
      "train_0020",
      "train_0021",
      "train_0022",
      "train_0023",
      "train_0024",
      "train_0025",
      "train_0026",
      "train_0027",
      "train_0028",
      "train_0029",
      "train_0031",
      "train_0032",
      "train_0033",
      "train_0034",
      "train_0036",
      "train_0038",
      "train_0040",
      "train_0041",
      "train_0043",
      "train_0045",
      "train_0046",
      "train_0047",
      "train_0048",
      "train_0049",
      "train_0050",
      "train_0051",
      "train_0052",
      "train_0053",
      "train_0054",
      "train_0055",
      "train_0056",
      "train_0057",
      "train_0058",
      "train_0059",
      "train_0061",
      "train_0062",
      "train_0063",
      "train_0064",
      "train_0065",
      "train_0066",
      "train_0067",
      "train_0068",
      "train_0069",
      "train_0070",
      "train_0071",
      "train_0072",
      "train_0073",
      "train_0075",
      "train_0076",
      "train_0077",
      "train_0079",
      "train_0080",
      "train_0084",
      "train_0085",
      "train_0086",
      "train_0087",
      "train_0088",
      "train_0089",
      "train_0090",
      "train_0091",
      "train_0092",
      "train_0093",
      "train_0094",
      "train_0095",
      "train_0096",
      "train_0097"
    ],
    "val": [
      "train_0002",
      "train_0008",
      "train_0011",
      "train_0012",
      "train_0015",
      "train_0030",
      "train_0035",
      "train_0037",
      "train_0039",
      "train_0042",
      "train_0044",
      "train_0060",
      "train_0074",
      "train_0078",
      "train_0081",
      "train_0082",
      "train_0083",
      "train_0098",
      "train_0099",
      "train_0100"
    ]
  }
]
```

</details>

<details>
<summary><b>Export the cross-validation splits to CSV</b></summary>

For evaluation and inference purposes, we use CSV variants of the cross-validation splits (which are originally in JSON format). This is because it is more convenient to use CSV files to add/remove cases, and simply collaborate with others.

You **MUST** have a copy of the original `splits_final.json` file locally. Either copy it from above, or run the `scripts/02_train.py` to generate it automatically.

```bash
python scripts/00_export_splits.py \
  --splits-path /path/to/splits_final.json \
  --output-dir ./data/splits/
```

</details>

<details>
<summary><b>Preprocess the Challenge Data</b></summary>

We provide several scripts to preprocess the challenge data and prepare it for training with nnUNet. You can find the scripts in the <code>scripts/</code> directory:

- `scripts/01a_preprocess_baseline.py`: Preprocess the data for the baseline model.
- `scripts/01b_preprocess_with_organs.py`: Preprocess the data with organ segmentation from TotalSegmentator.

To use these scripts, run:

```bash
python scripts/01a_preprocess_baseline.py           # Use the desired preprocessing script
  --data-dir /path/to/DEEP_PSMA_CHALLENGE_DATA/ \   # Directory of the raw challenge data
  --tracer-name PSMA \                              # Name of the tracer to use
  --dataset-id 801 \                                # Identifier used by nnUNet
  --yes                                             # Skip confirmation prompts
```

> Make sure to set the environment variables `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to the appropriate directories before running the script. You can place them in a `.env` file in the root directory of the project.

</details>

<details>
<summary><b>Training with nnUNet</b></summary>

To train a model on the preprocessed data, run:

```bash
python scripts/train.py \     # Run nnUNet training
  --dataset-id 801 \          # Identifier of the previously preprocessed dataset
  --fold 0 \                  # Fold to train (depends on the `splits_final.json` file, usually between 0 and 4)
  --device cuda               # Device to use for training (can be `cuda`, `cpu`, or `mps`)
```

> Make sure to set the environment variables `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to the appropriate directories before running the script. You can place them in a `.env` file in the root directory of the project.

</details>

<details>
<summary><b>Inference with nnUNet</b></summary>

You can run the inference of the trained model(s) using the `scripts/03X_inference_XXX.py` scripts, which will save the predicted segmentation masks to the desired location. This is specifically useful to try and test the impact of postprocessing techniques. For example, to run inference on the baseline model, use:

```bash
python scripts/03a_inference_baseline.py \            # Run inference on the baseline model
  --input-dir /path/to/DEEP_PSMA_CHALLENGE_DATA/ \    # Directory of the challenge data
  --input-csv ./data/val_fold0.csv \                  # CSV containing specific cases to run inference on
  --output-dir ./data/output/ \                       # Directory to save the predictions
  --tracer-name PSMA \                                # Name of the tracer to use (can be `PSMA` or `FDG`)
  --dataset-id 801 \                                  # Identifier of the dataset to use
  --fold 0                                            # Fold to use for inference (depends on the `splits_final.json` file, usually between 0 and 4)
```

> Make sure to set the environment variables `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to the appropriate directories before running the script. You can place them in a `.env` file in the root directory of the project.

</details>

<details>
<summary><b>Evaluation and Benchmarking</b></summary>

You can run the evaluation script to compute the metrics used in the official DEEP PSMA Grand Challenge. This script will compute the Dice score and other metrics for each (already computed) prediction:

```bash
python scripts/04_evaluate.py \                       # Evaluate the performances of the model on precomputed predictions
  --input-dir /path/to/DEEP_PSMA_CHALLENGE_DATA/ \    # Directory of the challenge data
  --input-csv ./data/val_fold0.csv \                  # CSV containing specific cases to run inference on
  --output-dir ./data/output/ \                       # Directory to save the predictions
  --output-csv ./data/output/results.csv \            # CSV to save the results
```

</details>

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

2. **Download the model weights:** The model weights are not included in this repository.

### How To Use

Once you have the Docker container on your machine, you can run it using the following command:

```bash
make run
```

The docker container also supports additional arguments and options to customize the inference process. You can specify them through environment variables (create a `.env` file in the root directory):

| Argument       | Description                                                                             | Default   |
| -------------- | --------------------------------------------------------------------------------------- | --------- |
| `INPUT_FORMAT` | The format of the input data. Can be `gc` (Grand Challenge format) or `csv` (CSV file). | `gc`      |
| `INPUT_DIR`    | The directory containing the input data.                                                | `/input`  |
| `OUTPUT_DIR`   | The directory where the output data will be saved.                                      | `/output` |
| `DEVICE`       | The device to use for inference. Can be `auto`, `cpu` or `cuda`.                        | `cuda`    |

> [!NOTE]
> **Grand Challenge format**: This is the default format used by the DEEP PSMA Grand Challenge. <br> **CSV format**: This format expects a CSV file with the following columns: `{psma|fdg}_pt_path`, `{psma|fdg}_ct_path`, `{psma|fdg}_organ_segmentation_path`, and `{psma|fdg}_suv_threshold`. The paths should point to the respective files relative to the input directory.

### Evaluation

You can quickly evaluate the performance of the model from the saved predictions mask using the `scripts/04_evaluate.py` script. This script will compute the Dice score and other metrics for the predictions:

```bash
python scripts/04_evaluate.py \                       # Evaluate the performances of the model on precomputed predictions
  --input-dir /path/to/DEEP_PSMA_CHALLENGE_DATA/ \    # Directory of the challenge data
  --input-csv ./data/val_fold0.csv \                  # CSV containing specific cases to run inference on
  --output-dir ./data/output/ \                       # Directory to save the predictions
  --output-csv ./data/output/results.csv \            # CSV to save the results
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

| Option           | Description                                                                             | Default   |
| ---------------- | --------------------------------------------------------------------------------------- | --------- |
| `--input-format` | The format of the input data. Can be `gc` (Grand Challenge format) or `csv` (CSV file). | `gc`      |
| `--input-dir`    | The directory containing the input data.                                                | `/input`  |
| `--output-dir`   | The directory where the output data will be saved.                                      | `/output` |
| `--device`       | The device to use for inference. Can be `auto`, `cpu` or `cuda`.                        | `cuda`    |

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
<summary><b>How to configure my VSCode project?</b></summary>

If you are using VSCode and want to configure your IDE to use our recommended settings,
copy-paste the below in your `.vscode/settings.json` file.

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

<details>
<summary><b>How to setup Latex and R Markdown?</b></summary>

We used Latex and R Markdown to write the report for the DEEP PSMA Grand Challenge.

1. Download R, R Studio
2. Download Pandoc
3. Download TextLive (full)

For example on Ubuntu, you can install the following packages:

```bash
# Install Latex
sudo apt-get update
sudo apt-get install texlive-full  # install latex and its packages
latex --version
# pdfTeX 3.141592653-2.6-1.40.22 (TeX Live 2022/dev/Debian)

# To setup R Markdown:
sudo apt-get update
sudo apt install r-base r-base-dev
R --version
# 4.1.2

# Install Pandoc
sudo apt-get install pandoc  # install pandoc for markdown conversion
# or: conda install -c conda-forge pandoc
pandoc --version
# 2.9.2.1
```

</details>

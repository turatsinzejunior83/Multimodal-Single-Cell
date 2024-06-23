# Multimodal-Single-Cell
Goal of the Competition The goal of this competition is to predict how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells. 


This repository contains code for the NeurIPS 2022 competition on multimodal single-cell integration across time, individuals, and batches.

## Directory Structure

- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `src/`: Python scripts for data preprocessing, model training, and evaluation.
- `requirements.txt`: List of Python dependencies.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/turatsinzejunior83/multimodal_single_cell_integration.git
    cd multimodal_single_cell_integration
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset files and place them in the `data/` directory.

## Usage

Run the Jupyter notebooks in the `notebooks/` directory for data preprocessing, model training, and evaluation.

Alternatively, you can run the Python scripts in the `src/` directory:

- Data Preprocessing:
    ```bash
    python src/data_preprocessing.py
    ```

- Baseline Models:
    ```bash
    python src/baseline_models.py
    ```

- Advanced Models:
    ```bash
    python src/advanced_models.py
    ```

- Evaluation:
    ```bash
    python src/evaluation.py
    ```

## License

This project is licensed under the GBU Public license.

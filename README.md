# Fraud-Label Constrained Dual-Task Graph Neural Networks for Bitcoin Anti-Money Laundering
[![DOI](https://zenodo.org/badge/1463429574.svg)](https://doi.org/10.5281/zenodo.24355989)

Code, results, and figures for the MPhil Data Science thesis of Gilbert Woasieworm Adjei, University of Ghana. Supervisor: Dr. Alvin Prince Ansah.

The thesis tests whether a fraud-label constraint improves a dual-task GraphSAGE model (node classification plus link prediction) for detecting illicit transactions in the Elliptic Bitcoin dataset. All experiments run on the complete graph (203,769 nodes, 234,355 edges, 49 time steps).

## Contents

| Path | What it holds |
|---|---|
| `notebook/elliptic-aml-dual-task-gnn.ipynb` | The single notebook that produces every number, figure, and table in Chapter 4. |
| `results/thesis_results.json` | All results from the run, including per-seed values, paired tests, audit, weight sweep and rolling folds. |
| `results/thesis_results_control.json` | Mechanism control experiment (Section 4.10). |
| `figures_and_tables/` | Figures (PNG) and tables (CSV) exported by the notebook. |
| `models/` | Saved weights for Options 1A and 1B on both splits (seed 42). |
| `requirements.txt` | Package versions of the recorded run. |

## Data

The Elliptic dataset is not redistributed here. Download it (Weber et al., 2019, published by Elliptic on Kaggle as "Elliptic Data Set") and place these three files in one folder:

- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

## Reproducing the run

1. Open the notebook in Google Colab and select a GPU runtime. The recorded run used a Tesla T4.
2. Put the three data files in `MyDrive/elliptic_bitcoin_dataset/` on Google Drive, or change `DATA_PATH` in the configuration cell.
3. Check that `DRY_RUN = False`. Setting it to `True` runs a short test with two seeds and 15 epochs.
4. Run all cells. The first cell installs `torch-geometric` and `scikit-learn` at the pinned versions.

The notebook writes the results JSON files, figures, tables, and model weights at the end.

## Recorded environment

Python 3.13.16, CUDA 13.0, Tesla T4 on Google Colab, torch 2.14.1+cu130, torch-geometric 2.6.1, scikit-learn 1.5.2, numpy 2.4.6, pandas 2.2.3, scipy 1.16.3. Deterministic algorithms were enabled. The run was exported on 2026-11-14.

## Reproducibility notes

- Training in single precision on a GPU is numerically sensitive. Results are exactly reproducible only on the same hardware and software. Two full runs of the notebook agreed to four decimals, with two exceptions. The audit variants "flaws 1 and 2 only" and "all five except flaw 1" did not train steadily, so their size differed between runs and only their direction agreed. The logistic regression values moved in the third decimal.
- The numbers in Chapter 4 come from the second full run.
- The data split is fixed. The eight seeds (42, 1042, 2042, 3042, 4042, 5042, 6042, 7042) change initialisation and dropout only.
- The attention-encoder (GAT) check did not train, so it is inconclusive and supports no claim.

## Main results (temporal split, test steps 35 to 49)

| Model | F1 at validated threshold | Average precision | Link AUC |
|---|---|---|---|
| Node-only | 0.5246 | 0.4564 | none |
| Option 1A | 0.5341 | 0.5305 | 0.6966 |
| Option 1B | 0.6082 | 0.6481 | 0.6123 |
| Random forest | 0.8008 | 0.7861 | none |

Option 1B beats Option 1A in all eight seeds on all three performance metrics, at a cost of 0.084 in link AUC. The gain is concentrated before step 43, and under the random split, the constraint does not help. A random forest on the same features beats every graph model. Full details are in Chapter 4 of the thesis and in `results/`.

## Citation

Adjei, G. W. (2026). Fraud-label constrained dual-task graph neural networks for Bitcoin anti-money laundering: code and results (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.24356225

## References

- Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., and Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv:1908.02591.

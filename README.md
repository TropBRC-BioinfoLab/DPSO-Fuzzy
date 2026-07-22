# DPSO-Fuzzy

Python implementation of **DPSO-Fuzzy**, a community-detection framework that combines Discrete Particle Swarm Optimization (DPSO) with fuzzy membership refinement to identify significant and overlapping disease communities in ontology-based lung cancer comorbidity networks.

The repository includes:

* benchmark evaluation on Karate, Dolphins, Football, Yeast-D2, and Y2H networks;
* comparison with several overlapping and fuzzy community-detection methods;
* application to NSCLC and SCLC comorbidity networks;
* preprocessing scripts for disease extraction and ontology-based network construction;
* stable significant-comorbidity identification across repeated runs;
* overlapping-node sensitivity analysis and visualization.

## Associated Study

This repository supports the study:

> **A DPSO-Fuzzy Framework for Detecting Significant and Overlapping Disease Communities in Ontology-Based Lung Cancer Comorbidity Networks**

Full bibliographic information will be added after publication.

## Method Overview

DPSO-Fuzzy consists of two main stages.

### 1. DPSO Community Initialization

DPSO represents a candidate solution as an integer community label for every node. The swarm searches for a high-quality hard partition by:

1. initializing particles from local graph neighborhoods;
2. evaluating particles using weighted modularity;
3. updating particle velocities from inertia, personal-best, and global-best information;
4. updating discrete community labels through a sigmoid-based rule;
5. splitting communities that contain disconnected components;
6. retaining the best hard community structure found by the swarm.

### 2. Fuzzy Community Refinement

The hard DPSO partition is converted into fuzzy memberships using an MSEFCD-inspired refinement procedure. Membership values are initialized from neighboring community labels and the DPSO prior, then iteratively updated through:

* membership smoothing;
* DPSO-guided membership enhancement;
* candidate-community selection based on node and community similarity;
* row-wise membership normalization.

A node is classified as overlapping when its membership reaches the selected alpha threshold in at least two communities.

## Repository Structure

```text
DPSO-Fuzzy/
├── Algorithms/
│   ├── benchmark_methods/
│   │   ├── tune_pso_lpa.py
│   │   ├── tune_othermethods.py
│   │   ├── evaluate_allmethods.py
│   │   ├── benchmark_common.py
│   │   ├── requirements_benchmark.txt
│   │   └── README_benchmark.md
│   └── comorbidity_methods/
│       ├── dpso_fuzzy.py
│       ├── pso_lpa.py
│       ├── fuzzy_lpa.py
│       ├── significant_comorbidity_vote20.py
│       ├── overlapping_nodes_nsclc.py
│       ├── overlapping_nodes_sclc.py
│       └── README.md
├── Data/
├── Preprocessing_data/
├── Supplementary/
└── README.md
```

Detailed instructions are available in:

* [`Algorithms/benchmark_methods/README_benchmark.md`](Algorithms/benchmark_methods/README_benchmark.md)
* [`Algorithms/comorbidity_methods/README.md`](Algorithms/comorbidity_methods/README.md)

## Requirements

Python **3.10 or newer** is recommended.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/TropBRC-BioinfoLab/DPSO-Fuzzy.git
cd DPSO-Fuzzy

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the benchmark dependencies:

```bash
pip install -r Algorithms/benchmark_methods/requirements_benchmark.txt
```

Install the additional packages used by the comorbidity and preprocessing workflows:

```bash
pip install scikit-learn matplotlib requests venn
```

The principal dependencies are:

* `networkx`
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `cdlib`
* `matplotlib`
* `requests`
* `venn`

`cdlib` is used for ONMI and Omega evaluation and, when available, overlapping-community stability. Some scripts provide fallback calculations when `cdlib` is unavailable.

## Data

The `Data/` directory contains benchmark and lung-cancer comorbidity networks used by the project.

| Dataset  | Graph                          | Ground truth                                       |
| -------- | ------------------------------ | -------------------------------------------------- |
| Karate   | Built into NetworkX            | Derived from the `club` node attribute             |
| Dolphins | `soc-dolphins.mtx`             | Manual two-community definition                    |
| Football | `football.gml`                 | Football ground-truth file                         |
| Yeast-D2 | `Yeast_D2.txt`                 | `Yeast_GT.txt`                                     |
| Y2H      | `Y2H_reconciled_full.edgelist` | `Y2H_GT.txt`                                       |
| NSCLC    | `nsclc_disease_pub.csv`        | Not required for unsupervised comorbidity analysis |
| SCLC     | `sclc_disease_pub.csv`         | Not required for unsupervised comorbidity analysis |

### Comorbidity CSV Columns

The NSCLC and SCLC CSV files use the following relevant columns:

* source node: `txt`
* target node: `txt.1`
* similarity weight: `num.2`

Pass these columns explicitly when running the comorbidity scripts:

```bash
--source-col txt --target-col txt.1 --weight-col num.2
```

This ensures that the ontology-based similarity score is used as the edge weight.

## Benchmark Workflow

The benchmark workflow tunes PSO-LPA, tunes the comparison methods, and evaluates all methods against ground truth.

Supported datasets are:

```text
karate, dolphins, football, yeast, y2h
```

The implemented comparison methods include:

* Fuzzy BLDLP;
* Fuzzy LPA;
* Fuzzy LDPA;
* CFinder;
* LFK;
* Hybrid C-Means;
* NMG.

### Step 1: Tune PSO-LPA

Example for Karate:

```bash
mkdir -p results/benchmark/karate

python Algorithms/benchmark_methods/tune_pso_lpa.py \
  --dataset karate \
  --data-dir Data \
  --runs-per-config 2 \
  --max-configs 30 \
  --out-json results/benchmark/karate/best_pso_lpa_karate.json \
  --out-csv results/benchmark/karate/tuning_pso_lpa_karate.csv
```

Use the complete parameter grid with:

```bash
--full-grid --max-configs 0
```

### Step 2: Tune the Comparison Methods

```bash
python Algorithms/benchmark_methods/tune_othermethods.py \
  --dataset karate \
  --data-dir Data \
  --methods all \
  --runs-per-config 2 \
  --max-configs 30 \
  --out-dir results/benchmark/karate
```

### Step 3: Evaluate All Methods

```bash
python Algorithms/benchmark_methods/evaluate_allmethods.py \
  --dataset karate \
  --data-dir Data \
  --pso-params results/benchmark/karate/best_pso_lpa_karate.json \
  --other-params results/benchmark/karate/best_other_methods_karate.json \
  --n-runs 20 \
  --out results/benchmark/karate/results_all_methods_karate.csv \
  --per-run-out results/benchmark/karate/results_all_methods_karate_per_run.csv
```

For the other benchmark datasets, replace `karate` with `dolphins`, `football`, `yeast`, or `y2h`. Explicit graph and ground-truth paths can be provided using `--graph` and `--gt`.

The benchmark evaluation reports metrics including:

* overlapping normalized mutual information (ONMI);
* Omega index;
* community precision;
* community recall;
* community F1-score.

## NSCLC and SCLC Comorbidity Workflow

Use separate output directories for each method because the scripts generate similarly named `best_*` files.

The following example uses the NSCLC network. For SCLC, replace:

```text
Data/nsclc_disease_pub.csv -> Data/sclc_disease_pub.csv
NSCLC                      -> SCLC
nsclc                      -> sclc
```

### Step 1: Tune and Run PSO-LPA

```bash
python Algorithms/comorbidity_methods/pso_lpa.py \
  --graph Data/nsclc_disease_pub.csv \
  --name NSCLC \
  --outdir results/comorbidity/nsclc/pso_lpa \
  --source-col txt \
  --target-col txt.1 \
  --weight-col num.2
```

PSO-LPA produces the DPSO parameter summary required by DPSO-Fuzzy:

```text
results/comorbidity/nsclc/pso_lpa/nsclc_best_run_summary.json
```

### Step 2: Run Fuzzy-LPA as a Comparison Method

```bash
python Algorithms/comorbidity_methods/fuzzy_lpa.py \
  --graph Data/nsclc_disease_pub.csv \
  --name NSCLC \
  --outdir results/comorbidity/nsclc/fuzzy_lpa \
  --source-col txt \
  --target-col txt.1 \
  --weight-col num.2
```

### Step 3: Tune and Run DPSO-Fuzzy

```bash
python Algorithms/comorbidity_methods/dpso_fuzzy.py \
  --graph Data/nsclc_disease_pub.csv \
  --name NSCLC \
  --dpso-params results/comorbidity/nsclc/pso_lpa/nsclc_best_run_summary.json \
  --outdir results/comorbidity/nsclc/dpso_fuzzy \
  --alpha-threshold 0.4 \
  --source-col txt \
  --target-col txt.1 \
  --weight-col num.2
```

The default workflow performs fuzzy-parameter tuning followed by 20 final runs. The alpha threshold should remain consistent with the value selected during benchmark calibration.

### Step 4: Identify Stable Significant Comorbidities

```bash
python Algorithms/comorbidity_methods/significant_comorbidity_vote20.py \
  --graph Data/nsclc_disease_pub.csv \
  --details-dir results/comorbidity/nsclc/dpso_fuzzy/nsclc_dpso_fuzzy_run_details \
  --outdir results/comorbidity/nsclc/dpso_fuzzy \
  --prefix nsclc \
  --source-col txt \
  --target-col txt.1 \
  --weight-col num.2 \
  --min-community-size 2 \
  --top-k 1 \
  --min-centrality-vote 2 \
  --min-significant-freq 0.50
```

For every community in every final run, the script calculates:

* degree centrality;
* betweenness centrality;
* closeness centrality;
* eigenvector centrality.

A disease becomes a significant candidate when it receives the required number of centrality votes in one run. Stable significant comorbidities are diseases that satisfy the frequency threshold across the repeated runs.

For weighted networks, similarity is converted to distance for betweenness and closeness calculations:

```text
distance = 1 / similarity
```

## DPSO-Fuzzy Outputs

The DPSO-Fuzzy workflow produces files such as:

```text
<prefix>_fixed_dpso_params.json
<prefix>_dpso_fuzzy_tuning.csv
<prefix>_dpso_fuzzy_best_config.json
<prefix>_dpso_fuzzy_20runs.csv
<prefix>_dpso_fuzzy_20runs_summary.csv
<prefix>_dpso_fuzzy_20runs_compact_summary.json
<prefix>_best_run_summary.json
<prefix>_best_memberships.json
<prefix>_best_communities.json
<prefix>_overlapping_nodes.csv
<prefix>_dpso_fuzzy_20runs_details_index.csv
<prefix>_dpso_fuzzy_run_details/
```

The per-run detail directory is used for stable significant-comorbidity voting.

## Overlapping-Node Analysis

The following scripts perform threshold-sensitivity analysis and visualization for the best DPSO-Fuzzy run:

* `Algorithms/comorbidity_methods/overlapping_nodes_nsclc.py`
* `Algorithms/comorbidity_methods/overlapping_nodes_sclc.py`

The analyses include:

* overlapping-node detection at multiple alpha thresholds;
* threshold-sensitivity tables and plots;
* Venn diagrams across thresholds;
* stable overlapping-node identification;
* representative-disease selection;
* overlapping-node and community-network visualization.

These files were exported from Google Colab notebooks. They contain notebook commands, Google Drive paths, and manually selected analysis sections. Run them in Google Colab or convert them into parameterized Python scripts before executing them with a standard Python interpreter.

## Data Preprocessing

The `Preprocessing_data/` directory contains scripts for constructing the NSCLC and SCLC disease networks.

The workflow includes:

1. retrieving lung-cancer and comorbidity publications from PubMed;
2. extracting disease mentions through PubTator;
3. mapping disease mentions to Disease Ontology identifiers;
4. filtering generic or excluded ontology concepts;
5. calculating ontology-based disease similarity;
6. constructing the weighted disease network;
7. extracting the largest connected component for community analysis.

Relevant scripts include:

```text
Preprocessing_data/generated_disease_nsclc.py
Preprocessing_data/generated_disease_sclc.py
Preprocessing_data/similarity_nsclc.py
Preprocessing_data/similarity_sclc.py
```

These preprocessing files were exported from notebooks and contain hard-coded paths and external-service settings. Before running them:

* replace local or Google Drive paths;
* provide required API credentials through environment variables;
* verify PubMed, PubTator, and BioPortal request limits;
* never commit private API keys to the repository.

## Supplementary Results

Benchmark evaluation results are available in:

* [`Supplementary/Benchmark _evaluation.csv`](Supplementary/Benchmark%20_evaluation.csv)

The supplementary table contains the mean and standard deviation of benchmark metrics for DPSO-Fuzzy and the comparison methods.

## Reproducibility

The main scripts support explicit random seeds and repeated execution. For a reproducible experiment, record:

* the exact input graph;
* the graph columns used for source, target, and weight;
* the largest connected component size;
* parameter grids;
* base random seed;
* number of tuning runs;
* number of final runs;
* alpha threshold;
* Python and package versions.

The method outputs are computational candidates. Significant comorbidities and overlapping disease nodes should be validated using clinical data, electronic health records, epidemiological evidence, or multi-omics data before clinical interpretation.

## Citation

If you use this repository, please cite the associated study:

> **A DPSO-Fuzzy Framework for Detecting Significant and Overlapping Disease Communities in Ontology-Based Lung Cancer Comorbidity Networks**

Full author, journal, year, volume, page, and DOI information will be added after publication.

## License

An explicit software license is not currently included in this repository. Contact the repository maintainers before redistributing or incorporating the code into another project.

## Contact

Questions, bug reports, and suggestions can be submitted through the repository's GitHub Issues page.

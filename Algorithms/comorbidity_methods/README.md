# Comorbidity Community Detection Methods

This folder contains Python implementations and analysis scripts for detecting community structures, overlapping disease nodes, and significant comorbidities in weighted disease networks. The workflow is designed for NSCLC and SCLC comorbidity networks stored as CSV edge lists.

## Folder Contents

| File | Purpose |
|---|---|
| `pso_lpa.py` | Tunes and evaluates PSO-LPA/DPSO-BMLPA on a comorbidity network. |
| `fuzzy_lpa.py` | Tunes and evaluates Fuzzy Label Propagation Algorithm (Fuzzy-LPA). |
| `dpso_fuzzy.py` | Runs DPSO-Fuzzy using fixed DPSO parameters obtained from PSO-LPA, tunes the fuzzy-refinement parameters, and performs final repeated runs. |
| `significant_comorbidity_vote20.py` | Identifies stable significant comorbidities using centrality voting across DPSO-Fuzzy run-detail files. |
| `overlapping_nodes_nsclc.py` | Performs overlapping-node threshold sensitivity and visualization for NSCLC. |
| `overlapping_nodes_sclc.py` | Performs overlapping-node threshold sensitivity and visualization for SCLC. |

## Workflow Overview

The recommended workflow is:

1. Run `pso_lpa.py` to tune DPSO parameters.
2. Run `fuzzy_lpa.py` as a comparison method.
3. Run `dpso_fuzzy.py` using the best DPSO parameters produced by `pso_lpa.py`.
4. Run `significant_comorbidity_vote20.py` on the detailed DPSO-Fuzzy outputs.
5. Use the NSCLC or SCLC overlapping-node scripts for threshold-sensitivity analysis and visualization.

The main computational flow is:

```text
Disease network CSV
        |
        +--> PSO-LPA tuning and final runs
        |        |
        |        +--> best DPSO parameter summary
        |
        +--> Fuzzy-LPA tuning and final runs
        |
        +--> DPSO-Fuzzy
                 |
                 +--> fuzzy parameter tuning
                 +--> final repeated runs
                 +--> best memberships
                 +--> overlapping nodes
                 +--> per-run community details
                          |
                          +--> significant-comorbidity voting
                          +--> overlapping-node sensitivity analysis
```

## Requirements

Install the main dependencies with:

```bash
pip install networkx numpy pandas scipy scikit-learn matplotlib cdlib venn
```

The scripts use the following packages:

- `networkx` for graph construction, modularity, and centrality.
- `numpy` and `pandas` for numerical analysis and tabular outputs.
- `scikit-learn` for normalized mutual information.
- `matplotlib` for plots.
- `cdlib` for overlapping NMI when available in `dpso_fuzzy.py`.
- `venn` for multi-set overlapping-node diagrams.

`cdlib` is optional for part of the DPSO-Fuzzy stability calculation. When it is unavailable, the script falls back to NMI calculated from dominant community labels.

## Input Graph Format

The main method scripts expect a CSV edge list.

Supported column structures include:

```text
source,target
```

```text
source,target,weight
```

```text
disease1,disease2,similarity
```

If column names are not recognized, the first two columns are treated as source and target nodes. A numeric third column may be used as the edge weight.

Example:

```csv
source,target,weight
chronic obstructive pulmonary disease,heart disease,0.72
heart disease,stroke,0.68
stroke,pulmonary embolism,0.61
```

Custom column names can be supplied with:

```bash
--source-col <column>
--target-col <column>
--weight-col <column>
```

The method scripts remove self-loops and operate on the largest connected component of the disease graph.

## 1. PSO-LPA

`pso_lpa.py` implements a DPSO-BMLPA-based PSO-LPA candidate method.

The procedure:

1. DPSO optimizes a hard community partition using weighted modularity.
2. A BMLPA-like neighborhood rule identifies potential overlapping nodes.
3. Fuzzy-like memberships are constructed from the primary DPSO community and neighboring community labels.
4. Hyperparameters are selected using a composite score based on:
   - fuzzy modularity;
   - stability;
   - membership entropy.
5. The selected configuration is run repeatedly, with 20 final runs by default.

### Default Parameter Grid

- Inertia weight `w`: `0.5, 0.7, 0.9`
- Cognitive coefficient `c1`: `1.0, 1.5, 2.0`
- Social coefficient `c2`: `1.0, 1.5, 2.0`
- Number of particles: `50, 100`
- Maximum iterations: `100, 300`
- Fixed `rho`: `0.5`
- Fixed initialization parameter `p`: `0.6`

### NSCLC Example

```bash
python pso_lpa.py \
  --graph /home/toto/Eska/Data/nsclc_disease.csv \
  --name NSCLC \
  --outdir /home/toto/Eska/top3/results_pso_lpa/nsclc
```

### SCLC Example

```bash
python pso_lpa.py \
  --graph /home/toto/Eska/Data/sclc_disease.csv \
  --name SCLC \
  --outdir /home/toto/Eska/top3/results_pso_lpa/sclc
```

### Important Outputs

For a network prefix such as `nsclc`, the script produces:

```text
nsclc_pso_lpa_tuning.csv
nsclc_pso_lpa_20runs.csv
nsclc_pso_lpa_20runs_summary.csv
nsclc_pso_lpa_20runs_compact_summary.json
nsclc_best_run_summary.json
nsclc_best_communities.json
nsclc_best_memberships.json
nsclc_overlapping_nodes.csv
```

The `<prefix>_best_run_summary.json` file is used as the DPSO parameter input for `dpso_fuzzy.py`.

## 2. Fuzzy-LPA

`fuzzy_lpa.py` tunes and evaluates Fuzzy-LPA.

The procedure:

1. Initialize each node with a unique label.
2. Propagate the most frequent neighboring label until convergence or `max_iter`.
3. Calculate fuzzy memberships using:

```text
mu = 1 - exp(-lambda * gamma)
```

4. Retain memberships that satisfy the selected threshold.
5. Select the best parameter combination using fuzzy modularity, stability, and entropy.
6. Execute the selected configuration over the final repeated runs.

### Default Parameter Grid

- Membership threshold: `0.01, 0.05, 0.10`
- Lambda: `1.0, 2.0, 5.0`
- Maximum iterations: `50, 100, 150`

### NSCLC Example

```bash
python fuzzy_lpa.py \
  --graph /home/toto/Eska/Data/nsclc_disease.csv \
  --name NSCLC \
  --outdir /home/toto/Eska/top3/results_fuzzy_lpa/nsclc
```

### SCLC Example

```bash
python fuzzy_lpa.py \
  --graph /home/toto/Eska/Data/sclc_disease.csv \
  --name SCLC \
  --outdir /home/toto/Eska/top3/results_fuzzy_lpa/sclc
```

### Important Outputs

```text
<prefix>_fuzzy_lpa_tuning.csv
<prefix>_fuzzy_lpa_20runs.csv
<prefix>_fuzzy_lpa_20runs_summary.csv
<prefix>_fuzzy_lpa_20runs_compact_summary.json
<prefix>_best_run_summary.json
<prefix>_best_communities.json
<prefix>_best_memberships.json
<prefix>_overlapping_nodes.csv
```

Because several methods generate similarly named `best_*` files, use separate output directories for PSO-LPA, Fuzzy-LPA, and DPSO-Fuzzy.

## 3. DPSO-Fuzzy

`dpso_fuzzy.py` combines DPSO hard-community initialization with an MSEFCD-inspired fuzzy-refinement process.

The procedure:

1. Load fixed DPSO parameters from the PSO-LPA best-run summary.
2. Generate hard DPSO communities.
3. Construct the modularity matrix and similarity matrix.
4. Initialize fuzzy memberships using neighboring hard labels and the DPSO prior.
5. Alternate:
   - one membership-smoothing step;
   - one DPSO-guided membership-enhancement step.
6. Tune fuzzy parameters using unsupervised internal metrics.
7. Execute the selected configuration for the final runs.
8. Save detailed memberships, communities, and overlapping-node outputs.

### Default Fuzzy Parameter Grid

- `alpha_self : beta_neighbor`:
  - `0.8 : 0.2`
  - `0.7 : 0.3`
  - `0.6 : 0.4`
  - `0.5 : 0.5`
- Prior boost: `0.05, 0.08, 0.10, 0.15`
- Candidate communities: `2, 3, 4`
- Fuzzy iterations: `12`
- Membership threshold: `0.4`

The membership threshold should be kept consistent with the threshold selected during benchmark calibration.

### NSCLC Example

```bash
python dpso_fuzzy.py \
  --graph /home/toto/Eska/Data/nsclc_disease.csv \
  --name NSCLC \
  --dpso-params /home/toto/Eska/top3/results_pso_lpa/nsclc/nsclc_best_run_summary.json \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/nsclc \
  --alpha-threshold 0.4
```

### SCLC Example

```bash
python dpso_fuzzy.py \
  --graph /home/toto/Eska/Data/sclc_disease.csv \
  --name SCLC \
  --dpso-params /home/toto/Eska/top3/results_pso_lpa/sclc/sclc_best_run_summary.json \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/sclc \
  --alpha-threshold 0.4
```

### Important Outputs

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

The run-detail directory contains one JSON file for each final run. These files are required by `significant_comorbidity_vote20.py`.

## 4. Significant Comorbidity Voting

`significant_comorbidity_vote20.py` identifies stable significant comorbidities from the detailed DPSO-Fuzzy results.

For every run and every detected community, the script calculates:

- degree centrality;
- betweenness centrality;
- closeness centrality;
- eigenvector centrality.

Nodes receive one vote for each centrality measure in which they are among the top-ranked nodes. A node is considered a significant candidate in a run when its centrality vote reaches the selected minimum.

Stable significant comorbidities are then identified according to how frequently they appear as significant candidates across all runs.

### Default Rules

- Minimum community size: `2`
- Top nodes per centrality: `1`
- Minimum centrality vote per run: `2` out of `4`
- Minimum significant frequency: `0.50`

With 20 runs and a frequency threshold of `0.50`, a disease must be significant in at least 10 runs.

### NSCLC Example

```bash
python significant_comorbidity_vote20.py \
  --graph /home/toto/Eska/Data/nsclc_disease.csv \
  --details-dir /home/toto/Eska/top3/results_dpso_fuzzy/nsclc/nsclc_dpso_fuzzy_run_details \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/nsclc \
  --prefix nsclc \
  --min-community-size 2 \
  --top-k 1 \
  --min-centrality-vote 2 \
  --min-significant-freq 0.50
```

### SCLC Example

```bash
python significant_comorbidity_vote20.py \
  --graph /home/toto/Eska/Data/sclc_disease.csv \
  --details-dir /home/toto/Eska/top3/results_dpso_fuzzy/sclc/sclc_dpso_fuzzy_run_details \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/sclc \
  --prefix sclc \
  --min-community-size 2 \
  --top-k 1 \
  --min-centrality-vote 2 \
  --min-significant-freq 0.50
```

The script may also read the run-detail index:

```bash
python significant_comorbidity_vote20.py \
  --graph /path/to/network.csv \
  --details-index /path/to/<prefix>_dpso_fuzzy_20runs_details_index.csv
```

### Outputs

```text
<prefix>_centrality_per_community_20runs.csv
<prefix>_significant_candidates_20runs.csv
<prefix>_significant_frequency_vote20.csv
<prefix>_stable_significant_comorbidities_vote20.csv
<prefix>_significant_vote20_summary.json
```

For weighted graphs, the edge weight is treated as similarity. Betweenness and closeness therefore use inverse similarity as distance:

```text
distance = 1 / weight
```

## 5. Overlapping-Node Analysis

The two overlapping-node scripts perform additional analysis on the best DPSO-Fuzzy run:

- `overlapping_nodes_nsclc.py`
- `overlapping_nodes_sclc.py`

Their analyses include:

1. Loading the best fuzzy membership matrix.
2. Detecting overlapping nodes at several alpha thresholds.
3. Measuring the sensitivity of the number of overlapping nodes.
4. Comparing threshold-specific node sets using Venn diagrams.
5. Identifying stable overlapping nodes.
6. Extracting community memberships for stable nodes.
7. Selecting representative diseases from connected communities.
8. Producing network visualizations.

### Default Thresholds

```text
0.30, 0.35, 0.40, 0.45, 0.50
```

A disease is treated as overlapping when it has membership values greater than or equal to the selected threshold in at least two communities.

### Important Note

The two files were exported from Google Colab notebooks. They currently contain notebook-specific statements and hard-coded paths, including:

```python
!pip install venn
```

and, in the SCLC script:

```python
from google.colab import drive
drive.mount(...)
```

They should therefore be run as notebook cells in Google Colab, or converted into standalone command-line scripts before being executed with a standard Python interpreter.

Before using them, update:

- `base_dir` or `best_dir`;
- `membership_file`;
- `summary_file`;
- `graph_path`;
- `out_dir`;
- the manually selected stable-node section where applicable.

The NSCLC and SCLC scripts are analysis notebooks rather than fully parameterized command-line programs.

## Composite Selection Score

The method scripts use the following internal selection logic:

```text
Composite = mean(M_norm, S_norm, 1 - E_norm)
```

where:

- `M_norm` is normalized fuzzy modularity;
- `S_norm` is normalized stability;
- `E_norm` is normalized membership entropy.

Higher modularity and stability are preferred, while lower entropy is preferred.

This score is used for unsupervised parameter selection and for ranking the final repeated runs. It is not a clinical validation metric.

## Recommended Directory Structure

```text
comorbidity_methods/
├── README.md
├── dpso_fuzzy.py
├── fuzzy_lpa.py
├── pso_lpa.py
├── significant_comorbidity_vote20.py
├── overlapping_nodes_nsclc.py
└── overlapping_nodes_sclc.py
```

A suggested results structure is:

```text
results/
├── pso_lpa/
│   ├── nsclc/
│   └── sclc/
├── fuzzy_lpa/
│   ├── nsclc/
│   └── sclc/
└── dpso_fuzzy/
    ├── nsclc/
    └── sclc/
```

## Reproducibility

All primary method scripts support a base random seed:

```bash
--seed 1234
```

Tuning and final evaluation use different seed ranges to reduce accidental reuse of identical runs.

For a reproducible experiment, record:

- the input graph;
- the largest connected component size;
- parameter grids;
- the base seed;
- the number of tuning runs;
- the number of final runs;
- the membership threshold;
- package versions.

## Interpretation Notes

The detected communities, significant comorbidities, and overlapping disease nodes are computational outputs.

- A significant comorbidity is a disease that is repeatedly central within detected communities.
- An overlapping node is a disease with sufficiently high fuzzy membership in more than one community.
- Neither result automatically establishes clinical comorbidity or causality.
- Candidate findings should be validated using clinical records, electronic health records, epidemiological evidence, or multi-omics data.

## Known Limitations

- The main graph loaders currently expect CSV edge lists.
- All main analyses use only the largest connected component.
- Parameter tuning is unsupervised and depends on internal graph metrics.
- The overlapping-node scripts contain hard-coded Colab paths and manual analysis sections.
- File names such as `<prefix>_best_run_summary.json` are reused by different methods, so separate method-specific output directories are necessary.
- Large parameter grids and repeated DPSO runs can require substantial computation time.

## License and Citation

Add the project license and the preferred citation for the associated thesis or publication before distributing this repository.

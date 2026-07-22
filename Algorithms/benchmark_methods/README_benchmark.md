# Benchmark Community Detection

Main files:

* `yeast.py`: tunes PSO-LPA and saves the best parameter configuration.
* `hypeyeast_all.py`: tunes all comparison algorithms and saves their best parameter configurations.
* `allmethodyeast.py`: retrains and evaluates all methods.
* `benchmark_common.py`: provides shared graph loaders, ground-truth loaders, evaluation functions, and utilities.

## Dependencies

```bash
pip install networkx numpy pandas scipy cdlib
```

`cdlib` is required to calculate ONMI and Omega. If it is unavailable, the remaining metrics will still be calculated, while ONMI and Omega will be recorded as `NaN`.

## Supported Dataset Structure

Place all dataset files in a single data directory, for example:

```text
/home/toto/Eska/Data
```

| Dataset  | Graph                          | Ground truth                                     |
| -------- | ------------------------------ | ------------------------------------------------ |
| Karate   | Built into NetworkX            | Manually obtained from the `club` node attribute |
| Dolphins | `soc-dolphins.mtx`             | Manual two-community partition                   |
| Football | `football.gml`                 | `gootball_GT.txt` or `football_GT.txt`           |
| Yeast    | `Yeast_D2.txt`                 | `Yeast_GT.txt`                                   |
| Y2H      | `Y2H_reconciled_full.edgelist` | `Y2H_GT.txt`                                     |

For the Football dataset, if the graph or ground-truth filenames differ from those listed in the table, specify them explicitly using `--graph` and `--gt`.

## Execution Order

Replace `DATA_DIR` with the path to your data directory.

### 1. Tune PSO-LPA

```bash
python yeast.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data \
  --runs-per-config 2 \
  --max-configs 30
```

Main outputs:

* `best_pso_lpa_yeast.json`
* `tuning_pso_lpa_yeast.csv`

Use the following options to evaluate the complete parameter grid:

```bash
--full-grid --max-configs 0
```

### 2. Tune the Comparison Algorithms

```bash
python hypeyeast_all.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data \
  --methods all \
  --runs-per-config 2 \
  --max-configs 30
```

Main outputs:

* `best_other_methods_yeast.json`
* One tuning CSV file for each method.

### 3. Evaluate All Methods

```bash
python allmethodyeast.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data \
  --n-runs 20
```

Outputs:

* `results_all_methods_yeast.csv`
* `results_all_methods_yeast_per_run.csv`

## Commands for Each Dataset

### Karate

```bash
python yeast.py \
  --dataset karate \
  --data-dir /home/toto/Eska/Data

python hypeyeast_all.py \
  --dataset karate \
  --data-dir /home/toto/Eska/Data \
  --methods all

python allmethodyeast.py \
  --dataset karate \
  --data-dir /home/toto/Eska/Data \
  --n-runs 20
```

### Dolphins

```bash
python yeast.py \
  --dataset dolphins \
  --data-dir /home/toto/Eska/Data

python hypeyeast_all.py \
  --dataset dolphins \
  --data-dir /home/toto/Eska/Data \
  --methods all

python allmethodyeast.py \
  --dataset dolphins \
  --data-dir /home/toto/Eska/Data \
  --n-runs 20
```

### Football

```bash
python yeast.py \
  --dataset football \
  --data-dir /home/toto/Eska/Data \
  --graph football.gml

python hypeyeast_all.py \
  --dataset football \
  --data-dir /home/toto/Eska/Data \
  --graph football.gml \
  --methods all

python allmethodyeast.py \
  --dataset football \
  --data-dir /home/toto/Eska/Data \
  --graph football.gml \
  --gt gootball_GT.txt \
  --n-runs 20
```

If the ground-truth file is named `football_GT.txt`, replace:

```bash
--gt gootball_GT.txt
```

with:

```bash
--gt football_GT.txt
```

### Yeast-D2

```bash
python yeast.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data

python hypeyeast_all.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data \
  --methods all

python allmethodyeast.py \
  --dataset yeast \
  --data-dir /home/toto/Eska/Data \
  --n-runs 20
```

### Y2H

```bash
python yeast.py \
  --dataset y2h \
  --data-dir /home/toto/Eska/Data

python hypeyeast_all.py \
  --dataset y2h \
  --data-dir /home/toto/Eska/Data \
  --methods all

python allmethodyeast.py \
  --dataset y2h \
  --data-dir /home/toto/Eska/Data \
  --n-runs 20
```

## Ground-Truth File Formats

The following format is supported automatically:

```text
C1: node1 node2 node3
C2: node4 node5 node6
```

A node-to-community assignment format is also supported:

```text
node1 C1
node2 C1
node3 C2
```

The ground truth may also contain one community per line:

```text
node1 node2 node3
node4 node5 node6
```

Node identifiers are matched against the graph nodes automatically. Therefore, string-based ground-truth identifiers and integer graph node identifiers can be aligned when they represent the same values.

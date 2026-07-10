## Get started
We introduce the unlearning method FedUCurv, which is a client-level unlearning technique for moderately sized deep learning models in cross-silo federated learning settings with small, heterogeneous and tabular local datasets. It is designed such that the global server never receives the raw tabular data or parameter gradients. We explain how the method works, and how the models were created that we tested the method on. 

---

### 1. Use the environment
The file 'environment.yml' contains all the necessary packages. Simply create an environment by installing conda if you have not already and 
running. You can then create and activate an environment in the following way:

```bash
conda env create -f environment.yml
conda activate fedl_env
```


---

### 2. Replicate the data
### 2.1 BPI data
The experiments in our research are executed on the BPI challenge dataset, which you can find here: https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1. There you can download this data as a .xes file. To preprocess the data so that it is suitable for the experiments, run the preprocessing script on the raw log file:

```bash
python preprocessing.py \
  --input BPI_Challenge_2019.xes \
  --output-dir .
```

this writes:

- `raw_data.pkl`
- `raw_train_data.pkl`
- `raw_test_data.pkl`
- `preprocessing_audit.json`

This will create a training  data set and a testing data set. The training data set will be later divided 
into training data and validation data. These experiments use data from the 
BPI challenge, which is financial audit data. In the original data set, 
each row contains an event. We have modified the data so that each row 
contains information about one order. The audit contains information about the
features such as includes label counts and missingness counts.

### 2.2 Artificial data
We also ran some experiments on artificial data to observe how the method would perform in a controlled environment under different data distributions. To create the artificial data, you can run create_artificial_data.py. Several parameter configurations are already predefined (notably the three mentioned in our research), and can be accessed by defining the `--experiment' argument.

## 3. create the client split

The next step is deviding the data over the different clients. The default splitting method is `balanced_feature_clusters`. It clusters processed train
features, rebalances client sizes, and keeps a readable audit. You can create the default split from python:

```bash
python save_default_partitions.py
```

this writes:

- `partitions.pkl`
- `split_audit.json`

The audit records client size, blocked rate, top categories, and the
pairwise blocked-rate spread.

## 4. run the main notebook flow

`create_models.ipynb` is the main experiment notebook. it now assumes:

- `train_data.pkl` and `test_data.pkl` already exist
- `raw_train_data.pkl` is available for the split audit
- the default bpi run is the basic method only
- model width is loaded from `feature_metadata.json`

if you want a headless full run, use:

```bash
python run_experiment.py \
  --xes-path BPI_Challenge_2019.xes \
  --data-dir . \
  --output-dir 1000-0.1-0.05-0.1-10-0-basic
```

the notebook creates:

- `non_fd_params.pt`
- `og_params.pt`
- `og_summaries`
- `og_hessian`
- `shadow_params.pt`
- `large_attack_params.pt`
- `small_attack_params.pt`
- `og_val_results`
- `og_test_results`
- `non_fd_val_results`
- `non_fd_test_results`
- `shadow_val_results`
- `shadow_test_results`
- `MIA_small_results`
- `MIA_large_results`
- `predictive_results`
- `unlearning_summary_small`
- `unlearning_summary_large`
- `KL_results`
- `L2_results`
- `experiment_summary.json` if you use `run_experiment.py`
- one folder per client with `retr_*`, `unl_*`, and `ft_*` artifacts

the MIA result files store per-client dictionaries. each client now has
`original`, `retrain`, `unlearn`, and `finetune` entries, and each entry keeps
metrics like `loss`, `acc`, `auc`, `balanced_acc`, `member_score`,
`non_member_score`, and `forgotten_score`.

the standard utility result files also keep:

- `f1`
- `auc`
- `brier`
- `ece`
- `positive_rate`
- `valid_rate`
- `non_finite_rate`

`predictive_results` adds the richer empirical certificate metrics per client
pair, including `kl`, `js`, `tv`, `agreement`, `logit_l2`, `logit_rmse`, and
`prob_rmse`.

`unlearning_summary_small` and `unlearning_summary_large` combine those
predictive metrics with:

- forgotten-client validation gaps
- retained-client mean validation gaps
- global test gaps
- privacy gaps versus retrain

## result layout

After a full run, a result folder looks like this:

```text
5-0.1-0.05-0.1-10-0-basic/
├── partitions.pkl
├── split_audit.json
├── og_params.pt
├── og_summaries
├── og_hessian
├── og_val_results
├── og_test_results
├── non_fd_params.pt
├── non_fd_val_results
├── non_fd_test_results
├── shadow_params.pt
├── shadow_val_results
├── shadow_test_results
├── small_attack_params.pt
├── large_attack_params.pt
├── MIA_small_results
├── MIA_large_results
├── predictive_results
├── unlearning_summary_small
├── unlearning_summary_large
├── KL_results
├── L2_results
├── experiment_summary.json
├── 0/
│   ├── retr_params.pt
│   ├── retr_val_results
│   ├── retr_test_results
│   ├── unl_params.pt
│   ├── unl_val_results
│   ├── unl_test_results
│   ├── ft_params.pt
│   ├── ft_val_results
│   └── ft_test_results
└── ...
```

## 5. Perfom additional experiments

Once you have trained a model, you can now further experiment with it. With the file `study_unlearning_selectors`, you can test out different hyperparameter configurations, as discussed in our research. With `study_client_mia`, you create different deletion inference attack models and test them on your selected unlearning results. Simply specify the folder where you have stored your experiments and run the files.

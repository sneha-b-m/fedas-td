# FedAS-TD: Drift-Aware Personalized Federated Learning Under Temporal Client Shift

This repository contains my modified version of **FedAS** with an added **temporal-drift-aware personalization and aggregation mechanism**.

The project extends the original FedAS idea by detecting when a client's data distribution changes over time and then using that information in:

- client-side personalization
- server-side aggregation

## Base Project

This work is built on top of:

- **FedAS**: *Bridging Inconsistency in Personalized Federated Learning*  
  Xiyuan Yang, Wenke Huang, Mang Ye  
  CVPR 2024  
  Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html

This codebase is also adapted from the PFLlib-style project structure.

## What I Modified

The original FedAS already uses:

- Fisher-trace-based client weighting
- prototype-based parameter alignment
- personalized local/global model interaction

My modification adds **drift awareness**.

### 1. Temporal Drift Simulation

Clients are exposed to a round-dependent subset of classes using:

- `--temporal_drift`
- `--drift_interval`
- `--drift_labels_per_phase`

This simulates changing local data distributions over time.

### 2. Drift Score for Each Client

For each client, I compute a drift score using three signals:

- **Fisher change**
- **prototype shift**
- **loss spike**

The combined drift score is:

`D_k^(t) = clip(λ_F ΔF_k^(t) + λ_P P_k^(t) + λ_L S_k^(t), 0, D_max)`

### 3. Adaptive Personalization

Instead of always fully replacing the local base model with the aligned global base, the modified version blends them:

`θ_k,base ← (1 - α_k^(t)) θ_g,aligned + α_k^(t) θ_k,old`

- low drift: accept more global knowledge
- high drift: retain more local knowledge

### 4. Drift-Aware Aggregation

Original FedAS trusts only Fisher magnitude:

`w_k = F_k / ∑_j F_j`

Modified FedAS-TD discounts unstable clients:

`w̃_k = F_k / (1 + β D_k)`

`w_k = w̃_k / ∑_j w̃_j`

This means a client must be both:

- important
- stable

to strongly influence the global model.

## Repository Structure

```text
fedas-td/
|-- dataset/                  # dataset generation scripts
|-- system/
|   |-- flcore/
|   |   |-- clients/
|   |   |   `-- clientas.py   # modified client-side drift logic
|   |   |-- servers/
|   |   |   `-- serveras.py   # modified drift-aware aggregation
|   |   `-- trainmodel/
|   `-- main.py               # training entry point and CLI flags
`-- README.md
```



## Run on Google Colab
### Step 1: If GPU is available in Colab

Go to:

- `Runtime`
- `Change runtime type`
- select `GPU`



### Step 2: Upload or clone the repository

If the repository is on GitHub:

```bash
!git clone https://github.com/sneha-b-m/fedas-td.git
%cd fedas-td
```

If you are using a zip file, upload and extract it in Colab.

### Step 3: Install dependencies

```bash
!pip install torch torchvision numpy ujson opacus torchviz calmsize
```

### Step 4: Generate the dataset

```bash
%cd /content/fedas-td/dataset
!python generate_cifar10.py noniid - dir
```

### Step 5: Run the code

```bash
%cd /content/fedas-td/system
!python main.py -did 0 -data Cifar10 -nb 10 -m cnn -lbs 16 -gr 40 -ls 5 -algo FedAS -jr 0.4 -nc 20 --temporal_drift --drift_interval 5 --drift_labels_per_phase 2 --drift_aware
```




###  Meaning of the main arguments

- `-did 0` : device id
- `-data Cifar10` : dataset name
- `-nb 10` : number of classes
- `-m cnn` : model
- `-lbs 16` : local batch size
- `-gr 40` : global rounds
- `-ls 5` : local epochs
- `-algo FedAS` : algorithm name
- `-jr 0.4` : join ratio
- `-nc 20` : number of clients
- `--temporal_drift` : enable temporal drift simulation
- `--drift_interval 5` : shift client data every 5 rounds
- `--drift_labels_per_phase 2` : classes visible in one drift phase
- `--drift_aware` : enable the modified FedAS-TD method



Default values:

- `drift_fim_weight = 1.0`
- `drift_proto_weight = 1.0`
- `drift_loss_weight = 0.5`
- `drift_clip = 2.0`
- `drift_beta = 1.0`
- `personalization_min = 0.1`
- `personalization_max = 0.9`



## Summary

FedAS-TD improves the original FedAS by making the method aware of temporal client drift.

Main contribution:

- compute a drift score per client
- use drift in personalization
- use drift in aggregation
- improve robustness under changing client distributions

## Citation

If you want to cite the original FedAS paper:

```bibtex
@inproceedings{cvpr24_xiyuan_fedas,
    author    = {Yang, Xiyuan and Huang, Wenke and Ye, Mang},
    title     = {FedAS: Bridging Inconsistency in Personalized Fedearated Learning},
    booktitle = {CVPR},
    year      = {2024}
}
```

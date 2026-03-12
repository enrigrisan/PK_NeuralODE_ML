# PK_NeuralODE_ML

A MATLAB implementation of a **Latent Neural Ordinary Differential Equation (Neural ODE)** model for population pharmacokinetics (PK), built using the Deep Learning Toolbox. The model learns continuous-time drug concentration trajectories from sparse, irregularly sampled patient observations following a bolus injection.

Current application is for a Tacrolimus PK model [D. Chen, et al. (2023)]
The generation of the synthetic population is from https://github.com/Dperazzolo/tacrolimus_data_generator 

---
## Overview

Following a bolus dose **D** at **t = 0**, drug concentrations **C(t)** are measured over **t = 0..48 h**. Each patient has a unique PK profile driven by individual parameters (clearance, volume of distribution, etc.). The model encodes these dynamics into a structured latent space and integrates forward in continuous time using a neural ODE.

<img width="1630" height="981" alt="Tacrolimus_LatentNODE_traintest" src="https://github.com/user-attachments/assets/da56b6d7-d93d-4003-87a8-45a9c5a14010" />

---

## Model Architecture

The latent space is split into two components:

| Component | Symbol | Role |
|---|---|---|
| Dynamic state | **z\_dyn** | Encodes the current and past drug concentration trajectory |
| PK characteristics | **z\_pk** | Encodes patient-level PK properties from sparse, irregular observations |

The full latent vector is **z = [z\_dyn ‖ z\_pk]**, integrated forward by a neural ODE whose drift function **f(z, t; θ)** is parameterised by a feedforward network.

---

## Components

### 1. Dynamic Encoder
- **Architecture**: GRU recurrent network
- **Input**: Full observed sequence `[C(t); t]` of shape `[2 × T]`
- **Output**: Posterior parameters `[μ_dyn, log σ²_dyn]` for the dynamic initial state `z_dyn(0)`
- **Purpose**: Captures temporal ordering and concentration history up to `t = 0`

### 2. PK Encoder (Patient-Level)
- **Architecture**: Pointwise fully-connected network + mean pooling (permutation-invariant set encoder)
- **Input**: Sparse, irregularly sampled observations `[C(tᵢ); tᵢ]` at 10 random time points (different per patient)
- **Output**: Posterior parameters `[μ_pk, log σ²_pk]` for the static PK embedding `z_pk`
- **Purpose**: Summarises individual PK characteristics (e.g. clearance, volume) into a fixed-length representation, invariant to the ordering of observations

### 3. Neural ODE (Latent Dynamics)
- **Architecture**: Time-augmented feedforward network `f([z; t]; θ) → dz/dt`
- **Integration**: `dlode45` (Runge-Kutta 45 with adjoint-method gradients)
- **Input**: Augmented state `[z_dyn; z_pk; t]` at each solver step
- **Output**: Continuous latent trajectory `Z(t)` over `t = 0..48 h`
- **Note**: Only `z_dyn` evolves meaningfully; `z_pk` is held constant in the initial condition and acts as a conditioning signal throughout integration

### 4. Decoder
- **Architecture**: Pointwise fully-connected network with softplus output
- **Input**: Dynamic latent trajectory `z_dyn(t)` at each time point
- **Output**: Predicted concentration `Ĉ(t) ≥ 0`

---

## Latent Space Design

```
z(0) = [ z_dyn(0) | z_pk ]
         ────────   ────────
         8-dim      4-dim
         evolves    fixed
         via ODE    (patient-level anchor)
```

The split ensures that:
- `z_dyn` captures **transient dynamics** (e.g. distribution and elimination phases)
- `z_pk` captures **static inter-patient variability** (e.g. fast vs slow metabolisers)
- The decoder only reads `z_dyn`, keeping the PK embedding's influence mediated through the initial condition

---

## Training Objective

The model is trained with a **β-ELBO** (Evidence Lower Bound):

```
L = E[||C(t) - Ĉ(t)||²] + β · KL(q(z_dyn|x) || N(0,I))
                         + β · KL(q(z_pk|x)  || N(0,I))
```

- **Reconstruction term**: Modified MAE (MAE divided by the avreage concentration) between predicted and observed concentrations over all 49 time points (`t = 0..48 h`), with `t = 0` prepended from the encoded initial condition
- **KL terms**: Regularise both latent posteriors towards a standard normal prior
- **β = 0.01**: Small weight preserving reconstruction fidelity while regularising the latent space

Optimised with **Adam** over mini-batches of 16 patients.

---

## Requirements

- MATLAB R2023b or later
- Deep Learning Toolbox (for `dlode45`, `dlarray`, `dlfeval`, `dlgradient`)
- No additional toolboxes required

---

## Configuration

Key hyperparameters are set in the `cfg` struct at the top of `main.m`:

| Parameter | Default | Description |
|---|---|---|
| `nPatients` | 200 | Number of training subjects |
| `dimZdyn` | 8 | Latent dimension for dynamic state |
| `dimZpk` | 4 | Latent dimension for PK characteristics |
| `dimHidden` | 64 | Hidden units in all networks |
| `nEpochs` | 1000 | Training epochs |
| `lrInit` | 1e-3 | Adam learning rate |
| `batchSize` | 32 | Mini-batch size |
| `rtolODE` | 1e-3 | Relative tolerance for `dlode45` |
| `atolODE` | 1e-4 | Absolute tolerance for `dlode45` |

---

## References

- R.T. Chen, et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
- Y. Rubanova, et al. (2019). *Latent ODEs for Irregularly-Sampled Time Series*. NeurIPS.
- MathWorks (2023). *Deep Learning Toolbox — dlode45 documentation*.
- R. Venkataramanan, et al. (1995) *Clinical pharmacokinetics of tacrolimus* Clinical Pharmacokinetics,
vol. 29, pp. 404–430, 1995.
- D. Chen, et al. (2023) *Population pk/pd model of tacrolimus for exploring the relationship between accumulated exposure and quantitative scores in myasthenia gravis patients* CPT: Pharmacometrics & Systems Pharmacology, vol. 12, no. 7, pp. 963–976,

# CoSL: Collaborative Spectral Learning for Temporal Domain Generalization in Large-Scale Specific Emitter Identification

# Submit to TCCN

CoSL targets **cross-day specific emitter identification (SEI) with hundreds of emitters**, where the model is trained on source days only and tested on unseen target days. A spectral analysis shows that, after channel equalization, transmitter fingerprints are preserved in the spectrum as structured, device-dependent patterns. CoSL exploits this with two components:

- **SpecFormer** — a spectral Transformer for each receiver. A *learnable spectral recalibration* (LSR) module applies `F` complex-valued filters that reweight the magnitude and phase of every frequency bin, and a lightweight Transformer encoder models cross-frequency dependencies over spectral patches.
- **Training-free collaborative decision** — the posteriors of the receivers observing the same emitter are fused by **geometric averaging**, i.e., a logarithmic-pooling MAP decision rule, with no target-day data and no extra parameters.

On WiSig ManyTx (150 transmitters, 18 receivers, 4 days), SpecFormer outperforms general-purpose and SEI-specific baselines with only 0.24–0.64 M parameters, and multi-receiver fusion raises the accuracy to about 79% / 90% / 92% under the three protocols.

---

## Repository structure

```
CoSL/
├── run_cosl.py             # train / test entry point, multi-receiver fusion
├── utils/
│   ├── load_data.py        # .pkl loader, per-sample power normalization
│   └── model.py            # SpecFormer (LSR + Transformer encoder)
├── dataset/                # <-- put WiSig ManyTx data here (see below)
└── weights/                # checkpoints (created at runtime)
```

## Environment

The results in the paper were obtained with:

| Component | Version |
|-----------|---------|
| Python    | 3.12 |
| PyTorch   | 2.5.1 |
| GPU       | NVIDIA V100 |

Minimal dependencies:

```bash
pip install torch numpy
```

## Data and pretrained weights

The preprocessed dataset and pretrained checkpoints are shared via Baidu Netdisk:

- **Link:** `<link>`
- **Code:** `<code>`

After downloading, place the files so the directory layout matches the loader.

### WiSig ManyTx

`utils/load_data.py` expects per-(receiver, day) pickle files:

```
dataset/ManyTx/<equalized|non_equalized>/date<d>/rx_<rx-id>_data.pkl
```

where `<d> ∈ {1,2,3,4}`, and each `.pkl` holds a dict with `data[tx_index]` of shape `(N, 256, 2)` (the loader transposes to `(N, 2, 256)`, keeps the first 50 samples per Tx, and applies per-sample power normalization). Receiver ids follow `rx_indexes_of_manytx` defined in the loader. The equalized version is used by default.

| Subset | Transmitters | Receivers | Days | Sample length |
|--------|:------------:|:---------:|:----:|:-------------:|
| ManyTx | 150          | 18        | 4    | 256           |

### Pretrained weights

Checkpoints are stored as

```
weights/dep<L_e>_p16_f4_d<source days>_sd<seed>/specformer_flt4_sd<seed>_dim128_dep<L_e>_h4_p16_do0.3_tx150_equalized.pt
```

e.g. `weights/dep1_p16_f4_d1_sd0/specformer_flt4_sd0_dim128_dep1_h4_p16_do0.3_tx150_equalized.pt` for SpecFormer-S under P1 with seed 0.

## Quick start

Three temporal domain generalization protocols are used. Target-day data are never seen during training; 30% of the source-day data is held out for validation and early stopping. Results are averaged over seeds `0–4`.

| Protocol | `--protocol` | Source days | Target days |
|:--------:|:------------:|:-----------:|:-----------:|
| P1 | `a` | Day 1     | Day 2 ~ 4 |
| P2 | `b` | Day 1 ~ 2 | Day 3 ~ 4 |
| P3 | `c` | Day 1 ~ 3 | Day 4     |

```bash
# test pretrained SpecFormer-S under P1, seed 0
python run_cosl.py --mode test_only --protocol a --depth 1 --seed 0

# train + test SpecFormer-S / M / L (depth 1 / 2 / 3)
python run_cosl.py --mode train_test --protocol a --depth 1 --seed 0

# all protocols, models and seeds
for P in a b c; do
  for D in 1 2 3; do
    for S in 0 1 2 3 4; do
      python run_cosl.py --mode train_test --protocol $P --depth $D --seed $S
    done
  done
done
```

The script reports the single-receiver accuracy and the multi-receiver accuracy of three fusion rules evaluated on the same receiver groups:

```
[result] source days [1] -> target days [2, 3, 4]
    single-Rx (avg) : ...
    fusion hard     : ...    # majority voting
    fusion soft     : ...    # arithmetic averaging
    fusion geo      : ...    # geometric averaging (CoSL)
```

Key arguments (`run_cosl.py`):

| Argument          | Default       | Description |
|-------------------|---------------|-------------|
| `--protocol`      | –             | `a` / `b` / `c` for P1 / P2 / P3 |
| `--mode`          | `test_only`   | `test_only` or `train_test` |
| `--depth`         | `1`           | Transformer depth (1 / 2 / 3 → SpecFormer-S / M / L) |
| `--patch`         | `16`          | spectral patch size |
| `--n_filters`     | `4`           | number of LSR filters `F` |
| `--use_filter`    | `1`           | `0` disables LSR (ablation) |
| `--is_eq`         | `equalized`   | `equalized` or `non_equalized` input |
| `--seed`          | `0`           | random seed |
| `--collab_trials` | `50`          | random receiver groups per emitter for fusion |
| `--dataset_root`  | `dataset`     | data folder |
| `--weights_root`  | `weights`     | checkpoint folder |

## Acknowledgments

This work builds on a publicly available dataset, and we gratefully acknowledge its authors:

- **WiSig** — S. Hanna, S. Karunaratne, and D. Cabric, "WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting," *IEEE Access*, vol. 10, pp. 22808–22818, 2022.

Please cite the dataset if you use it through this repository.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2026cosl,
  author  = {Wang, Yu and Shi, Zheng},
  title   = {{CoSL}: Collaborative Spectral Learning for Temporal Domain Generalization in Large-Scale Specific Emitter Identification},
  journal = {Submit to TCCN},
  year    = {2026}
}
```

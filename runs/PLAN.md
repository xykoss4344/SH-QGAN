# Wave 1 — launched 2026-08-29

8 parallel single-threaded runs, 500 epochs each. Every run has its own
out_folder: a shared output directory is what produced the untraceable epoch-10
std_z recorded in the Experiment Log.

| folder | floors | madelung | sf | force | seed | purpose |
|---|---|---|---|---|---|---|
| w1_base_s0   | literature | - | - | - | 0 | v10 reference, reproduces the epoch-20 regression |
| w1_base_s1   | literature | - | - | - | 1 | seed variance on the reference |
| w1_floors_s0 | data | - | - | - | 0 | **the hypothesis test** |
| w1_floors_s1 | data | - | - | - | 1 | |
| w1_floors_s2 | data | - | - | - | 2 | |
| w1_mad_s0    | data | 0.01 | - | - | 0 | + electrostatics |
| w1_sf_s0     | data | 0.01 | 1.0 | - | 0 | + quantum structure factor |
| w1_force_s0  | data | 0.01 | 1.0 | 0.01 | 0 | + DFT force distillation |

**Primary question:** does `--floors data` stop the epoch-20 peak from being a
peak? Compare w1_base_* against w1_floors_* on validity and std_z past epoch 20.
Three seeds on the floors row because a single seed cannot answer it.

Each row differs from the one above by exactly one flag, so any effect is
attributable.

**Read results with `checkpoint_best.pt`, not the last checkpoint**, and never
report validity without volume per atom beside it.

# Wave 9 — launched 2026-09-25

4 runs, 500 epochs, ~87 s/epoch -> ~12 h. PIDs in `wave9_pids.csv` (TrainPID
is the real interpreter; LauncherPID is the Python Manager shim -- stop the
TrainPID with `Stop-Process -Id <id> -Force`).

Base config (every run), reconstructed from the w7/w8 logs because nothing
recorded their flags: `--batch_size 64 --floors bond --lambda_dist 0
--lambda_vol 0 --lambda_nn 1.0 --lambda_vpa_dist 0.2 --lambda_mode_seek 2.0`.
bond floors, batch 64 and lambda_dist 0 are confirmed by the logs; nn/vpa_dist
follow wave 6, and mode_seek 2.0 is read from the name "ms20" by analogy with
"ldist02" = 0.2. Every run now writes `args.json`, so this cannot happen again.

| folder | ema | seed | purpose |
|---|---|---|---|
| w9_ema_s0 | 0.999 | 0 | EMA generator weights for eval |
| w9_ema_s1 | 0.999 | 1 | seed variance |
| w9_ema_s2 | 0.999 | 2 | seed variance |
| w9_noema_s0 | - | 0 | control: same config and seed, no EMA |

Also new in this wave, for every run: a step whose gradient is non-finite is
skipped instead of written into the weights (the w8_ms20_s4 death).

**Question:** does EMA hold validity past the epoch-70 peak that every w7/w8 run
decayed from, without costing std_z or energy? Compare w9_ema_s0 against
w9_noema_s0 on the *last* checkpoint, not the best.

## Wave 9b — added the same day

| folder | change vs w9_ema | seed |
|---|---|---|
| w9_msinv_ema_s0 | `--mode_seek_form inv --lambda_mode_seek 0.002` | 0 |
| w9_msinv_ema_s1 | same | 1 |

Why: the corrected eval shows the w7/w8 late decay is NOT z collapse -- std_z
*rises* 0.03 -> 0.10 as validity falls. In w7_ms20_s0 the -ratio mode-seeking
term grows -0.03 -> -0.08 after epoch 100 while the NN-distance loss doubles
(0.21 -> 0.47) and validity drops 79% -> 38%. -ratio is unbounded; the original
MSGAN 1/(ratio+eps) fades once z is alive. Compare against w9_ema_s0/s1 on the
LAST checkpoint.

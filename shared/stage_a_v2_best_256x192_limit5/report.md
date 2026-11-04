# Sealed-set evaluation: policy:best.pt

5 sealed targets x 3 seeds (seeds 7, 8, 9) = 15 paintings. Canvas 256 x 192, 40 stroke slots. Scores are measured on the fully dried sheet.

Sealed manifest sha256 `74fe016242db949183d78d88f256c1ec11d312793c1126af8c919ee81c3473fd`. Mean and sample standard deviation over paintings.

| metric | mean | std |
|---|---|---|
| total (dried, lower is better) | 8.847 | 1.362 |
| lab_l2 (dried, lower is better) | 716.6 | 128.5 |
| ssim (dried, higher is better) | 0.5996 | 0.0372 |
| strokes used | 40.000 | 0.0000 |
| simulator steps per painting, all simulators | 112.0 | 0.0000 |
| strokes simulated per painting, all simulators | 40.000 | 0.0000 |
| simulator steps on the sheet itself | 662.0 | 52.610 |
| wall time per painting, s | 6.523 | 3.539 |
| total the painter predicted while painting | 8.817 | 1.085 |
| dried total minus predicted total | 0.0302 | 0.5210 |
| spread of stroke start points, x, px | 75.425 | 8.371 |
| spread of stroke start points, y, px | 35.991 | 8.540 |
| distinct strokes per painting | 19.000 | 2.952 |

Total wall time 21.4 s with 8 worker(s).

# Judge calibration

Judge `qwen/qwen3.5-397b-a17b`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 37 | 4 / 33 | 0.508 [0.242, 0.799] | 0.312 / 0.278 | 0.727 | 0.147 |
| rated by owner | 37 | 4 / 33 | 0.508 [0.242, 0.788] | 0.312 / 0.278 | 0.727 | 0.147 |
| deepseek/deepseek-v4-flash | 10 | 2 / 8 | 0.250 [0.000, 0.625] | 0.125 / 0.385 | 0.556 | 0.389 |
| google/gemini-3.1-flash-lite | 15 | 1 / 14 | 0.321 [-, -] | 0.000 / 0.232 | 0.769 | 0.037 |
| moonshotai/kimi-k2.6 | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| qwen/qwen3-coder-next | 7 | 0 / 7 | - [-, -] | - / 0.083 | 1.000 | 0.077 |
| z-ai/glm-5.3-flash | 5 | 1 / 4 | 0.875 [-, -] | 1.000 / 0.562 | 0.500 | 0.100 |

## Direct same-subject pairs, one keep against one nope

- 7 valid pairs of 40 asked; the judge picked the keep 0.929 of the time; order flipped the answer in 0.143.
- rated by mixed: 1 pairs, keep picked 1.000
- rated by reviewer: 6 pairs, keep picked 0.917

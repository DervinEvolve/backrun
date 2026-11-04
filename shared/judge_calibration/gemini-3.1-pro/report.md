# Judge calibration

Judge `google/gemini-3.1-pro-preview`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 99 | 15 / 84 | 0.623 [0.490, 0.746] | 0.572 / 0.387 | 0.622 | 0.110 |
| rated by owner | 99 | 15 / 84 | 0.623 [0.490, 0.760] | 0.572 / 0.387 | 0.622 | 0.110 |
| deepseek/deepseek-v4-flash | 25 | 3 / 22 | 0.462 [0.212, 0.728] | 0.333 / 0.424 | 0.571 | 0.094 |
| google/gemini-3.1-flash-lite | 33 | 3 / 30 | 0.806 [0.600, 0.950] | 0.833 / 0.375 | 0.655 | 0.111 |
| moonshotai/kimi-k2.6 | 4 | 2 / 2 | 0.000 [0.000, 0.000] | 0.000 / 0.875 | 0.000 | 0.167 |
| qwen/qwen3-coder-next | 22 | 3 / 19 | 0.947 [0.842, 1.000] | 0.556 / 0.079 | 0.909 | 0.091 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.511 [0.227, 0.773] | 0.854 / 0.788 | 0.357 | 0.150 |

## Direct same-subject pairs, one keep against one nope

- 26 valid pairs of 40 asked; the judge picked the keep 0.808 of the time; order flipped the answer in 0.231.
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 25 pairs, keep picked 0.800

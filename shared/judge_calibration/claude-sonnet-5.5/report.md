# Judge calibration

Judge `anthropic/claude-sonnet-5.5`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.625 [0.483, 0.779] | 0.550 / 0.367 | 0.653 | 0.099 |
| rated by owner | 108 | 15 / 93 | 0.625 [0.466, 0.769] | 0.550 / 0.367 | 0.653 | 0.099 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.417 [0.115, 0.885] | 0.333 / 0.404 | 0.600 | 0.115 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.599 [0.203, 0.906] | 0.500 / 0.352 | 0.677 | 0.113 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.167 [0.000, 0.667] | 0.250 / 0.694 | 0.250 | 0.111 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.865 [0.667, 1.000] | 0.583 / 0.135 | 0.864 | 0.026 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.614 [0.341, 0.864] | 0.875 / 0.682 | 0.462 | 0.133 |

## Direct same-subject pairs, one keep against one nope

- 37 valid pairs of 40 asked; the judge picked the keep 0.797 of the time; order flipped the answer in 0.135.
- rated by mixed: 6 pairs, keep picked 0.833
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 30 pairs, keep picked 0.783

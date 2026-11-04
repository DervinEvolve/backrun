# Judge calibration

Judge `z-ai/glm-4.6v`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| rated by owner | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| deepseek/deepseek-v4-flash | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| google/gemini-3.1-flash-lite | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| moonshotai/kimi-k2.6 | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| qwen/qwen3-coder-next | 0 | 0 / 0 | - [-, -] | - / - | - | - |
| z-ai/glm-5.3-flash | 0 | 0 / 0 | - [-, -] | - / - | - | - |

## Direct same-subject pairs, one keep against one nope

- 0 valid pairs of 40 asked; the judge picked the keep - of the time; order flipped the answer in -.

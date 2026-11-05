# Sample: warm_v1_6144_control

Model `/checkpoints/scribe/warm_v1/merged`, judge `anthropic/claude-opus-5.5`, pool manifest `7cfed21d2848`.

| metric | this batch | warm_v1_sample |
|---|---|---|
| completions | 288 | 288 |
| gate pass rate | 0.767 | 0.660 |
| cut off at the token limit | 0.073 | 0.219 |
| mean pairwise win share | 0.066 | 0.051 |
| mean reward | 0.135 | 0.111 |
| prompt groups with reward spread | 33 | 33 |
| prompt groups | 36 | 36 |
| mean within-group reward sd | 0.097 | 0.089 |
| rollouts excluded (judge failures) | 0 | 0 |
| judge calls | 884 | 760 |
| judge calls that failed | 0 | 0 |
| judge cost, $ | 2.063 | 1.762 |

**77% pass the gate and 92% of prompt groups have reward spread (mean within-group sd 0.097): GRPO has something to push on.**

Top gate rejections:
- 19: not valid JSON
- 8: only one flat colour; use at least two hues or two clearly d
- 3: point [100, 195] is outside the 256 x 192 paper
- 2: point [90, 195] is outside the 256 x 192 paper
- 2: point [130, 195] is outside the 256 x 192 paper
- 2: point [234, 194] is outside the 256 x 192 paper

One cut-off reply, `lemons__Qwen3-4B-Instruct-2507__02`, 10627 characters over 75 lines; its last 240 characters:

```
 "width": 8.0, "water": 0.5, "pigments": {"cadmium_yellow": 0.65}}
{"op": "stroke", "path": [[30.0, 180.0], [40.0, 176.0], [50.0, 180.0]], "width": 7.0, "water": 0.45, "pigments": {"cadmium_yellow": 0.7}}
{"op": "stroke", "path": [[90.0, 19
```

## Reward terms

```
288 rollouts used, 0 excluded for infrastructure failures
term         mean      sd  saturated
gate        0.767   0.423       100%
economy     0.740   0.439       100%
aesthetic   0.000   0.000       100%
pairwise    0.066   0.215        93%
correlation
              gate  economy  aesthet  pairwis
gate          1.00     0.93      n/a     0.17
economy       0.93     1.00      n/a     0.18
aesthetic      n/a      n/a      n/a      n/a
pairwise      0.17     0.18      n/a     1.00
WARNING: aesthetic barely varies (sd 0.000): it gives the trainer no gradient on this batch
WARNING: gate and economy are correlated at 0.93: the same signal counted twice
```

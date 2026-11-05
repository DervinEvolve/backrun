# Sample: grpo_v1_retry1_checkpoint_10 [vs_warm]

Model `/checkpoints/scribe/grpo_v1_retry1/checkpoint-10_merged`, judge `anthropic/claude-opus-5.5`, pool manifest `7cfed21d2848`. References: valid paintings of batch `warm_v1_6144_control`.

| metric | this batch | base |
|---|---|---|
| completions | 288 | 288 |
| gate pass rate | 0.764 | 0.146 |
| cut off at the token limit | 0.076 | 0.000 |
| mean pairwise win share | 0.386 | 0.000 |
| mean reward | 0.423 | 0.000 |
| prompt groups with reward spread | 36 | 0 |
| prompt groups | 36 | 36 |
| mean within-group reward sd | 0.354 | 0.000 |
| rollouts excluded (judge failures) | 0 | 42 |
| judge calls | 880 | 168 |
| judge calls that failed | 0 | - |
| judge cost, $ | 3.241 | 0 |

**76% pass the gate and 100% of prompt groups have reward spread (mean within-group sd 0.354): GRPO has something to push on.**

Top gate rejections:
- 24: not valid JSON
- 11: only one flat colour; use at least two hues or two clearly d
- 8: unknown pigment 'cadmium_yellow_1'
- 2: point [256, 194] is outside the 256 x 192 paper
- 2: point [260, 114] is outside the 256 x 192 paper
- 2: point [260, 146] is outside the 256 x 192 paper

One cut-off reply, `hibiscus__Qwen3-4B-Instruct-2507__04`, 9954 characters over 47 lines; its last 240 characters:

```
r": 0.16}}
{"op": "stroke", "path": [[207.0, 126.0], [214.0, 136.0], [222.0, 140.0], [214.0, 146.0], [207.0, 136.0]], "width": 4.0, "water": 0.35, "pigments": {"cadmium_yellow": 0.48, "quinacridone_rose": 0.36, "burnt_umber": 0.2}}
{"op": "
```

## Reward terms

```
288 rollouts used, 0 excluded for infrastructure failures
term         mean      sd  saturated
gate        0.764   0.425       100%
economy     0.736   0.441       100%
aesthetic   0.000   0.000       100%
pairwise    0.386   0.405        67%
correlation
              gate  economy  aesthet  pairwis
gate          1.00     0.93      n/a     0.53
economy       0.93     1.00      n/a     0.49
aesthetic      n/a      n/a      n/a      n/a
pairwise      0.53     0.49      n/a     1.00
WARNING: aesthetic barely varies (sd 0.000): it gives the trainer no gradient on this batch
WARNING: gate and economy are correlated at 0.93: the same signal counted twice
```

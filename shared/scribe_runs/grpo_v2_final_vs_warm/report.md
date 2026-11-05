# Sample: grpo_v2_final

Model `/checkpoints/scribe/grpo_v2/final_merged`, judge `anthropic/claude-opus-5.5`, pool manifest `7cfed21d2848`. References: valid paintings of batch `warm_v1_6144_control`.

| metric | this batch | warm_v1_6144_control |
|---|---|---|
| completions | 288 | 288 |
| gate pass rate | 0.910 | 0.767 |
| cut off at the token limit | 0.014 | 0.073 |
| mean pairwise win share | 0.527 | 0.066 |
| mean reward | 0.563 | 0.135 |
| prompt groups with reward spread | 36 | 33 |
| prompt groups | 36 | 36 |
| mean within-group reward sd | 0.324 | 0.097 |
| rollouts excluded (judge failures) | 0 | 0 |
| judge calls | 1048 | 884 |
| judge calls that failed | 0 | 0 |
| judge cost, $ | 3.824 | 2.063 |

**91% pass the gate and 100% of prompt groups have reward spread (mean within-group sd 0.324): GRPO has something to push on.**

Top gate rejections:
- 7: not valid JSON
- 4: only one flat colour; use at least two hues or two clearly d
- 2: 5 strokes; need 6 to 200
- 2: point [260, 88] is outside the 256 x 192 paper
- 1: point [128, 194] is outside the 256 x 192 paper
- 1: point [80, 194] is outside the 256 x 192 paper

One cut-off reply, `storm__Qwen3-4B-Instruct-2507__00`, 10647 characters over 122 lines; its last 240 characters:

```
th": 2.0, "water": 0.15, "pigments": {"burnt_umber": 0.05, "paynes_grey": 0.05}}
{"op": "dry", "steps": 5}
{"op": "stroke", "path": [[90.0, 190.0], [180.0, 185.0], [256.0, 190.0]], "width": 1.5, "water": 0.1, "pigments": {"burnt_umber": 0.0
```

## Reward terms

```
288 rollouts used, 0 excluded for infrastructure failures
term         mean      sd  saturated
gate        0.910   0.287       100%
economy     0.872   0.335       100%
aesthetic   0.000   0.000       100%
pairwise    0.527   0.405        61%
correlation
              gate  economy  aesthet  pairwis
gate          1.00     0.82      n/a     0.41
economy       0.82     1.00      n/a     0.34
aesthetic      n/a      n/a      n/a      n/a
pairwise      0.41     0.34      n/a     1.00
WARNING: aesthetic barely varies (sd 0.000): it gives the trainer no gradient on this batch
WARNING: gate and economy are correlated at 0.82: the same signal counted twice
```

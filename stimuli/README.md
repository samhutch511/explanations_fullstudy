# Scenario stimulus images

The images participants saw, one set per scenario. These are reference material — the analysis
notebook does not read them.

```
orig/   15 scenarios (indices 00–14) — original 6×9 map
hub/    45 scenarios (indices 00–44) — 9×10 hub-and-spoke map
```

## Index → scenario set

`hub/` indices follow the notebook's `stim_id` ordering, which concatenates the three sets in order
(`all_scenarios_raw = scenarios_l1_raw + scenarios_l2_raw + scenarios_l3_raw`):

| Indices | Set | Designed to discriminate |
|---|---|---|
| `00`–`14` | L1 | Full vs. No-inference |
| `15`–`29` | L2 | Full vs. No-future |
| `30`–`44` | L3 | Full vs. No-social |

`orig/` indices `00`–`14` correspond directly to `stim_data_orig` stimulus IDs 0–14.

## File types

| Pattern | Contents |
|---|---|
| `NN-init.png` | Starting configuration of the map |
| `NN-path.png` | Alice's actual path |
| `NN-path-alt-K.png` | Alternative paths shown for comparison |
| `NN-comic-K.png` | Sequential panels presenting the trial |

Panel counts vary by scenario, so `K` is not a fixed range.

## Provenance

Copied from the `pragmatic_explanation` repository — `stimuli_hub/` (hub) and
`explanation-gridworld-backup-main/stimuli/` (orig). That repository remains the source of truth;
regenerate there rather than editing these copies.

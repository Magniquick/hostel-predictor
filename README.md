# MIT Manipal Hostel Predictor — AY2026

Predicts hostel block allotment probabilities for MIT Manipal students based on CGPA and gender. Two views:

- **"Can I get this block?"** — P(you get block Y | you choose Y). Independent per-block. Pick a safe block or gamble.
- **"Where will I end up?"** — Allocation distribution assuming serial dictatorship preferences. Sums to 100%.

Live at **[magniquick.github.io/hostel-predictor](https://magniquick.github.io/hostel-predictor/)**

All computation runs client-side. No student data is stored or transmitted.

---

## How it works

### The allotment mechanism

MIT Manipal allocates hostels by **serial dictatorship**: students are ranked by effective CGPA (room-average CGPA for pairs), and each picks their preferred available block in rank order. Higher CGPA = pick first = better block.

### Available blocks (AY2026)

| Gender | Blocks (best → worst) | Total rooms |
|---|---|---|
| Male | XIV (544) → XV (485) → IX (62) → XX (140) → XXIII (20) → X (482) | 1,733 |
| Female | XXII (211) → XXI (95) → XIII (448) → VIII (139) | 893 |

XIX Block and XVIII BLOCK were removed from the AY2026 male selection — 453 rooms lost (~21% of male capacity).

### The model

**Core idea:** Block Y has rooms available until `cumulative_capacity(Y)` students ahead of you are served. So:

```
P(get block Y | you choose Y, CGPA) = min(1, sigmoid(CGPA, cutoff_Y, sigma_Y) / ceiling)
```

Where:
- `cutoff_Y` = drift-corrected CGPA at the cumulative capacity rank for block Y
- `sigma_Y` = transition width (uncertainty in exact cutoff position)
- `ceiling` = fraction of students who actually register (removes locals from denominator)

The **allocation distribution** (which block you'll end up in) is just the difference between consecutive eligibility probabilities — no separate model needed.

### Training pipeline

1. **Room averaging**: Effective CGPA computed for true doubles (shared rooms) and pseudo-single A/B pairs (same base room number)
2. **Outlier removal**: Per-block thresholds remove special quota students who bypass CGPA ranking
3. **Preference ordering**: Blocks ranked by median effective CGPA among allotted students
4. **Cumulative capacity cutoffs**: For each block, the effective CGPA at its cumulative rank position becomes the base cutoff
5. **Drift correction**: Adjusts cutoffs for population change between academic years (participation-weighted)
6. **Ceiling normalization**: Observed high-CGPA registration rate (~93% male, ~88% female) estimates the fraction of students in the market; dividing by it gives P(hostel | interested)

### What the output means

At CGPA 10, every block shows 100% — you pick first, guaranteed.

At CGPA 7.5 male (eligibility view):
- XIV Block: 14.9% — tough, need to be top ~544 students
- X Block: 64.1% — safe bet, top ~1,733 students
- The gap between these is your risk/reward tradeoff

At CGPA 7.5 male (allocation view):
- XIV Block: 14.9% — if you'd naturally rank into the top 544
- X Block: 28.3% — the safety net catching ranks 1,251–1,733
- No hostel: 35.9% — outside the top 1,733

### Limitations

- **Limited training data**. Multi-year data would stabilize cutoff estimates.
- **Special quota** students (sports, staff) are handled via manual outlier thresholds, not explicit flags.
- **Room pairing**: Model uses individual CGPA as input. If you know your roommate, input the average instead.
- **4 unknown female blocks** (Nehru Ladies, New International A, New Sarojini, Sharada) have no historical data.
- **Population drift**: Incoming batch CGPA distribution is assumed to resemble prior years. If it shifts significantly, cutoffs will be off.

### Trained parameters

All model parameters are in [`model_params.json`](model_params.json). The sigmoid function:

```
sigmoid(CGPA, c, sigma) = 1 / (1 + exp(-(CGPA - c) / sigma))
P(block) = min(1, sigmoid(CGPA, c_new, sigma) / ceiling)
```

No student data, CGPAs, or registration IDs are included — only the trained cutoffs, sigmas, capacities, and ceilings.

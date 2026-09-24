# Historical ablation models

These modules reproduce the model definitions in the historical `exp53`, `exp54`, `exp56`, `exp57`, and `exp58` experiment directories. Each `experimentNN.py` module exports `create_model(input_channels)`, accepts `(batch, channels, time)` input, and returns the keys consumed by the SHINE training loop. Import example:

```python
from ablations.experiment53 import create_model
model = create_model(input_channels=64)
```

## Interpretation

`exp53` and `exp54` both include envelope-guided decoding (EGD): the predicted envelope guides Mel reconstruction through a learned projection and gate. They are not pure single-component ablations relative to `exp55`. `exp53` additionally removes pooled context; `exp54` uses pooled context with fixed 0.5 local/context output averaging. The EGD component is shared by both.

| Variant | Historical change | Runner/output notes |
|---|---|---|
| exp53 | No pooled context; EGD enabled | Returns `gate`; one local auxiliary prediction |
| exp54 | Fixed local/context averaging; EGD enabled | Returns `gate`; two auxiliary predictions |
| exp56 | No multi-level hierarchy; last local block only | No `gate`; two auxiliary predictions |
| exp57 | Single 1x1 spatial projection | No `gate`; two auxiliary predictions |
| exp58 | Ungated depthwise temporal blocks | No `gate`; two auxiliary predictions |

All variants use 96 hidden channels, eight temporal blocks, and default full training-window length 1920 samples. The pooled context implementation requires a sequence length for which `avg_pool1d(..., kernel_size=8, stride=8, ceil_mode=True)` is valid. The runner selects a model through `--experiment`; these modules are importable as `ablations.experimentNN` when `shine_recon` is on `PYTHONPATH`.

No training is started by these modules. Parameter counts for 204 input channels (the MEG setting) are reported in `manifest.json`.

# Historical comparison models

These modules preserve the implementations and default initializations used in historical experiments 40–43 and 45–49. They are protocol adaptations used for this comparison, not official upstream implementations of the cited papers. Each module exports `create_model(input_channels)` and returns the runner's `{"envelope": [B, 1, T], "mel": [B, 10, T]}` interface.

All models depend only on PyTorch. Parameter counts below include trainable and non-trainable parameters returned by `model.parameters()` with the original default widths; counts are shown for EEG (64 input channels) and MEG (204 input channels).

| Module | Historical model name | Parameters (64 / 204 channels) |
| --- | --- | ---: |
| `experiment40.py` | BrainMagick-MEG adaptation (`MEGReadyBrainMagic`) | 63,755 / 90,635 |
| `experiment41.py` | VLAAI-MEG adaptation (`MEGReadyVLAAI`) | 1,600,971 / 1,627,851 |
| `experiment42.py` | HappyQuokka-MEG adaptation (`MEGReadyHappyQuokka`) | 1,119,691 / 1,146,571 |
| `experiment43.py` | ConvConcatNet adaptation | 291,275 / 300,235 |
| `experiment45.py` | AWaveNet adaptation (`AWaveNet`) | 1,197,899 / 1,206,859 |
| `experiment46.py` | CNN-LSTM internal comparison | 7,336 / 18,536 |
| `experiment47.py` | DilatedConv internal comparison | 41,931 / 50,891 |
| `experiment48.py` | SSM2Mel protocol adaptation (`SSM2MelAdapted`) | 528,875 / 542,315 |
| `experiment49.py` | DMF2Mel protocol adaptation (`DMF2MelAdapted`) | 409,259 / 422,699 |

Experiment 45 implements AWaveNet: its module, class, and registry entry must not be named CAT-WaveNet. Its historical implementation has no cross-attention component. Experiment 48 uses an S4-like convolutional surrogate and self-conditioned channel-strength modulation instead of official SSM2Mel components and subject-ID input. Experiment 49 uses a GRU-based bidirectional convolution-state substitute and radial spline gate; it is likewise an adaptation, not an upstream reproduction.

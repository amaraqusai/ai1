# Signal Generation (FR-1, FR-2)

## Inputs
- Frequencies: [5, 15, 30, 50] Hz
- Sampling rate: 200 Hz
- Duration: 10s per frequency
- Amplitude: independently configurable [0, 2.0]
- Phase offset: [0, 2π)

## Outputs
- Datasets saved in `data/` as `.npz`.
- Features containing noisy samples and frequency labels.

## Test Scenarios (SG-T*)
- SG-T1: Length of dataset correctly generated based on 10s duration
- SG-T2: Noise scaling works correctly
- SG-T3: Normalization produces mean 0, variance 1
- SG-T4: Split shapes match (70%, 15%, 15%)
- SG-T5: Sliding window creates sequences of length 10
- SG-T6: Phase randomization affects signal shape but not frequency
- SG-T7: Extracted labels match their target frequency

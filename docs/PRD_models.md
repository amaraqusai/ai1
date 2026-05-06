# Models (FR-3..FR-5)

## 1. MLP
- **Inputs**: 14 (10 noisy samples + 4 labels)
- **Architecture**: 14 -> 64 -> 128 -> 64 -> 1
- **Activations**: Tanh
- **Rationale**: Understand how standard fully-connected networks without memory fail or succeed in time series prediction compared to networks with memory.

## 2. RNN
- **Inputs**: Sequence of 10 timesteps, each with 5 features (1 noisy sample + 4 labels)
- **Architecture**: 2 hidden layers, 64 hidden size -> linear -> 1
- **Activations**: Default (Tanh)
- **Rationale**: Short-term dependency capturing across 10 steps. 

## 3. LSTM
- **Inputs**: Sequence of 10 timesteps, each with 5 features
- **Architecture**: 2 hidden layers, 64 hidden size -> linear -> 1
- **Activations**: Default LSTM gating
- **Rationale**: Long-term dependency and robust learning across steps.

## Parameter Counts
- Track parameter counts for evaluating complexity.

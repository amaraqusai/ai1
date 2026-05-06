# Training (FR-6)

## Parameters
- Loss: MSE
- Optimizer: Adam
- Default LR: 0.001
- Max Epochs: 300
- Early stopping patience: 20 steps
- Batch Size: 64

## Metrics
- Checkpoint at best validation MSE
- Log training MSE vs Validation MSE

## Test Scenarios (TR-T*)
- TR-T1: Loss decreases over first 5 epochs
- TR-T2: Model checkpoints saved properly
- TR-T3: Early stopping triggers when validation loss stalls
- TR-T4: Tensor type casting matches devices correctly (CPU/GPU)
- TR-T5: Evaluation skips training steps (model.eval(), torch.no_grad())
- TR-T6: Deterministic execution correctly preserves identical runs
- TR-T7: OOM errors fallback correctly

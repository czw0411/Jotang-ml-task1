| name | params | epochs | final_train_acc | final_val_acc | final_val_loss | test_acc | generalization_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small data (n=30) | 132866 | 2000 | 1.0000 | 0.9400 | 0.9056 | 0.9700 | 0.0300 |
| 20% label noise (n=30) | 132866 | 2000 | 1.0000 | 0.7133 | 16.7390 | 0.6967 | 0.3033 |
| label noise + weight decay | 132866 | 2000 | 0.8333 | 0.8300 | 0.4416 | 0.8367 | -0.0033 |

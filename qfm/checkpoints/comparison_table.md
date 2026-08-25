| Method | Encoding | Accuracy | F1 | Kernel train (s) | SVM train (s) | Total (s) |
|---|---|---|---|---|---|---|
| Classical_SVM | - | 0.8828 | 0.8761 | 0.00 | 0.04 | 0.05 |
| Classical_SVM | - | 0.8647 | 0.8541 | 0.00 | 0.04 | 0.05 |
| QK_trainable | x (no-ent) | 0.8340 | 0.8268 | 38.20 | 13.25 | 56.77 |
| QK_trainable | xyz (no-ent) | 0.8131 | 0.8091 | 41.00 | 13.31 | 59.94 |
| PQK_trainable | xyz (no-ent) | 0.8619 | 0.8551 | 1831.30 | 604.77 | 2679.50 |
| PQK_trainable | xyz (no-ent) | 0.8382 | 0.8362 | 1857.78 | 612.23 | 2716.43 |
| PQK_trainable | xyz (no-ent) | 0.8605 | 0.8555 | 1838.95 | 615.74 | 2705.71 |
| PQK_trainable | xyz (no-ent) | 0.8563 | 0.8509 | 1875.28 | 609.00 | 2732.44 |
| PQK_trainable | xyz (ent) | 0.8117 | 0.8153 | 1908.00 | 635.69 | 2807.25 |
| QK_real_hardware | xyz (no-ent) | 0.3333 | 0.0000 | 0.89 | 0.00 | 2.06 |

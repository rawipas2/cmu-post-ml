# Factorial Workflow

Run the four Table VI cells as separate experiments:

```powershell
python train.py --preprocessing-mode shared --loss-mode bce --svm-policy freeze_v1_2
python train.py --preprocessing-mode shared --loss-mode focal --svm-policy freeze_v1_2
python train.py --preprocessing-mode model_specific --loss-mode bce --svm-policy freeze_v1_2
python train.py --preprocessing-mode model_specific --loss-mode focal --svm-policy freeze_v1_2
```

Generate Table VI outputs after all four runs complete:

```powershell
python generate_table_vi.py
```

Recommended sanity check before long runs:

```powershell
python sanity_check_factorial.py
```

Outputs:

- `versions/factorial_shared-bce*/`
- `versions/factorial_shared-focal*/`
- `versions/factorial_model_specific-bce*/`
- `versions/factorial_model_specific-focal*/`
- `results/Table_VI.csv`
- `results/Table_VI.md`
- `results/Table_VI_summary.json`

Notes:

- `freeze_v1_2` keeps the model-specific SVM on the stable shared TF-IDF path.
- The BCE vs Focal factor applies to differentiable models only.
- Table VI cells are formatted as `accuracy / weighted F1`.

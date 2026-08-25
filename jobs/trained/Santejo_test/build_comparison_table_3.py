import os
import json
import glob

checkpoint_dir = os.path.join('qfm', 'checkpoints')
result_files = glob.glob(os.path.join(checkpoint_dir, '*_results.json'))

if not result_files:
    print(f'No results found in {checkpoint_dir}. '
          f'Run run_SVM_checkpointed.py, run_QK_checkpointed.py, and '
          f'run_PQK_checkpointed.py first.')
    raise SystemExit(0)

rows = []
for path in sorted(result_files):
    with open(path) as f:
        rows.append(json.load(f))

# order: classical SVM, then QK, then PQK - matches the paper's Table 1 layout
method_order = {'Classical_SVM': 0, 'QK_trainable': 1, 'PQK_trainable': 2}
rows.sort(key=lambda r: method_order.get(r.get('method', ''), 99))

# resultados de hardware real usam 'hardware_seconds' em vez de
# 'kernel_training_seconds' (nome de campo diferente, mesmo significado) -
# normaliza aqui para a tabela ficar coerente entre todos os métodos
for r in rows:
    if 'kernel_training_seconds' not in r and 'hardware_seconds' in r:
        r['kernel_training_seconds'] = r['hardware_seconds']

print(f'{"Method":<16}{"Encoding":<12}{"Accuracy":<12}{"F1":<12}{"Kernel train (s)":<18}{"SVM train (s)":<16}{"Total (s)":<12}')
print('-' * 98)
for r in rows:
    enc_label = r.get('encoding', '-')
    if r.get('full_ent') is not None:
        enc_label += f'({"ent" if r["full_ent"] else "no-ent"})'
    print(f'{r.get("method", "?"):<16}'
          f'{enc_label:<12}'
          f'{r.get("accuracy", float("nan")):<12.4f}'
          f'{r.get("f1", float("nan")):<12.4f}'
          f'{r.get("kernel_training_seconds", 0.0):<18.2f}'
          f'{r.get("svm_training_seconds", 0.0):<16.2f}'
          f'{r.get("total_seconds", 0.0):<12.2f}')

#also write a markdown version
md_path = os.path.join(checkpoint_dir, 'comparison_table.md')
with open(md_path, 'w') as f:
    f.write('| Method | Encoding | Accuracy | F1 | Kernel train (s) | SVM train (s) | Total (s) |\n')
    f.write('|---|---|---|---|---|---|---|\n')
    for r in rows:
        enc_label = r.get('encoding', '-')
        if r.get('full_ent') is not None:
            enc_label += f' ({"ent" if r["full_ent"] else "no-ent"})'
        f.write(f'| {r.get("method", "?")} | {enc_label} | {r.get("accuracy", 0):.4f} | {r.get("f1", 0):.4f} | '
                f'{r.get("kernel_training_seconds", 0.0):.2f} | {r.get("svm_training_seconds", 0.0):.2f} | '
                f'{r.get("total_seconds", 0.0):.2f} |\n')

print(f'\nMarkdown table saved to: {md_path}')

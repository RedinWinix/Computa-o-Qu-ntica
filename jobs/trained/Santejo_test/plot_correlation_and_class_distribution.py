"""
Gera os dois gráficos exploratórios apresentados na Figura 4 do artigo
"Assessing Projected Quantum Kernels for the Classification of IoT Data":
  (a) matriz de correlação entre as features do dataset
  (b) distribuição das classes (occupancy: -1 / +1)

Uso:
    python plot_correlation_and_class_distribution.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- CONFIGURAÇÃO ----
DATA_FILE = 'data/env.sel3.csv'  # dataset bruto (não escalado) - a correlação
                                   # não muda com StandardScaler, mas os valores
                                   # brutos mantêm a escala original interpretável
OUTPUT_DIR = 'figures'
FEATURE_COLUMNS = ['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']
LABEL_COLUMN = 'occupancy'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- CARREGAMENTO DOS DADOS ----
df = pd.read_csv(DATA_FILE)
print(f'Dataset carregado: {df.shape[0]} observações, {df.shape[1]} colunas')
print(f'Distribuição de classes:\n{df[LABEL_COLUMN].value_counts()}')

# ==========================================================================
# FIGURA COMBINADA (2 painéis, lado a lado - igual ao layout da Figura 4)
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- (a) GRÁFICO DE CORRELAÇÃO ----
corr_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
corr_matrix = df[corr_columns].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',      # escala divergente: azul (negativa) - vermelho (positiva)
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Correlação de Pearson'},
    ax=axes[0],
)
axes[0].set_title('(a) Matriz de Correlação', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# ---- (b) DISTRIBUIÇÃO DAS CLASSES ----
class_counts = df[LABEL_COLUMN].value_counts().sort_index()
labels = ['Não ocupado (-1)', 'Ocupado (+1)']
colors = ['#4C72B0', '#DD8452']

bars = axes[1].bar(labels, class_counts.values, color=colors, edgecolor='black', linewidth=0.8)
axes[1].set_title('(b) Distribuição das Classes', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Número de observações')

# rótulos com contagem e percentual em cima de cada barra
total = class_counts.sum()
for bar, count in zip(bars, class_counts.values):
    pct = 100 * count / total
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.01,
        f'{count}\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=10,
    )

axes[1].set_ylim(0, max(class_counts.values) * 1.15)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'correlation_and_class_distribution.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'Figura combinada salva em: {output_path}')

# ==========================================================================
# TAMBÉM SALVA CADA PAINEL SEPARADAMENTE (caso precise inserir individualmente
# no relatório, em vez da figura combinada)
# ==========================================================================

# (a) só a correlação
fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
    vmin=-1, vmax=1, square=True, linewidths=0.5,
    cbar_kws={'label': 'Correlação de Pearson'}, ax=ax_corr,
)
ax_corr.set_title('Matriz de Correlação', fontsize=13, fontweight='bold')
ax_corr.tick_params(axis='x', rotation=45)
plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
plt.savefig(corr_path, dpi=200, bbox_inches='tight')
print(f'Matriz de correlação salva em: {corr_path}')

# (b) só a distribuição de classes
fig_dist, ax_dist = plt.subplots(figsize=(5.5, 5.5))
bars = ax_dist.bar(labels, class_counts.values, color=colors, edgecolor='black', linewidth=0.8)
ax_dist.set_title('Distribuição das Classes', fontsize=13, fontweight='bold')
ax_dist.set_ylabel('Número de observações')
for bar, count in zip(bars, class_counts.values):
    pct = 100 * count / total
    ax_dist.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.01,
        f'{count}\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=10,
    )
ax_dist.set_ylim(0, max(class_counts.values) * 1.15)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
plt.savefig(dist_path, dpi=200, bbox_inches='tight')
print(f'Distribuição de classes salva em: {dist_path}')

plt.close('all')
print('\nConcluído.')

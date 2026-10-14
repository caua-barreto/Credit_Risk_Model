"""
plots.py — Modulo de visualizacao grafica e diagnostico de variaveis,
distribuicoes, matrizes de correlacao, curvas de performance e drift.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.drift import calcular_psi

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'


def _salvar_ou_exibir_grafico(filepath: Optional[Union[str, Path]] = None) -> None:
    """
    Utilitario para salvar a figura em reports/figures/ ou exibir via plt.show().
    """
    plt.tight_layout()
    if filepath is not None:
        save_path = Path(filepath)
        if not save_path.is_absolute():
            save_path = FIGURES_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def analisar_valores_nulos(
    df: pd.DataFrame,
    plotar: bool = True,
    figsize: tuple = (10, 6),
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Analisa valores nulos do DataFrame, retornando tabela com quantidade
    e percentual por coluna. Opcionalmente plota gráfico de barras.

    Parameters
    ----------
    df : pd.DataFrame
    plotar : bool
    figsize : tuple

    Returns
    -------
    pd.DataFrame
        Colunas com nulos, ordenadas da maior para menor quantidade.
    """
    total_nulos = df.isnull().sum()
    percentual_nulos = (total_nulos / len(df)) * 100

    df_nulos = pd.DataFrame({
        'Total Nulos': total_nulos,
        'Percentual (%)': percentual_nulos,
        'Tipo de Dado': df.dtypes
    })
    df_nulos = df_nulos[df_nulos['Total Nulos'] > 0].sort_values(
        by='Total Nulos', ascending=False
    )

    if df_nulos.empty:
        print("Nenhum valor nulo encontrado no DataFrame.")
        return df_nulos

    if plotar:
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            x='Percentual (%)', y=df_nulos.index, data=df_nulos,
            color='#E66C37', edgecolor='black', alpha=0.8
        )
        for p in ax.patches:
            width = p.get_width()
            plt.text(
                width + 0.5, p.get_y() + p.get_height() / 2.0 + 0.1,
                f'{width:.1f}%', ha="left", va="center", fontsize=10, color='black'
            )
        plt.title('Percentual de Valores Nulos por Variavel', fontsize=14, fontweight='bold')
        plt.xlabel('Percentual (%)', fontsize=12)
        plt.ylabel('Variaveis', fontsize=12)
        plt.xlim(0, max(df_nulos['Percentual (%)']) + 10)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    return df_nulos


# =============================================================================
# 5. VISUALIZAÇÕES CORPORATIVAS
# =============================================================================


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'target', filepath: Optional[Union[str, Path]] = None) -> Tuple[pd.Series, pd.Series, float]:
    """
    Plota a distribuição da variável alvo com estilo corporativo.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    Tuple[pd.Series, pd.Series, float]
        (contagens, percentuais, ratio adimplente:default)
    """
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    ratio = target_counts[0] / target_counts[1]

    print(f"Distribuicao do target:\n")
    print(f"  target=0 (adimplente): {target_counts[0]:>10,}  ({target_pct[0]:.2f}%)")
    print(f"  target=1 (default):    {target_counts[1]:>10,}  ({target_pct[1]:.2f}%)")
    print(f"\n  Ratio (0:1):           {ratio:.1f}:1")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    cor_adimplente = "#AFAFAF"
    cor_default = '#e60000'

    bars = ax.bar(
        [0, 1], target_counts.values,
        color=[cor_adimplente, cor_default],
        edgecolor=['#cccccc', '#cc0000'],
        linewidth=1.5, width=0.55
    )

    max_y = target_counts.max()
    for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
        texto = f'{count:,.0f}\n({pct:.2f}%)'.replace(',', '.')
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_y * 0.03),
            texto, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333'
        )

    ax.set_title('Distribuicao de Clientes: Bom Pagador vs Devedor',
                 fontsize=16, fontweight='bold', color='#333333', pad=25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Bom Pagador\n(0)', 'Devedor\n(1)'],
                       fontsize=13, fontweight='bold', color='#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.get_yaxis().set_visible(False)
    ax.set_ylim(0, max_y * 1.15)
    _salvar_ou_exibir_grafico(filepath)

    return target_counts, target_pct, ratio


def plot_sazonalidade(
    df: pd.DataFrame,
    target_col: str = 'target',
    month_col: str = 'MONTH',
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Plota volume mensal (barras) vs taxa de default (linha) com eixo duplo.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    month_col : str

    Returns
    -------
    pd.DataFrame
        Tabela com Default Rate (%) e Volume por mês.
    """
    periodo_min = df['date_decision'].min()
    periodo_max = df['date_decision'].max()
    n_meses = df[month_col].nunique()

    print(f"Periodo: {periodo_min.strftime('%Y-%m-%d')} -> {periodo_max.strftime('%Y-%m-%d')}")
    print(f"Meses unicos: {n_meses}")
    print(f"Volume medio/mes: {len(df) / n_meses:,.0f}")

    taxa_mensal = df.groupby(month_col)[target_col].agg(
        Volume='count', Default_Rate='mean'
    ).copy()
    taxa_mensal['Default_Rate'] *= 100
    meses_str = taxa_mensal.index.astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')

    cor_volume = '#f0f0f0'
    cor_borda = '#cccccc'
    ax1.bar(meses_str, taxa_mensal['Volume'], color=cor_volume,
            edgecolor=cor_borda, width=0.6, label='Volume de Pedidos')
    ax1.set_ylabel('Volume de Pedidos', fontsize=12, fontweight='bold', color='#666666')
    ax1.tick_params(axis='y', colors='#666666')
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.'))
    )

    ax2 = ax1.twinx()
    cor_taxa = '#e60000'
    ax2.plot(meses_str, taxa_mensal['Default_Rate'], color=cor_taxa,
             marker='o', linewidth=3.5, markersize=8, label='Taxa de Default (%)')
    ax2.set_ylabel('Taxa de Default (%)', fontsize=12, fontweight='bold', color=cor_taxa)
    ax2.tick_params(axis='y', colors=cor_taxa)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
    ax2.set_ylim(0, taxa_mensal['Default_Rate'].max() * 1.3)

    for i, txt in enumerate(taxa_mensal['Default_Rate']):
        ax2.annotate(f'{txt:.2f}%', (meses_str[i], txt),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=10, fontweight='bold', color=cor_taxa)

    plt.title('Sazonalidade: Taxa de Default vs Volume de Pedidos',
              fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax1.set_xlabel('Mes de Decisao (YYYYMM)', fontsize=12, fontweight='bold', color='#555555')
    ax1.set_xticks(range(len(meses_str)))
    ax1.set_xticklabels(meses_str, rotation=45, ha='right', fontsize=11)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left',
               frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=11)

    _salvar_ou_exibir_grafico(filepath)

    taxa_display = taxa_mensal.copy()
    taxa_display.index.name = 'Mes'
    taxa_display = taxa_display.rename(columns={'Default_Rate': 'Default Rate (%)'})
    taxa_display = taxa_display[['Default Rate (%)', 'Volume']]

    return taxa_display


def plot_matriz_correlacao(df, titulo='Matriz de Correlação', triangulo=True,
                           cmap='coolwarm', figsize=(18, 14), annot_size=9, filepath: Optional[Union[str, Path]] = None):
    """
    Plota heatmap da matriz de correlação de variáveis numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas numéricas.
    titulo : str, default='Matriz de Correlação'
        Título do gráfico.
    triangulo : bool or str, default=True
        - True ou 'inferior': exibe apenas o triângulo inferior.
        - 'superior': exibe apenas o triângulo superior.
        - False ou 'completa': exibe a matriz completa.
    cmap : str, default='coolwarm'
        Paleta de cores do heatmap. Ex: 'RdBu_r', 'viridis', 'magma'.
    figsize : tuple, default=(18, 14)
        Dimensões da figura.
    annot_size : int, default=9
        Tamanho da fonte das anotações.

    Retorna
    -------
    pd.DataFrame
        Matriz de correlação calculada.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    mascara = None
    if triangulo in (True, 'inferior'):
        mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangulo == 'superior':
        mascara = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=.5, cbar_kws={'shrink': .75},
        mask=mascara, annot_kws={'size': annot_size}
    )
    plt.title(titulo, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    _salvar_ou_exibir_grafico(filepath)

    return corr


def plot_feature_importance(
    modelo,
    feature_names: list,
    top_n: int = 20,
    titulo: str = None,
    figsize: tuple = (14, 10),
    cor: str = 'steelblue',
    xlabel: str = None,
    annot: bool = True,
    fmt: str = '.3f',
    alpha: float = 0.85,
    filepath: Optional[Union[str, Path]] = None,
):
    """
    Plota a importância das features de um modelo treinado (coeficientes
    para modelos lineares ou feature_importances_ para árvores).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn já treinado.
    feature_names : list
        Lista com os nomes das features.
    top_n : int, default=20
        Número de features mais importantes a exibir.
    titulo : str, optional
        Título do gráfico.
    figsize : tuple, default=(14, 10)
        Dimensões da figura.
    cor : str, default='steelblue'
        Cor das barras.
    xlabel : str, optional
        Rótulo do eixo X. Se None, usa o tipo detectado.
    annot : bool, default=True
        Se True, exibe o valor numérico ao lado de cada barra.
    fmt : str, default='.3f'
        Formatação dos valores anotados.
    alpha : float, default=0.85
        Transparência das barras.
    """
    clf = modelo
    if hasattr(modelo, 'named_steps') and 'clf' in modelo.named_steps:
        clf = modelo.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        importancias = clf.feature_importances_
        tipo = 'Feature Importance'
    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importancias = np.abs(coefs)
        tipo = '|Coeficientes|'
    else:
        raise ValueError("O modelo não possui 'feature_importances_' nem 'coef_'.")

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importância': importancias
    }).sort_values('Importância', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        df_imp['Feature'], df_imp['Importância'],
        color=cor, edgecolor='black', alpha=alpha,
    )

    if annot:
        for bar, val in zip(bars, df_imp['Importância']):
            ax.text(
                bar.get_width() + df_imp['Importância'].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:{fmt}}',
                va='center', fontsize=9, fontweight='bold',
            )

    ax.set_xlabel(xlabel or tipo, fontsize=12)
    ax.set_title(
        titulo or f'Top {top_n} {tipo}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    _salvar_ou_exibir_grafico(filepath)

    # Organiza do maior para o menor e remove o índice desordenado
    df_retorno = df_imp.sort_values('Importância', ascending=False).reset_index(drop=True)
    
    # Retorno estruturado sem dependencia de IPython display
    return df_retorno


def plot_curvas_roc_pr_sobrepostas(y_test, prob_test, y_oot, prob_oot, figsize=(14, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Plota as curvas ROC e Precision-Recall sobrepondo as bases de Teste (In-Time) e OOT.
    Apresenta paleta corporativa (Navy Blue e Crimson Red).
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    fpr_test, tpr_test, _ = roc_curve(y_test, prob_test)
    auc_test = roc_auc_score(y_test, prob_test)
    fpr_oot, tpr_oot, _ = roc_curve(y_oot, prob_oot)
    auc_oot = roc_auc_score(y_oot, prob_oot)

    prec_test, rec_test, _ = precision_recall_curve(y_test, prob_test)
    pr_auc_test = average_precision_score(y_test, prob_test)
    prec_oot, rec_oot, _ = precision_recall_curve(y_oot, prob_oot)
    pr_auc_oot = average_precision_score(y_oot, prob_oot)

    cor_teste = '#1B365D'  # Navy Blue
    cor_oot = '#8B0000'    # Deep Crimson / Vinho

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Curva ROC
    ax1.plot(fpr_test, tpr_test, color=cor_teste, linewidth=2.2,
             label=f'Teste (In-Time) | ROC-AUC = {auc_test:.4f}')
    ax1.plot(fpr_oot, tpr_oot, color=cor_oot, linewidth=2.2, linestyle='--',
             label=f'OOT (Out-of-Time) | ROC-AUC = {auc_oot:.4f}')
    ax1.plot([0, 1], [0, 1], color='#7f8c8d', linestyle=':', linewidth=1.2, label='Aleatório (Baseline = 0.50)')
    ax1.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Curva ROC — Teste vs. OOT', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.grid(True, linestyle='--', alpha=0.35)

    # Curva Precision-Recall
    ax2.plot(rec_test, prec_test, color=cor_teste, linewidth=2.2,
             label=f'Teste (In-Time) | PR-AUC = {pr_auc_test:.4f}')
    ax2.plot(rec_oot, prec_oot, color=cor_oot, linewidth=2.2, linestyle='--',
             label=f'OOT (Out-of-Time) | PR-AUC = {pr_auc_oot:.4f}')
    base_test = y_test.mean()
    base_oot = y_oot.mean()
    ax2.axhline(base_test, color=cor_teste, linestyle=':', alpha=0.6, label=f'Prevalência Teste ({base_test:.2%})')
    ax2.axhline(base_oot, color=cor_oot, linestyle=':', alpha=0.6, label=f'Prevalência OOT ({base_oot:.2%})')
    ax2.set_xlabel('Recall (Cobertura de Inadimplentes)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precisão (Taxa de Acerto no Risco)', fontsize=11, fontweight='bold')
    ax2.set_title('Curva Precision-Recall — Teste vs. OOT', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.grid(True, linestyle='--', alpha=0.35)

    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)


def plot_distribuicao_kde_scores(y_true, prob, nome_base='OOT', figsize=(10, 5), filepath: Optional[Union[str, Path]] = None):
    """
    Plota a distribuição de densidade (KDE) dos scores preditos separando
    a classe Adimplente (Bons) da Inadimplente (Maus).
    """
    df_plot = pd.DataFrame({'score': prob, 'target': y_true.values if hasattr(y_true, 'values') else y_true})
    
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df_plot[df_plot['target'] == 0], x='score', color='#1B365D',
                fill=True, alpha=0.3, linewidth=2.0, label='Bons Pagadores (Target = 0)')
    sns.kdeplot(data=df_plot[df_plot['target'] == 1], x='score', color='#8B0000',
                fill=True, alpha=0.35, linewidth=2.0, label='Inadimplentes (Target = 1)')

    plt.title(f'Distribuição de Densidade dos Scores (KDE) — Base {nome_base}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Probabilidade Prevista de Inadimplência (Score)', fontsize=11, fontweight='bold')
    plt.ylabel('Densidade', fontsize=11, fontweight='bold')
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)


def plot_target_drift_mensal(df_train, df_test, df_oot, col_mes='MONTH', col_target='target', figsize=(12, 5), filepath: Optional[Union[str, Path]] = None):
    """
    Gera gráfico temporal corporativo da taxa de default safra a safra (Target Drift)
    cobrindo Treino, Teste e OOT.
    """
    dados = []
    for nome_base, base in [('Treino', df_train), ('Teste', df_test), ('OOT', df_oot)]:
        agrupado = base.groupby(col_mes)[col_target].agg(['count', 'mean']).reset_index()
        agrupado['Base'] = nome_base
        dados.append(agrupado)

    df_consolidado = pd.concat(dados, ignore_index=True)
    df_consolidado['Taxa_Default_Pct'] = df_consolidado['mean'] * 100
    df_consolidado = df_consolidado.sort_values(col_mes)

    plt.figure(figsize=figsize)
    paleta = {'Treino': '#1B365D', 'Teste': '#4682B4', 'OOT': '#8B0000'}

    for base, cor in paleta.items():
        subset = df_consolidado[df_consolidado['Base'] == base]
        if not subset.empty:
            plt.plot(subset[col_mes].astype(str), subset['Taxa_Default_Pct'],
                     marker='o', linewidth=2.2, color=cor, label=f'Taxa de Default — {base}')

    media_total = (df_train[col_target].sum() + df_test[col_target].sum() + df_oot[col_target].sum()) / \
                  (len(df_train) + len(df_test) + len(df_oot)) * 100
    plt.axhline(media_total, color='#7f8c8d', linestyle=':', linewidth=1.5,
                label=f'Média Global ({media_total:.2f}%)')

    plt.title('Target Drift — Evolução da Taxa de Default por Safra (Mês de Decisão)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Safra de Decisão (MONTH)', fontsize=11, fontweight='bold')
    plt.ylabel('Taxa de Default (%)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)

    return df_consolidado


def plot_comparativo_psi_features(X_test, X_oot, features_list, bins=10, figsize=(15, 4), filepath: Optional[Union[str, Path]] = None):
    """
    Plota histogramas de densidade comparativos entre Teste e OOT para uma lista de features
    e exibe o valor de PSI de cada uma no título.
    """
    n_feats = len(features_list)
    fig, axes = plt.subplots(1, n_feats, figsize=(figsize[0], figsize[1]))
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_list):
        s_test = X_test[feat].dropna()
        s_oot = X_oot[feat].dropna()
        psi_val = calcular_psi(s_test, s_oot, bins=bins)

        # Truncamento de percentis para visualização limpa em dados assimétricos
        p_low = min(np.percentile(s_test, 1), np.percentile(s_oot, 1))
        p_high = max(np.percentile(s_test, 99), np.percentile(s_oot, 99))

        sns.kdeplot(s_test[(s_test >= p_low) & (s_test <= p_high)], ax=ax,
                    color='#1B365D', linewidth=2.0, label='Teste (In-Time)', fill=True, alpha=0.25)
        sns.kdeplot(s_oot[(s_oot >= p_low) & (s_oot <= p_high)], ax=ax,
                    color='#8B0000', linewidth=2.0, linestyle='--', label='OOT', fill=True, alpha=0.25)

        status = 'Estável' if psi_val < 0.10 else ('Alerta' if psi_val < 0.25 else 'Crítico')
        ax.set_title(f'{feat}\nPSI = {psi_val:.4f} ({status})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor da Feature', fontsize=10)
        ax.set_ylabel('Densidade', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc')

    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)

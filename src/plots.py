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
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

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
    Plota a distribuição da variável alvo com paleta corporativa vermelha/cinza.
    """
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    ratio = target_counts[0] / target_counts[1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Paleta Corporativa
    cor_adimplente = "#D5D8DC" # Cinza claro
    cor_default = '#C00000'    # Vermelho corporativo

    bars = ax.bar(
        [0, 1], target_counts.values,
        color=[cor_adimplente, cor_default],
        edgecolor=['#AFAFAF', '#8B0000'],
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
                 fontsize=15, fontweight='bold', color='#1A1A1A', pad=25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Bom Pagador (0)', 'Devedor (1)'],
                       fontsize=12, fontweight='bold', color='#555555')
    
    # Limpeza visual (Decluttering)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.get_yaxis().set_visible(False)
    ax.set_ylim(0, max_y * 1.20) # Respiro para o texto
    
    _salvar_ou_exibir_grafico(filepath)
    return target_counts, target_pct, ratio


def plot_sazonalidade(df: pd.DataFrame, target_col: str = 'target', month_col: str = 'MONTH', filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Plota volume mensal e taxa de default usando a paleta vermelha.
    """
    taxa_mensal = df.groupby(month_col)[target_col].agg(Volume='count', Default_Rate='mean').copy()
    taxa_mensal['Default_Rate'] *= 100
    meses_str = taxa_mensal.index.astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')

    cor_volume = '#F2F3F4' # Cinza muito claro de fundo
    cor_borda = '#D5D8DC'
    ax1.bar(meses_str, taxa_mensal['Volume'], color=cor_volume, edgecolor=cor_borda, width=0.6, label='Volume de Pedidos')
    ax1.set_ylabel('Volume de Pedidos', fontsize=12, fontweight='bold', color='#666666')
    ax1.tick_params(axis='y', colors='#666666')

    ax2 = ax1.twinx()
    cor_taxa = '#C00000' # Vermelho destaque para o risco
    ax2.plot(meses_str, taxa_mensal['Default_Rate'], color=cor_taxa, marker='o', linewidth=3.5, markersize=8, label='Taxa de Default (%)')
    ax2.set_ylabel('Taxa de Default (%)', fontsize=12, fontweight='bold', color=cor_taxa)
    ax2.tick_params(axis='y', colors=cor_taxa)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
    ax2.set_ylim(0, taxa_mensal['Default_Rate'].max() * 1.3)

    for i, txt in enumerate(taxa_mensal['Default_Rate']):
        ax2.annotate(f'{txt:.2f}%', (meses_str[i], txt), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=cor_taxa)

    plt.title('Sazonalidade: Risco de Crédito vs Volume de Pedidos', fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)
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
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=11)

    _salvar_ou_exibir_grafico(filepath)
    return taxa_mensal[['Default_Rate', 'Volume']]


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


def plot_curvas_roc_pr_sobrepostas(y_test, prob_test, y_oot, prob_oot, figsize=(15, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Plota Curva ROC e PR lado a lado com paleta de vermelhos e cinzas corporativos.
    """
    fpr_test, tpr_test, _ = roc_curve(y_test, prob_test)
    auc_test = roc_auc_score(y_test, prob_test)
    fpr_oot, tpr_oot, _ = roc_curve(y_oot, prob_oot)
    auc_oot = roc_auc_score(y_oot, prob_oot)

    prec_test, rec_test, _ = precision_recall_curve(y_test, prob_test)
    pr_auc_test = average_precision_score(y_test, prob_test)
    prec_oot, rec_oot, _ = precision_recall_curve(y_oot, prob_oot)
    pr_auc_oot = average_precision_score(y_oot, prob_oot)

    cor_teste = '#7F8C8D'  # Cinza chumbo para o in-time (referencia)
    cor_oot = '#C00000'    # Vermelho vibrante para o OOT (foco principal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Plotagem Curva ROC
    ax1.plot(fpr_test, tpr_test, color=cor_teste, linewidth=2.2, label=f'Teste (In-Time) | ROC-AUC = {auc_test:.4f}')
    ax1.plot(fpr_oot, tpr_oot, color=cor_oot, linewidth=2.5, label=f'OOT (Out-of-Time) | ROC-AUC = {auc_oot:.4f}')
    ax1.plot([0, 1], [0, 1], color='#cccccc', linestyle=':', linewidth=1.5, label='Baseline Aleatório')
    
    ax1.set_xlabel('Taxa de Falsos Positivos', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sensibilidade (Verdadeiros Positivos)', fontsize=11, fontweight='bold')
    ax1.set_title('Capacidade de Ordenação: Curva ROC', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.35)

    # 2. Plotagem Curva PR
    ax2.plot(rec_test, prec_test, color=cor_teste, linewidth=2.2, label=f'Teste (In-Time) | PR-AUC = {pr_auc_test:.4f}')
    ax2.plot(rec_oot, prec_oot, color=cor_oot, linewidth=2.5, label=f'OOT (Out-of-Time) | PR-AUC = {pr_auc_oot:.4f}')
    
    base_test = y_test.mean()
    base_oot = y_oot.mean()
    ax2.axhline(base_test, color=cor_teste, linestyle=':', alpha=0.6)
    ax2.axhline(base_oot, color=cor_oot, linestyle=':', alpha=0.6)
    
    ax2.set_xlabel('Recall (Cobertura)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precisão', fontsize=11, fontweight='bold')
    ax2.set_title('Qualidade Preditiva: Curva Precision-Recall', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.35)

    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
from typing import Optional, Union

def plot_curva_roc(y_test, prob_test, y_oot, prob_oot, figsize=(8, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Plota exclusivamente a Curva ROC sobrepondo as bases Teste e OOT com padrão executivo.
    """
    # 1. Cálculo das métricas
    fpr_test, tpr_test, _ = roc_curve(y_test, prob_test)
    auc_test = roc_auc_score(y_test, prob_test)
    fpr_oot, tpr_oot, _ = roc_curve(y_oot, prob_oot)
    auc_oot = roc_auc_score(y_oot, prob_oot)

    # 2. Paleta Corporativa (Vermelhos e Cinzas)
    cor_teste = '#95A5A6'  # Cinza suave para a base de referência (In-Time)
    cor_oot = '#C00000'    # Vermelho corporativo vibrante para o foco principal (OOT)
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    # 3. Inicialização da Figura com fundo branco sólido
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(cor_fundo)
    ax.set_facecolor(cor_fundo)

    # 4. Plotagem (uso de zorder para garantir que a OOT fique sempre à frente)
    # Baseline Aleatório
    ax.plot([0, 1], [0, 1], color='#BDC3C7', linestyle='--', linewidth=1.5, zorder=1, label='Baseline Aleatório (0.50)')
    
    # Teste (In-Time)
    ax.plot(fpr_test, tpr_test, color=cor_teste, linewidth=2.5, zorder=2, 
            label=f'Teste (In-Time) | AUC = {auc_test:.3f}')
    
    # OOT (Foco) + Sombreamento da Área (AUC)
    ax.plot(fpr_oot, tpr_oot, color=cor_oot, linewidth=3.0, zorder=3, 
            label=f'OOT (Out-of-Time) | AUC = {auc_oot:.3f}')
    ax.fill_between(fpr_oot, tpr_oot, color=cor_oot, alpha=0.05, zorder=1)

    # 5. Formatação Executiva de Eixos e Títulos
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    
    # Título alinhado à esquerda atrai a atenção para o insight
    ax.set_title('Capacidade de Ordenação: Curva ROC (Teste vs. OOT)', 
                 fontsize=14, fontweight='bold', color='#1A1A1A', pad=20, loc='left')

    # Limites ligeiramente afastados das bordas para o traço não "colar" no eixo
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.tick_params(axis='both', colors=cor_texto, labelsize=10)

    # 6. Grid e Legenda
    ax.grid(True, linestyle=':', alpha=0.7, color='#BDC3C7', zorder=0)
    ax.set_axisbelow(True) # Joga o grid para trás das linhas
    
    ax.legend(loc='lower right', frameon=True, facecolor=cor_fundo, edgecolor='#E5E7E9', 
              fontsize=11, labelcolor=cor_texto, borderpad=1.2)

    # 7. Decluttering (Remoção de Spines)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_color('#E5E7E9')
    ax.spines['bottom'].set_color('#E5E7E9')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    _salvar_ou_exibir_grafico(filepath)


def plot_curva_pr(y_test, prob_test, y_oot, prob_oot, figsize=(8, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Plota exclusivamente a Curva Precision-Recall (PR) sobrepondo Teste e OOT.
    """
    prec_test, rec_test, _ = precision_recall_curve(y_test, prob_test)
    pr_auc_test = average_precision_score(y_test, prob_test)
    prec_oot, rec_oot, _ = precision_recall_curve(y_oot, prob_oot)
    pr_auc_oot = average_precision_score(y_oot, prob_oot)

    cor_teste = '#7F8C8D'
    cor_oot = '#C00000'

    plt.figure(figsize=figsize)
    plt.plot(rec_test, prec_test, color=cor_teste, linewidth=2.2, label=f'Teste (In-Time) | PR-AUC = {pr_auc_test:.4f}')
    plt.plot(rec_oot, prec_oot, color=cor_oot, linewidth=2.5, label=f'OOT (Out-of-Time) | PR-AUC = {pr_auc_oot:.4f}')

    base_test = y_test.mean()
    base_oot = y_oot.mean()
    plt.axhline(base_test, color=cor_teste, linestyle=':', alpha=0.6, label=f'Prevalência Teste ({base_test:.2%})')
    plt.axhline(base_oot, color=cor_oot, linestyle=':', alpha=0.6, label=f'Prevalência OOT ({base_oot:.2%})')

    plt.xlabel('Recall (Cobertura de Inadimplentes)', fontsize=11, fontweight='bold')
    plt.ylabel('Precisão (Taxa de Acerto no Risco)', fontsize=11, fontweight='bold')
    plt.title('Qualidade Preditiva: Curva Precision-Recall (Teste vs. OOT)', fontsize=13, fontweight='bold', pad=14)
    plt.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)


def plot_distribuicao_kde_scores(
    *args,
    y_test=None,
    prob_test=None,
    y_oot=None,
    prob_oot=None,
    y_true=None,
    prob=None,
    threshold: Optional[float] = None,
    nome_base: str = 'Out-of-Time (OOT)',
    figsize: Optional[Tuple[int, int]] = None,
    filepath: Optional[Union[str, Path]] = None
):
    """
    Plota a distribuição de densidade (KDE) dos scores. Suporta tanto:
      1) Modo Comparativo: (y_test, prob_test, y_oot, prob_oot, threshold=...)
      2) Modo Base Única:  (y_true, prob, threshold=..., nome_base='OOT')
    Possui padrão estético executivo, com linha de threshold sutil e transparente.
    """
    # Desempacotar args posicionais se passados
    if len(args) == 4:
        y_test, prob_test, y_oot, prob_oot = args
    elif len(args) == 2:
        y_true, prob = args
    elif len(args) == 1 and y_test is None and y_true is None:
        y_true = args[0]

    # Se chamado com y_true / prob (ou apenas OOT)
    if y_true is not None or (y_test is None and y_oot is not None):
        target_series = y_true if y_true is not None else y_oot
        prob_series = prob if prob is not None else prob_oot
        df_plot = pd.DataFrame({
            'score': prob_series,
            'target': target_series.values if hasattr(target_series, 'values') else target_series
        })

        fig_size = figsize if figsize is not None else (10, 5.5)
        fig, ax = plt.subplots(figsize=fig_size)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        cor_bons = '#95A5A6'
        cor_maus = '#C00000'

        sns.kdeplot(data=df_plot[df_plot['target'] == 0], x='score', color=cor_bons,
                    fill=True, alpha=0.15, linewidth=2.0, zorder=2, label='Bom Pagador (Target = 0)', ax=ax)
        sns.kdeplot(data=df_plot[df_plot['target'] == 1], x='score', color=cor_maus,
                    fill=True, alpha=0.35, linewidth=2.5, zorder=3, label='Inadimplente (Target = 1)', ax=ax)

        if threshold is not None:
            ax.axvline(threshold, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4,
                       label=f'Ponto de Corte Otimizado ({threshold:.4f})')
            max_y = ax.get_ylim()[1]
            ax.annotate(
                'Ponto de Corte\\nOtimizado',
                xy=(threshold, max_y * 0.72),
                xytext=(threshold + 0.05, max_y * 0.82),
                arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', lw=1.2, connectionstyle="arc3,rad=.2", alpha=0.8),
                fontsize=10, color='#333333', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFFF', edgecolor='#E5E7E9', alpha=0.9),
                zorder=5
            )

        ax.set_title(f'Distribuição de Densidade dos Scores (KDE) — {nome_base}', fontsize=13, fontweight='bold', color='#1A1A1A', pad=15)
        ax.set_xlabel('Probabilidade Predita de Inadimplência (Score)', fontsize=11, fontweight='bold', color='#333333', labelpad=10)
        ax.set_ylabel('Densidade', fontsize=11, fontweight='bold', color='#333333', labelpad=10)
        ax.set_xlim([-0.02, 1.02])
        ax.tick_params(axis='both', colors='#333333', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7', zorder=0)
        ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7E9', fontsize=10)
        sns.despine(ax=ax, top=True, right=True)

        _salvar_ou_exibir_grafico(filepath)
        return

    # Modo Comparativo (Teste vs OOT)
    df_test = pd.DataFrame({'score': prob_test, 'target': y_test.values if hasattr(y_test, 'values') else y_test})
    df_oot = pd.DataFrame({'score': prob_oot, 'target': y_oot.values if hasattr(y_oot, 'values') else y_oot})

    cor_bons = '#95A5A6'     # Cinza suave
    cor_maus = '#C00000'     # Vermelho corporativo
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    fig_size = figsize if figsize is not None else (16, 6)
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True)
    fig.patch.set_facecolor(cor_fundo)

    datasets = [
        (df_test, 'Teste (In-Time)', axes[0]),
        (df_oot, 'Out-of-Time (OOT)', axes[1])
    ]

    for df_plot, title, ax in datasets:
        ax.set_facecolor(cor_fundo)
        
        # Bons pagadores
        sns.kdeplot(data=df_plot[df_plot['target'] == 0], x='score', color=cor_bons,
                    fill=True, alpha=0.15, linewidth=2.0, zorder=2, label='Bom Pagador (Target = 0)', ax=ax)
        
        # Inadimplentes
        sns.kdeplot(data=df_plot[df_plot['target'] == 1], x='score', color=cor_maus,
                    fill=True, alpha=0.35, linewidth=2.5, zorder=3, label='Inadimplente (Target = 1)', ax=ax)

        # Threshold (Tornado mais transparente e sutil)
        if threshold is not None:
            ax.axvline(threshold, color='#555555', linestyle='--', linewidth=1.5, alpha=0.6, zorder=4, label=f'Threshold ({threshold:.4f})')
            
            max_y = ax.get_ylim()[1]
            ax.annotate(
                'Ponto de Corte\\nOtimizado',
                xy=(threshold, max_y * 0.72),
                xytext=(threshold + 0.05, max_y * 0.82),
                arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', lw=1.2, connectionstyle="arc3,rad=.2", alpha=0.8),
                fontsize=10, color='#555555', alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=cor_fundo, edgecolor='#E5E7E9', alpha=0.8),
                zorder=5
            )

        # Formatação Executiva por Subplot
        ax.set_title(f'{title}', fontsize=13, fontweight='bold', color='#1A1A1A', pad=15)
        ax.set_xlabel('Probabilidade Predita (Score)', fontsize=11, color=cor_texto, labelpad=10)
        ax.set_xlim([-0.02, 1.02])
        ax.tick_params(axis='both', colors=cor_texto, labelsize=10)

        # Grid e Legenda
        ax.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', frameon=True, facecolor=cor_fundo, edgecolor='#E5E7E9', fontsize=10, labelcolor=cor_texto)

        # Decluttering
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['left'].set_color('#E5E7E9')
        ax.spines['bottom'].set_color('#E5E7E9')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    axes[0].set_ylabel('Densidade', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    axes[1].set_ylabel('')

    # Título Geral da Figura
    fig.suptitle('Separação de Risco (KDE) — Distribuição de Scores: In-Time vs. Out-of-Time', 
                 fontsize=15, fontweight='bold', color='#1A1A1A', y=1.02)
    
    plt.tight_layout()

    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()  

def plot_cdf_ks(y_test, prob_test, y_oot, prob_oot, figsize=(16, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Plota as Funções de Distribuição Acumulada (CDF) empíricas comparando In-Time e Out-of-Time lado a lado.
    Evidencia visualmente o ponto de separação máxima (Estatística K-S) com padrão estético corporativo e limpo.
    """
    cor_bons = '#95A5A6'     # Cinza suave
    cor_maus = '#C00000'     # Vermelho corporativo
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor(cor_fundo)
    
    datasets = [
        (y_test, prob_test, 'Teste (In-Time)', axes[0]),
        (y_oot, prob_oot, 'Out-of-Time (OOT)', axes[1])
    ]
    
    ks_results = []
    
    for y_true, prob, title, ax in datasets:
        ax.set_facecolor(cor_fundo)
        y_arr = np.asarray(y_true)
        prob_arr = np.asarray(prob)

        scores_bons = np.sort(prob_arr[y_arr == 0])
        scores_maus = np.sort(prob_arr[y_arr == 1])

        # CDFs empíricas e Grid comum para cálculo da diferença
        todos_scores = np.sort(np.unique(prob_arr))
        cdf_bons_interp = np.searchsorted(scores_bons, todos_scores, side='right') / len(scores_bons)
        cdf_maus_interp = np.searchsorted(scores_maus, todos_scores, side='right') / len(scores_maus)

        # Cálculo K-S
        diff_cdf = np.abs(cdf_bons_interp - cdf_maus_interp)
        idx_ks = np.argmax(diff_cdf)
        ks_score_point = todos_scores[idx_ks]
        ks_valor = diff_cdf[idx_ks] * 100

        y_bons_point = cdf_bons_interp[idx_ks]
        y_maus_point = cdf_maus_interp[idx_ks]
        
        ks_results.append(ks_valor)

        # Plotagem das Curvas
        ax.plot(todos_scores, cdf_bons_interp, color=cor_bons, linewidth=2.5, zorder=2, label='CDF Bom Pagador (0)')
        ax.plot(todos_scores, cdf_maus_interp, color=cor_maus, linewidth=2.5, zorder=3, label='CDF Inadimplente (1)')
        
        # Segmento Vertical do K-S
        ax.vlines(x=ks_score_point, ymin=min(y_bons_point, y_maus_point), ymax=max(y_bons_point, y_maus_point),
                  color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4, label=f'K-S = {ks_valor:.2f}%')
        ax.plot([ks_score_point, ks_score_point], [y_bons_point, y_maus_point], 'o', color='#555555', markersize=5, alpha=0.7, zorder=4)

        # Anotação Refinada
        ax.annotate(
            f'K-S Máximo | Score: {ks_score_point:.3f}',
            xy=(ks_score_point, (y_bons_point + y_maus_point) / 2),
            xytext=(ks_score_point + 0.08, (y_bons_point + y_maus_point) / 2),
            arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', lw=1.2, connectionstyle="arc3,rad=.2", alpha=0.8),
            fontsize=10, color='#555555', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=cor_fundo, edgecolor='#E5E7E9', alpha=0.8),
            zorder=5
        )
        
        # Formatação Executiva por Subplot
        ax.set_title(f'{title}', fontsize=13, fontweight='bold', color='#1A1A1A', pad=15)
        ax.set_xlabel('Probabilidade Predita (Score)', fontsize=11, color=cor_texto, labelpad=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(axis='both', colors=cor_texto, labelsize=10)

        # Grid e Legenda
        ax.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', frameon=True, facecolor=cor_fundo, edgecolor='#E5E7E9', fontsize=10, labelcolor=cor_texto)

        # Decluttering
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['left'].set_color('#E5E7E9')
        ax.spines['bottom'].set_color('#E5E7E9')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
    axes[0].set_ylabel('Probabilidade Acumulada (CDF)', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    axes[1].set_ylabel('')

    # Título Geral da Figura
    fig.suptitle('Distribuição Acumulada (CDF) e Separação Máxima (K-S): In-Time vs. Out-of-Time', 
                 fontsize=15, fontweight='bold', color='#1A1A1A', y=1.02)
    
    plt.tight_layout()

    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

    return tuple(ks_results)

def plot_target_drift_mensal(df_test, df_oot, col_mes='MONTH', col_target='target', figsize=(14, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Gera gráfico temporal corporativo da taxa de default safra a safra (Target Drift)
    focado exclusivamente em Teste (In-Time) e OOT, com paleta executiva e rótulos de dados.
    """
    # 1. Preparação dos Dados
    dados = []
    for nome_base, base in [('Teste (In-Time)', df_test), ('Out-of-Time (OOT)', df_oot)]:
        agrupado = base.groupby(col_mes)[col_target].agg(['count', 'mean']).reset_index()
        agrupado['Base'] = nome_base
        dados.append(agrupado)

    df_consolidado = pd.concat(dados, ignore_index=True)
    df_consolidado['Taxa_Default_Pct'] = df_consolidado['mean'] * 100
    df_consolidado = df_consolidado.sort_values(col_mes)

    # 2. Definição da Paleta Corporativa
    cor_teste = '#95A5A6'  # Cinza suave
    cor_oot = '#C00000'    # Vermelho corporativo
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(cor_fundo)
    ax.set_facecolor(cor_fundo)

    paleta = {'Teste (In-Time)': cor_teste, 'Out-of-Time (OOT)': cor_oot}
    estilos = {'Teste (In-Time)': '--', 'Out-of-Time (OOT)': '-'}
    espessuras = {'Teste (In-Time)': 2.2, 'Out-of-Time (OOT)': 3.0}

    # 3. Plotagem das Linhas e Rótulos (Data Labels)
    for base, cor in paleta.items():
        subset = df_consolidado[df_consolidado['Base'] == base]
        if not subset.empty:
            x_vals = subset[col_mes].astype(str).values
            y_vals = subset['Taxa_Default_Pct'].values
            
            # Desenha a linha e os marcadores
            ax.plot(x_vals, y_vals, marker='o', markersize=7, linewidth=espessuras[base],
                    linestyle=estilos[base], color=cor, label=f'Taxa de Default — {base}', zorder=3)
            
            # Adiciona os rótulos acima dos pontos
            for x, y in zip(x_vals, y_vals):
                ax.annotate(f'{y:.2f}%', 
                            xy=(x, y), 
                            xytext=(0, 10), # Deslocamento vertical de 10 pontos
                            textcoords='offset points',
                            ha='center', va='bottom', 
                            fontsize=9.5, fontweight='bold', color=cor, zorder=4)

    # 4. Linha de Média Global
    total_target = df_test[col_target].sum() + df_oot[col_target].sum()
    total_count = len(df_test) + len(df_oot)
    media_total = (total_target / total_count) * 100
    ax.axhline(media_total, color='#555555', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1,
               label=f'Média Global ({media_total:.2f}%)')

    # 5. Formatação Executiva
    ax.set_title('Target Drift — Estabilidade da Taxa de Inadimplência por Safra (Teste vs. OOT)', 
                 fontsize=14, fontweight='bold', color='#1A1A1A', pad=20, loc='left')
    ax.set_xlabel('Safra de Decisão (Mês)', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    ax.set_ylabel('Taxa de Inadimplência (%)', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
    
    # Ajuste dos eixos para não cortar os rótulos de dados superiores
    y_max = df_consolidado['Taxa_Default_Pct'].max()
    y_min = df_consolidado['Taxa_Default_Pct'].min()
    margem = (y_max - y_min) * 0.35
    ax.set_ylim([max(0, y_min - margem), y_max + margem])
    
    ax.tick_params(axis='x', rotation=45, colors=cor_texto, labelsize=10)
    ax.tick_params(axis='y', colors=cor_texto, labelsize=10)

    # Grid e Legenda
    ax.grid(True, linestyle=':', alpha=0.7, color='#BDC3C7', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', frameon=True, facecolor=cor_fundo, edgecolor='#E5E7E9', fontsize=10.5, labelcolor=cor_texto)
    
    # Decluttering
    import seaborn as sns
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_color('#E5E7E9')
    ax.spines['bottom'].set_color('#E5E7E9')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 6. Salvar/Exibir
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

    return df_consolidado


def plot_comparativo_psi_features(X_test, X_oot, features_list, bins=10, figsize=(16, 5), filepath: Optional[Union[str, Path]] = None):
    """
    Plota distribuições comparativas entre Teste e OOT para features críticas
    com paleta vermelha e cinza corporativa, exibindo o PSI no título.
    Estética executiva aprimorada (decluttering).
    """
    cor_teste = '#95A5A6'
    cor_oot = '#C00000'
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    n_feats = len(features_list)
    fig, axes = plt.subplots(1, n_feats, figsize=(figsize[0], figsize[1]))
    fig.patch.set_facecolor(cor_fundo)
    
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_list):
        ax.set_facecolor(cor_fundo)
        s_test = X_test[feat].dropna()
        s_oot = X_oot[feat].dropna()
        
        # Cálculo do PSI
        psi_val = calcular_psi(s_test, s_oot, bins=bins)

        # Truncamento de percentis para visualização limpa
        p_low = min(np.percentile(s_test, 1), np.percentile(s_oot, 1))
        p_high = max(np.percentile(s_test, 99), np.percentile(s_oot, 99))

        # Plotagem
        sns.kdeplot(s_test[(s_test >= p_low) & (s_test <= p_high)], ax=ax,
                    color=cor_teste, linewidth=2.5, zorder=2, label='Teste (In-Time)', fill=True, alpha=0.15)
        sns.kdeplot(s_oot[(s_oot >= p_low) & (s_oot <= p_high)], ax=ax,
                    color=cor_oot, linewidth=2.5, zorder=3, label='OOT (Out-of-Time)', fill=True, alpha=0.35)

        # Regra de Cores para o Status (Farol)
        cor_status = '#27AE60' if psi_val < 0.10 else ('#E67E22' if psi_val < 0.25 else '#C0392B')
        status = 'Estável' if psi_val < 0.10 else ('Alerta' if psi_val < 0.25 else 'Crítico')
        
        # Formatação do Título e Eixos
        ax.set_title(f'{feat}\nPSI = {psi_val:.4f} ({status})', fontsize=12, fontweight='bold', color=cor_status, pad=15)
        ax.set_xlabel('Valor da Feature', fontsize=11, color=cor_texto, labelpad=10)
        
        # Oculta o eixo Y dos gráficos da direita para manter o visual limpo
        if ax == axes[0]:
            ax.set_ylabel('Densidade', fontsize=11, fontweight='bold', color=cor_texto, labelpad=12)
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='both', colors=cor_texto, labelsize=10)
        
        # Grid e Legenda
        ax.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=True, facecolor=cor_fundo, edgecolor='#E5E7E9', loc='upper right')
        
        # Decluttering (Spines)
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['left'].set_color('#E5E7E9')
        ax.spines['bottom'].set_color('#E5E7E9')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    # Adicionando um pequeno espaçamento no topo para o título não colar caso exista
    plt.subplots_adjust(top=0.88) 

    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()


def plot_matriz_confusao(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.5,
    titulo: str = "Matriz de Confusão",
    nome_base: str = "Out-of-Time (OOT)",
    figsize: Tuple[int, int] = (7, 6),
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Plota e analisa a matriz de confusao corporativa com contagens absolutas e percentuais normalizados.
    Calcula e exibe metricas operacionais de corte: Acuracia, Precisao, Recall, Especificidade e F1-Score.

    Parametros
    ----------
    y_true : pd.Series ou np.ndarray
        Classes reais (0 ou 1).
    y_pred : pd.Series ou np.ndarray, opcional
        Classes preditas ja binarizadas. Se None, y_prob e threshold serao utilizados.
    y_prob : pd.Series ou np.ndarray, opcional
        Probabilidades preditas de default para binarizacao com base no threshold.
    threshold : float, default=0.5
        Threshold de decisao aplicado caso y_prob seja informado.
    titulo : str, default='Matriz de Confusão'
        Titulo do grafico.
    nome_base : str, default='Out-of-Time (OOT)'
        Identificador do conjunto de dados avaliado.
    figsize : tuple, default=(7, 6)
        Dimensoes da figura matplotlib.
    filepath : str ou Path, opcional
        Caminho para gravacao em reports/figures/.

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo a matriz de confusao rotulada.
    """
    if y_pred is None:
        if y_prob is None:
            raise ValueError("Informe y_pred ou y_prob juntamente com o threshold.")
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    # Metricas derivadas
    acuracia = (tp + tn) / total if total > 0 else 0.0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0

    # Formatacao textual das celulas (Absoluto + Percentual do total)
    labels = np.array([
        [f"VN: {tn:,}\n({tn/total:.2%})", f"FP: {fp:,}\n({fp/total:.2%})"],
        [f"FN: {fn:,}\n({fn/total:.2%})", f"VP: {tp:,}\n({tp/total:.2%})"]
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        linewidths=1.5,
        linecolor="#ffffff",
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )

    ax.set_title(
        f"{titulo} — {nome_base}\n"
        f"Threshold = {threshold:.4f} | Recall (Sensib.) = {recall:.2%} | Precisão = {precisao:.2%}",
        fontsize=12,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Classe Predita (0 = Bom Pagador, 1 = Default)", fontsize=11, labelpad=10)
    ax.set_ylabel("Classe Real (0 = Bom Pagador, 1 = Default)", fontsize=11, labelpad=10)
    ax.set_xticklabels(["Aprovado (0)", "Reprovado (1)"], fontsize=10)
    ax.set_yticklabels(["Adimplente (0)", "Inadimplente (1)"], fontsize=10, rotation=0)

    # Subtitulo textual com resumo
    resumo_texto = (
        f"Acurácia: {acuracia:.2%} | Especificidade: {especificidade:.2%} | F1: {f1:.4f}\n"
        f"Total Casos: {total:,} | Falsos Positivos (Bons Reprovados): {fp:,} | Falsos Negativos (Maus Aprovados): {fn:,}"
    )
    plt.figtext(0.5, -0.05, resumo_texto, wrap=True, horizontalalignment='center', fontsize=9, style='italic', color='#333333')

    _salvar_ou_exibir_grafico(filepath)

    df_cm = pd.DataFrame(
        cm,
        index=['Real: Adimplente (0)', 'Real: Inadimplente (1)'],
        columns=['Pred: Aprovado (0)', 'Pred: Reprovado (1)']
    )
    return df_cm


def plot_importancia_shap(shap_values, features, top_n=15, titulo="Análise SHAP: Motores de Decisão", nome_modelo="Modelo LITE", filepath=None):
    """
    Ranking corporativo do SHAP (Vermelho escuro para Top 3, cinza para os demais).
    """
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
        shap_matriz = vals[:, :, 1] if vals.ndim == 3 else vals
    else:
        shap_matriz = np.asarray(shap_values)

    mean_abs_shap = np.abs(shap_matriz).mean(axis=0)
    df_shap = pd.DataFrame({'Feature': list(features), 'Impacto_Medio_SHAP': mean_abs_shap})
    df_top = df_shap.sort_values('Impacto_Medio_SHAP', ascending=False).head(top_n).sort_values('Impacto_Medio_SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Cores: Top 3 em destaque vermelho, o resto cinza
    cores = ['#C00000' if idx >= (len(df_top) - 3) else '#D5D8DC' for idx in range(len(df_top))]

    y_pos = range(len(df_top))
    bars = ax.barh(y_pos, df_top['Impacto_Medio_SHAP'], color=cores, edgecolor='white', height=0.68)
    ax.bar_label(bars, fmt='%.4f', padding=6, color='#2C3E50', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['Feature'], fontsize=11, color='#1A1A1A')
    ax.set_xlim(0, df_top['Impacto_Medio_SHAP'].max() * 1.20)
    ax.set_xlabel('Impacto Médio no Risco (|SHAP|)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f"{titulo} — {nome_modelo}", fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    sns.despine(top=True, right=True)
    _salvar_ou_exibir_grafico(filepath)
    return df_top


def plot_estabilidade_gini_temporal(
    df_gini: pd.DataFrame,
    meses_oot: Optional[List[Union[str, int]]] = None,
    media_oot: Optional[float] = None,
    titulo: str = "Estabilidade Temporal do GINI por Safra",
    nome_modelo: str = "LightGBM LITE Calibrado",
    filepath: Optional[Union[str, Path]] = None
) -> None:
    """
    Plota a evolução temporal do GINI safra a safra para todo o horizonte temporal
    (Treino + Teste + OOT), destacando a zona Out-of-Time com axvspan e paleta executiva.

    Parameters
    ----------
    df_gini : pd.DataFrame
        DataFrame contendo as colunas 'MONTH' (ou safra) e 'GINI'.
    meses_oot : list, optional
        Lista de identificadores de safras pertencentes à janela OOT.
    media_oot : float, optional
        Média do GINI na janela OOT para linha de referência executiva.
    titulo : str
        Título do gráfico.
    nome_modelo : str
        Nome do modelo exibido no subtítulo.
    filepath : str or Path, optional
        Caminho para salvamento em reports/figures/.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Formatação consistente do eixo X
    df_plot = df_gini.copy()
    df_plot['MONTH_STR'] = df_plot['MONTH'].astype(str)
    x_labels = df_plot['MONTH_STR'].tolist()
    x_indices = np.arange(len(x_labels))
    gini_values = df_plot['GINI'].values

    # Identificação da zona OOT para axvspan
    if meses_oot is not None:
        meses_oot_str = [str(m) for m in meses_oot]
        oot_indices = [i for i, m in enumerate(x_labels) if m in meses_oot_str]
    else:
        oot_indices = []

    if oot_indices:
        start_oot = min(oot_indices) - 0.5
        end_oot = max(oot_indices) + 0.5
        # Zona sombreada cinza clara (alpha=0.2)
        ax.axvspan(start_oot, end_oot, color='#BDC3C7', alpha=0.20, label='Zona Out-of-Time (OOT)')
        
        # Anotação textual corporativa
        y_pos_note = gini_values.max() * 1.05 if len(gini_values) > 0 else 0.70
        ax.text(
            (start_oot + end_oot) / 2,
            y_pos_note,
            "Zona Out-of-Time\n(Avaliação Exclusiva OOT)",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F3F4', edgecolor='#BDC3C7', alpha=0.9)
        )

    # Linha principal temporal do GINI (Vermelho executivo para destaque do modelo campeão)
    ax.plot(
        x_indices,
        gini_values,
        marker='o',
        markersize=6,
        linewidth=2.4,
        color='#C00000',
        label=f'GINI ({nome_modelo})'
    )

    # Rótulos de dados sobre cada ponto
    for x, y in zip(x_indices, gini_values):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#1A1A1A'
        )

    # Linha de referência da média OOT se fornecida
    if media_oot is not None:
        ax.axhline(
            media_oot,
            color='#7F8C8D',
            linestyle='--',
            linewidth=1.5,
            label=f'Média OOT: {media_oot:.4f}'
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # Margens seguras para evitar textos cortados
    y_min = max(0.0, gini_values.min() - 0.08)
    y_max = min(1.0, gini_values.max() + 0.12)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)

    ax.set_title(f"{titulo} — {nome_modelo}", fontsize=13, fontweight='bold', pad=25)
    ax.set_xlabel("Mês de Decisão (Safra)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("GINI Score", fontsize=11, fontweight='bold', labelpad=10)

    # Decluttering e grid sutil
    ax.grid(True, linestyle='--', alpha=0.3, color='#BDC3C7')
    sns.despine(top=True, right=True)
    ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#BDC3C7', loc='lower right', fontsize=10)

    _salvar_ou_exibir_grafico(filepath)




"""
drift.py — Metricas de estabilidade temporal, Population Stability Index (PSI) e Drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'


def teste_qui_quadrado_temporal(
    taxa_mensal_display: pd.DataFrame,
    col_taxa: str = 'Default Rate (%)',
    col_volume: str = 'Volume',
) -> Dict[str, float]:
    """
    Aplica teste qui-quadrado de independencia entre mes e target,
    e calcula V de Cramer como tamanho do efeito.

    Parameters
    ----------
    taxa_mensal_display : pd.DataFrame
        Tabela mensal (index = mes) com a taxa de default e o volume.
    col_taxa : str, default 'Default Rate (%)'
        Nome da coluna com a taxa de default EM % (ex: 2.599 = 2.599%).
    col_volume : str, default 'Volume'
        Nome da coluna com o volume de observacoes do mes.

    Returns
    -------
    Dict[str, float]
        Dicionario com chi2, p_valor, gl, freq_esperada_min, cramers_v.
    """

    # ---- Validacao amigavel de colunas ----
    faltantes = [c for c in (col_taxa, col_volume)
                 if c not in taxa_mensal_display.columns]
    if faltantes:
        raise KeyError(
            f"Colunas ausentes em 'taxa_mensal_display': {faltantes}\n"
            f"Colunas disponiveis: {taxa_mensal_display.columns.tolist()}\n"
            f"Dica: chame com col_taxa='Default_Rate' se for o seu caso."
        )

    # ---- Construcao da tabela de contingencia ----
    contingencia = taxa_mensal_display.copy()
    contingencia['taxa_decimal'] = contingencia[col_taxa] / 100
    contingencia['eventos'] = (
        contingencia['taxa_decimal'] * contingencia[col_volume]
    ).round().astype(int)
    contingencia['nao_eventos'] = (
        contingencia[col_volume] - contingencia['eventos']
    )
    tabela = contingencia[['eventos', 'nao_eventos']]

    # ---- Teste qui-quadrado ----
    chi2, p, gl, expected = stats.chi2_contingency(tabela)

    n = tabela.values.sum()
    k = min(tabela.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else float('nan')

    # ---- Relatorio ----
    print(f"Qui-Quadrado:            {chi2:.2f}")
    print(f"Graus de Liberdade (gl): {gl}")
    print(f"P-valor:                 {p:.4e}")
    print(f"Frequencia esperada min: {expected.min():.2f}")
    print(f"V de Cramer:             {cramers_v:.4f}")

    return {
        'chi2': chi2,
        'p_valor': p,
        'gl': gl,
        'freq_esperada_min': expected.min(),
        'cramers_v': cramers_v,
    }


# =============================================================================
# 6. MERGE DEPTH 0
# =============================================================================


def calcular_psi(esperado, atual, bins=10):
    """Calcula o PSI para uma variável numérica contínua para deteção de drift."""
    esperado = esperado.dropna()
    atual = atual.dropna()
    
    limites = np.unique(np.percentile(esperado, np.linspace(0, 100, bins + 1)))
    if len(limites) < 2: return 0.0
    
    limites[0], limites[-1] = -np.inf, np.inf
    
    contagem_esperado, _ = np.histogram(esperado, bins=limites)
    contagem_atual, _ = np.histogram(atual, bins=limites)
    
    contagem_esperado = np.where(contagem_esperado == 0, 0.001, contagem_esperado)
    contagem_atual = np.where(contagem_atual == 0, 0.001, contagem_atual)
    
    perc_esperado = contagem_esperado / np.sum(contagem_esperado)
    perc_atual = contagem_atual / np.sum(contagem_atual)
    
    return np.sum((perc_atual - perc_esperado) * np.log(perc_atual / perc_esperado))


def relatorio_feature_drift_psi(X_hist, X_oot, bins=10):
    """Gera relatório de PSI (Population Stability Index) de todas as features numéricas."""
    from scipy.stats import ks_2samp
    cols_num = X_hist.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']).columns
    resultados = []

    for col in cols_num:
        psi_val = calcular_psi(X_hist[col], X_oot[col], bins=bins)
        ks_val, _ = ks_2samp(X_hist[col].dropna(), X_oot[col].dropna())

        if psi_val < 0.10: status = " Estável"
        elif psi_val < 0.25: status = " Alerta (Drift Moderado)"
        else: status = " Crítico (Drift Severo)"

        resultados.append({'Variável': col, 'PSI': psi_val, 'KS': ks_val, 'Status': status})

    df_psi = pd.DataFrame(resultados).sort_values(by='PSI', ascending=False).reset_index(drop=True)
    df_psi['PSI'] = df_psi['PSI'].apply(lambda x: f"{x:.4f}")
    df_psi['KS'] = df_psi['KS'].apply(lambda x: f"{x:.4f}")

    return df_psi


# =============================================================================


def calcular_psi_scores(probs_treino, probs_oot, bins=10):
    """
    PSI (Population Stability Index) entre a distribuição de SCORES do treino
    e do OOT, com faixas definidas pelos decis do treino (limites em [0, 1]).

    Nota: renomeada de `calcular_psi` (versão do notebook) para não sobrescrever
    a função `calcular_psi(esperado, atual, bins)` já existente neste módulo,
    usada por `relatorio_feature_drift_psi`. Lógica inalterada.

    Retorno:
    -------
    (pd.DataFrame, float)
        Tabela de PSI por faixa e o PSI total.
    """
    _, limites = pd.qcut(probs_treino, q=bins, retbins=True, duplicates='drop')
    limites[0] = 0.0
    limites[-1] = 1.0

    esperado = pd.cut(probs_treino, bins=limites).value_counts() / len(probs_treino)
    real = pd.cut(probs_oot, bins=limites).value_counts() / len(probs_oot)

    df_psi = pd.DataFrame({'Faixa': esperado.index, 'Treino_%': esperado.values, 'OOT_%': real.values})
    df_psi['Treino_%'] = np.where(df_psi['Treino_%'] == 0, 0.0001, df_psi['Treino_%'])
    df_psi['OOT_%'] = np.where(df_psi['OOT_%'] == 0, 0.0001, df_psi['OOT_%'])
    df_psi['PSI'] = (df_psi['OOT_%'] - df_psi['Treino_%']) * np.log(df_psi['OOT_%'] / df_psi['Treino_%'])

    psi_total = df_psi['PSI'].sum()

    print(f'PSI Total: {psi_total:.4f}')
    if psi_total < 0.10:
        print('[STATUS] EXCELENTE (nenhuma mudanca populacional).')
    elif psi_total < 0.25:
        print('[STATUS] ATENCAO (leve degradacao).')
    else:
        print('[STATUS] ALERTA (concept drift provavel).')

    return df_psi, psi_total


# =============================================================================
# 14. VALIDAÇÃO OOT E ANÁLISE DE NEGÓCIO (04_Model_Avaliation.ipynb)
# =============================================================================


def analisar_estabilidade_gini(X_val, y_val, meses, dicionario_modelos, filepath: Optional[Union[str, Path]] = None):
    """
    GINI mês a mês (safra) do ensemble por MÉDIA SIMPLES dos 3 Boostings
    ('LightGBM_Opt', 'XGBoost_Opt', 'CatBoost_Opt') e plota a série temporal.

    Retorno:
    -------
    (pd.DataFrame, np.ndarray)
        DataFrame ['MONTH', 'GINI'] e o vetor de probabilidades do ensemble.
    """
    probs_ensemble = np.zeros(len(X_val))
    modelos_avaliados = list(dicionario_modelos.keys())
    for nome, modelo in dicionario_modelos.items():
        probs_ensemble += modelo.predict_proba(X_val)[:, 1]
    probs_ensemble /= max(len(modelos_avaliados), 1)

    df_eval = pd.DataFrame({'MONTH': meses.values, 'target': y_val.values, 'prob': probs_ensemble})
    resultados_mes = []

    for mes in sorted(df_eval['MONTH'].unique()):
        df_mes = df_eval[df_eval['MONTH'] == mes]
        if len(df_mes['target'].unique()) > 1:
            auc = roc_auc_score(df_mes['target'], df_mes['prob'])
            gini = (2 * auc) - 1
            resultados_mes.append({'MONTH': str(mes), 'GINI': gini})

    df_gini = pd.DataFrame(resultados_mes)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_gini, x='MONTH', y='GINI', marker='o', linewidth=2.5, color='#2980b9')
    media_gini = df_gini['GINI'].mean()
    plt.axhline(media_gini, color='#e74c3c', linestyle='--', label=f'Media OOT: {media_gini:.4f}')
    plt.title('Estabilidade Temporal do GINI (Out-of-Time)', fontsize=14, pad=15)
    plt.xlabel('Mes da Decisao (Safra)', fontsize=12)
    plt.ylabel('GINI Score', fontsize=12)
    plt.ylim(0.40, 0.70)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filepath:
        save_path = Path(filepath)
        if not save_path.is_absolute():
            save_path = FIGURES_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return df_gini, probs_ensemble

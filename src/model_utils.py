"""
model_utils.py — Treinamento, avaliacao, otimizacao de threshold por lucro,
poda SHAP e serializacao de artefatos de modelos preditivos.
"""
from __future__ import annotations
import types
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'


def avaliar_classificador(modelo, X_train, y_train, X_test, y_test, nome_modelo, cv=5, thresholds_dict=None):
    # Definição do Threshold
    threshold = 0.5
    if thresholds_dict is not None and nome_modelo in thresholds_dict:
        threshold = thresholds_dict[nome_modelo]
        
    print(f"{'='*105}")
    print(f" {nome_modelo.upper()} (CV = {cv} | Threshold = {threshold:.4f})")
    print(f"{'-'*105}")
    
    # Proporção inicial do target (Baseline)
    prop_tr = y_train.mean() * 100
    prop_te = y_test.mean() * 100
    print(f" Taxa Base de Default (Target=1): Treino = {prop_tr:.2f}% | Teste = {prop_te:.2f}%")
    print(f"{'-'*105}")
    
    # 1. Cross-Validation (estritamente no Treino)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Listas Independentes
    cv_roc, cv_gini, cv_pr, cv_ks = [], [], [], []
    # Listas Dependentes do Threshold
    cv_acc, cv_prec, cv_rec = [], [], []
    
    X_tr_arr, y_tr_arr = X_train.values, y_train.values
    
    for train_idx, val_idx in skf.split(X_tr_arr, y_tr_arr):
        X_f_train, y_f_train = X_tr_arr[train_idx], y_tr_arr[train_idx]
        X_f_val, y_f_val = X_tr_arr[val_idx], y_tr_arr[val_idx]
        
        modelo.fit(X_f_train, y_f_train)
        probs_val = modelo.predict_proba(X_f_val)[:, 1]
        preds_val = (probs_val >= threshold).astype(int)
        
        # Métricas Independentes do Fold
        roc = roc_auc_score(y_f_val, probs_val)
        cv_roc.append(roc)
        cv_gini.append(2 * roc - 1)
        cv_pr.append(average_precision_score(y_f_val, probs_val))
        pos = probs_val[y_f_val == 1]
        neg = probs_val[y_f_val == 0]
        cv_ks.append(ks_2samp(pos, neg).statistic if len(pos) > 0 else 0)
        
        # Métricas Dependentes do Fold
        cv_acc.append(accuracy_score(y_f_val, preds_val))
        cv_prec.append(precision_score(y_f_val, preds_val, zero_division=0))
        cv_rec.append(recall_score(y_f_val, preds_val, zero_division=0))
        
    # Intervalos de Confiança (CV) - Métricas principais
    ic_roc = f"[{np.mean(cv_roc) - 2*np.std(cv_roc):.4f}, {np.mean(cv_roc) + 2*np.std(cv_roc):.4f}]"
    ic_gini = f"[{np.mean(cv_gini) - 2*np.std(cv_gini):.4f}, {np.mean(cv_gini) + 2*np.std(cv_gini):.4f}]"

    # 2. Treino Final (No Dataset Completo de Treino)
    modelo.fit(X_train, y_train)
    
    # 3. Previsões Finais
    prob_train = modelo.predict_proba(X_train)[:, 1]
    prob_test = modelo.predict_proba(X_test)[:, 1]
    
    pred_train = (prob_train >= threshold).astype(int)
    pred_test = (prob_test >= threshold).astype(int)
    
    # 4. Cálculo de Métricas Finais
    # Independentes
    roc_tr, roc_te = roc_auc_score(y_train, prob_train), roc_auc_score(y_test, prob_test)
    gini_tr, gini_te = 2 * roc_tr - 1, 2 * roc_te - 1
    pr_tr, pr_te = average_precision_score(y_train, prob_train), average_precision_score(y_test, prob_test)
    ks_tr = ks_2samp(prob_train[y_train == 1], prob_train[y_train == 0]).statistic
    ks_te = ks_2samp(prob_test[y_test == 1], prob_test[y_test == 0]).statistic
    
    # Dependentes
    acc_tr, acc_te = accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)
    prec_tr, prec_te = precision_score(y_train, pred_train, zero_division=0), precision_score(y_test, pred_test, zero_division=0)
    rec_tr, rec_te = recall_score(y_train, pred_train, zero_division=0), recall_score(y_test, pred_test, zero_division=0)
    
    # 5. Funções de formatação de output
    def format_row(name, tr, te, cv_mean, ic="-"):
        gap = te - tr
        gap_str = f"{'+' if gap > 0 else ''}{gap:.4f}"
        return f" {name:<10} | {tr:.4f}     | {te:.4f}     | {gap_str:<10} | {cv_mean:.4f}             | {ic}"

    header = f" {'Metrica':<10} | {'Treino':<10} | {'Teste':<10} | {'Gap(Te-Tr)':<10} | {'Media CV (Treino)':<18} | {'IC 95% (CV Treino)'}"
    
    # Tabela 1: Métricas Independentes do Threshold (Probabilidades)
    print("\n [ METRICAS INDEPENDENTES DO THRESHOLD (Probabilidades) ]")
    print(header)
    print("-" * len(header))
    print(format_row("ROC-AUC", roc_tr, roc_te, np.mean(cv_roc), ic_roc))
    print(format_row("GINI", gini_tr, gini_te, np.mean(cv_gini), ic_gini))
    print(format_row("PR-AUC", pr_tr, pr_te, np.mean(cv_pr)))
    print(format_row("KS", ks_tr, ks_te, np.mean(cv_ks)))
    
    # Tabela 2: Métricas Dependentes do Threshold (Classes Binárias)
    print(f"\n [ METRICAS DEPENDENTES DO THRESHOLD (Corte = {threshold:.4f}) ]")
    print(header)
    print("-" * len(header))
    print(format_row("Accuracy", acc_tr, acc_te, np.mean(cv_acc)))
    print(format_row("Precision", prec_tr, prec_te, np.mean(cv_prec)))
    print(format_row("Recall", rec_tr, rec_te, np.mean(cv_rec)))
    print(f"{'='*105}\n")
    
    return modelo

# ==============================================================================
# 2. EXECUÇÃO DO LOOP COM AS BASES SEPARADAS
# ==============================================================================


def avaliar_classificacao_oot(modelo, X_test, y_test, X_hist, y_hist, threshold=0.5, nome_modelo="Modelo"):
    """
    Avalia o modelo na base Out-of-Time (OOT), exibindo a proporção do target 
    e comparando as métricas independentes lado a lado com a base Histórica.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from scipy.stats import ks_2samp
    
    tx_hist = y_hist.mean() * 100
    tx_oot = y_test.mean() * 100
    dif_tx = tx_oot - tx_hist
    
    prob_hist = modelo.predict_proba(X_hist)[:, 1]
    met_hist = {
        'ROC-AUC': roc_auc_score(y_hist, prob_hist),
        'PR-AUC': average_precision_score(y_hist, prob_hist)
    }
    ks_hist, _ = ks_2samp(prob_hist[y_hist == 1], prob_hist[y_hist == 0])
    met_hist['KS'] = ks_hist

    prob_oot = modelo.predict_proba(X_test)[:, 1]
    pred_oot = (prob_oot >= threshold).astype(int)
    
    met_oot = {
        'ROC-AUC': roc_auc_score(y_test, prob_oot),
        'PR-AUC': average_precision_score(y_test, prob_oot)
    }
    ks_oot, _ = ks_2samp(prob_oot[y_test == 1], prob_oot[y_test == 0])
    met_oot['KS'] = ks_oot

    print("=" * 80)
    print(f" DESEMPENHO OOT: {nome_modelo.upper()}")
    print("-" * 80)
    
    fmt_dif_tx = f"+{dif_tx:.2f}%" if dif_tx >= 0 else f"{dif_tx:.2f}%"
    print(f" DISTRIBUIÇÃO DO TARGET (DEFAULT)")
    print(f"  -> Histórico : {tx_hist:.2f}%")
    print(f"  -> Teste/OOT : {tx_oot:.2f}% (Variação: {fmt_dif_tx})")
    print("-" * 80)
    
    print(f"  {'Métrica':<15} | {'Histórico':<15} | {'Teste/OOT':<15} | {'Variação (Drop)'}")
    print(f"  {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}")
    
    for metrica in met_hist.keys():
        val_h = met_hist[metrica]
        val_o = met_oot[metrica]
        diff = val_o - val_h
        
        fmt_diff = f"{diff:>+8.4f}"
        print(f"  {metrica:<15} | {val_h:<15.4f} | {val_o:<15.4f} | {fmt_diff}")
        
    print("=" * 80)
    return prob_oot, pred_oot


def objective_lgbm(trial, X_tune, y_tune, cv):
    """Funcao objetivo Optuna para LightGBM. Maximiza ROC-AUC via CV."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    }

    modelo = LGBMClassifier(**params)
    auc_scores = []
    for tr_idx, val_idx in cv.split(X_tune, y_tune):
        modelo.fit(X_tune.iloc[tr_idx], y_tune.iloc[tr_idx])
        probs = modelo.predict_proba(X_tune.iloc[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_tune.iloc[val_idx], probs))
    return np.mean(auc_scores)


def objective_xgb(trial, X_tune, y_tune, cv, taxa_desbalanceamento):
    """Funcao objetivo Optuna para XGBoost. Maximiza ROC-AUC via CV."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'scale_pos_weight': taxa_desbalanceamento,
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'verbosity': 0,
    }

    modelo = XGBClassifier(**params)
    auc_scores = []
    for tr_idx, val_idx in cv.split(X_tune, y_tune):
        modelo.fit(X_tune.iloc[tr_idx], y_tune.iloc[tr_idx], verbose=False)
        probs = modelo.predict_proba(X_tune.iloc[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_tune.iloc[val_idx], probs))
    return np.mean(auc_scores)


def objective_cat(trial, X_tune, y_tune, cv):
    """Funcao objetivo Optuna para CatBoost. Maximiza ROC-AUC via CV."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    params = {
        'iterations': trial.suggest_int('iterations', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'auto_class_weights': 'Balanced',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,
    }

    modelo = CatBoostClassifier(**params)
    auc_scores = []
    for tr_idx, val_idx in cv.split(X_tune, y_tune):
        modelo.fit(X_tune.iloc[tr_idx], y_tune.iloc[tr_idx])
        probs = modelo.predict_proba(X_tune.iloc[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_tune.iloc[val_idx], probs))
    return np.mean(auc_scores)


def obter_menores_features(
    modelo,
    X_amostra: pd.DataFrame,
    bottom_n: int = 20,
    exibir: bool = True,
) -> pd.DataFrame:
    """
    Calcula a média absoluta dos SHAP values (|SHAP|) para cada variável em uma amostra
    e retorna/exibe as variáveis com menor impacto ('fraqueza' preditiva).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn / LightGBM / CatBoost / XGBoost treinado ou Pipeline.
    X_amostra : pd.DataFrame
        Amostra de dados (ex: X_test.sample(2000, random_state=42)) usada para extrair os SHAP values.
    bottom_n : int, default=20
        Quantidade de features mais fracas a retornar.
    exibir : bool, default=True
        Se True, imprime a tabela no console.

    Retorno:
    -------
    pd.DataFrame:
        DataFrame com colunas ['Feature', 'fraqueza'] ordenado de forma ascendente.
    """
    clf = modelo
    # Extrai estimador final caso seja uma Pipeline do sklearn
    if hasattr(modelo, 'named_steps'):
        if 'clf' in modelo.named_steps:
            clf = modelo.named_steps['clf']
        elif 'classifier' in modelo.named_steps:
            clf = modelo.named_steps['classifier']
        elif 'model' in modelo.named_steps:
            clf = modelo.named_steps['model']
        else:
            clf = modelo[-1]

    # Identifica o explainer adequado (TreeExplainer para árvores, Explainer genérico caso contrário)
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_amostra)
    except Exception:
        explainer = shap.Explainer(clf, X_amostra)
        shap_values = explainer(X_amostra).values

    # Tratamento para modelos de classificação binária
    # Alguns modelos retornam lista [shap_classe_0, shap_classe_1] ou array 3D (amostras, features, classes)
    if isinstance(shap_values, list):
        shap_matriz = shap_values[1]  # Classe positiva (target = 1)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_matriz = shap_values[:, :, 1]
    else:
        shap_matriz = shap_values

    # Calcula a média do impacto absoluto (|SHAP|) por feature
    shap_media_abs = np.abs(shap_matriz).mean(axis=0)

    # Monta o DataFrame, ordena da menor para a maior e filtra as 'bottom_n'
    df_bottom = (
        pd.DataFrame({
            'Feature': list(X_amostra.columns),
            'fraqueza': shap_media_abs
        })
        .sort_values('fraqueza', ascending=True)
        .head(bottom_n)
        .reset_index(drop=True)
    )

    if exibir:
        print(df_bottom.to_string(index=False))

    return df_bottom


# =============================================================================
# 13. MODELAGEM — THRESHOLD FINANCEIRO, PODA SHAP E ESTABILIDADE TEMPORAL
#     (funções migradas do notebook 03_Modeling.ipynb — lógica inalterada)
# =============================================================================


def podar_features_por_shap(dicionario_modelos, X_train, y_train, X_test, y_test, X_amostra, top_n_remover=8):
    """
    Para cada modelo de árvore (Pipelines e Logistic são ignorados):
      1. calcula o impacto global |SHAP| médio via TreeExplainer em X_amostra;
      2. identifica as `top_n_remover` features de menor impacto;
      3. re-treina uma versão LITE (mesmos hiperparâmetros) sem essas features;
      4. compara o ROC-AUC original vs LITE em X_test (aceita se delta >= -0.0010).

    Retorno:
    -------
    dict[str, list[str]]
        Features sugeridas para corte, por modelo.
    """
    features_sugeridas_para_corte = {}

    for nome, modelo in dicionario_modelos.items():
        if 'Pipeline' in str(type(modelo)) or 'Logistic' in nome:
            continue

        print(f'\n{"=" * 65}')
        print(f' ANALISE SHAP E SANITY CHECK DE PODA: {nome} ')
        print(f'{"=" * 65}')

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_amostra)

        if isinstance(shap_values, list):
            shap_values_positivos = shap_values[1]
        else:
            shap_values_positivos = shap_values

        importancia_global = np.abs(shap_values_positivos).mean(axis=0)
        df_importancia = pd.DataFrame({
            'Feature': X_amostra.columns,
            'SHAP_Impact': importancia_global
        }).sort_values(by='SHAP_Impact', ascending=True)

        piores_features = df_importancia.head(top_n_remover)['Feature'].tolist()
        features_sugeridas_para_corte[nome] = piores_features

        print(f'As {top_n_remover} features menos relevantes:')
        for f in piores_features:
            print(f'  - {f}')

        probs_originais = modelo.predict_proba(X_test)[:, 1]
        auc_original = roc_auc_score(y_test, probs_originais)

        X_train_podado = X_train.drop(columns=piores_features)
        X_test_podado = X_test.drop(columns=piores_features)

        modelo_podado = modelo.__class__(**modelo.get_params())
        print(f'\nTreinando arquitetura LITE sem {top_n_remover} features...')
        if 'CatBoost' in nome or 'XGB' in nome:
            modelo_podado.fit(X_train_podado, y_train, verbose=False)
        else:
            modelo_podado.fit(X_train_podado, y_train)

        probs_podadas = modelo_podado.predict_proba(X_test_podado)[:, 1]
        auc_podado = roc_auc_score(y_test, probs_podadas)
        delta_auc = auc_podado - auc_original

        print(f'  ROC-AUC Original : {auc_original:.4f}')
        print(f'  ROC-AUC Lite     : {auc_podado:.4f}')
        print(f'  Delta (AUC)      : {delta_auc:+.4f}')

        if delta_auc >= -0.0010:
            print('\n  [VEREDITO] Valido! Modelo mais limpo e rapido sem perda significativa.')
        else:
            print('\n  [VEREDITO] Rejeitado! Features tinham interacoes ocultas.')

    return features_sugeridas_para_corte


def gerar_tabela_decis_ks(y_true, prob, n_decis=10, filepath: Optional[Union[str, Path]] = None):
    """
    Calcula a tabela de ordenação de risco por decis de score e plota a curva
    de Taxa de Default por Decil ('escadinha' de risco) e a estatística K-S.
    """
    df = pd.DataFrame({'target': y_true.values if hasattr(y_true, 'values') else y_true, 'score': prob})
    
    # Decil 10 = Maior Risco (score mais alto)
    df['decil'] = pd.qcut(df['score'], q=n_decis, labels=False, duplicates='drop') + 1

    tabela = df.groupby('decil').agg(
        total_clientes=('target', 'count'),
        inadimplentes=('target', 'sum'),
        score_min=('score', 'min'),
        score_medio=('score', 'mean'),
        score_max=('score', 'max')
    ).reset_index()

    tabela['bons'] = tabela['total_clientes'] - tabela['inadimplentes']
    tabela['taxa_default_pct'] = (tabela['inadimplentes'] / tabela['total_clientes']) * 100

    total_maus = tabela['inadimplentes'].sum()
    total_bons = tabela['bons'].sum()

    tabela['pct_maus'] = (tabela['inadimplentes'] / total_maus) * 100
    tabela['pct_bons'] = (tabela['bons'] / total_bons) * 100

    # Ordenação do decil 10 (maior score) para o decil 1 (menor score) para cálculo de K-S
    tabela = tabela.sort_values(by='decil', ascending=False).reset_index(drop=True)
    tabela['cum_pct_maus'] = tabela['pct_maus'].cumsum()
    tabela['cum_pct_bons'] = tabela['pct_bons'].cumsum()
    tabela['ks'] = np.abs(tabela['cum_pct_maus'] - tabela['cum_pct_bons'])

    # Retorna tabela ordenada por decil (1 a 10)
    tabela = tabela.sort_values(by='decil', ascending=True).reset_index(drop=True)

    # Gráfico de barras da taxa de default por decil
    plt.figure(figsize=(11, 5))
    bars = plt.bar(tabela['decil'].astype(str), tabela['taxa_default_pct'],
                   color='#1B365D', edgecolor='#0A192F', alpha=0.88)

    # Destaque no decil de maior risco
    bars[-1].set_color('#8B0000')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f'{h:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ks_max = tabela['ks'].max()
    plt.title(f'Ordenação de Risco por Decis de Score — Base OOT (K-S Máximo = {ks_max:.2f}%)',
              fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('Decil de Score (1 = Menor Risco | 10 = Maior Risco)', fontsize=11, fontweight='bold')
    plt.ylabel('Taxa Real de Default (%)', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    if filepath:
        save_path = Path(filepath)
        if not save_path.is_absolute():
            save_path = FIGURES_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return tabela


def simular_lucro_e_curva_tradeoff(y_true, prob, valores_credito, margem_lucro=0.15, threshold_otimizado=0.7425, figsize=(12, 6), filepath: Optional[Union[str, Path]] = None):
    """
    Simula o lucro financeiro em OOT no threshold ótimo e gera o gráfico
    de trade-off entre Taxa de Aprovação vs. Risco da Carteira Aprovada.
    Estética executiva com threshold suavizado e paleta corporativa.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    y_true_arr = np.array(y_true)
    valores_arr = np.array(valores_credito)

    taxas_aprovacao = []
    riscos_carteira = []
    lucros = []

    # 1. Loop de Simulação Financeira
    for t in thresholds:
        aprovados = prob < t
        n_aprovados = np.sum(aprovados)

        if n_aprovados > 0:
            tx_aprov = (n_aprovados / len(prob)) * 100
            tx_risco = (np.sum(aprovados & (y_true_arr == 1)) / n_aprovados) * 100
            lucro_bons = np.sum(valores_arr[aprovados & (y_true_arr == 0)] * margem_lucro)
            perda_maus = np.sum(valores_arr[aprovados & (y_true_arr == 1)])
            lucro_total = lucro_bons - perda_maus
        else:
            tx_aprov, tx_risco, lucro_total = 0.0, 0.0, 0.0

        taxas_aprovacao.append(tx_aprov)
        riscos_carteira.append(tx_risco)
        lucros.append(lucro_total)

    # 2. Ponto de operação no threshold selecionado
    aprov_op = prob < threshold_otimizado
    tx_aprov_op = (np.sum(aprov_op) / len(prob)) * 100
    tx_risco_op = (np.sum(aprov_op & (y_true_arr == 1)) / max(np.sum(aprov_op), 1)) * 100
    lucro_op = np.sum(valores_arr[aprov_op & (y_true_arr == 0)] * margem_lucro) - np.sum(valores_arr[aprov_op & (y_true_arr == 1)])

    # Console Output Executivo
    print('=' * 80)
    print(f' SIMULAÇÃO DE RESULTADO FINANCEIRO (Threshold = {threshold_otimizado:.4f}) ')
    print('=' * 80)
    print(f' Taxa de Aprovação da Carteira : {tx_aprov_op:.2f}% ({np.sum(aprov_op):,} contratos)')
    print(f' Taxa de Default dos Aprovados : {tx_risco_op:.2f}%')
    print(f' Lucro Financeiro Estimado     : $ {lucro_op:,.2f}')
    print('=' * 80)

    # 3. Plotagem Executiva
    cor_aprov = '#95A5A6'    # Cinza para Aprovação
    cor_risco = '#C00000'    # Vermelho para o Risco
    cor_texto = '#333333'
    cor_fundo = '#FFFFFF'

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(cor_fundo)
    ax1.set_facecolor(cor_fundo)
    ax2 = ax1.twinx()  # Cria eixo Y secundário

    # Linhas de Negócio
    linha1 = ax1.plot(thresholds, taxas_aprovacao, color=cor_aprov, linewidth=3.0, zorder=3, label='Aprovação de Crédito (%)')
    linha2 = ax2.plot(thresholds, riscos_carteira, color=cor_risco, linewidth=3.0, zorder=3, label='Inadimplência Aprovada (%)')

    # Threshold Suavizado (Transparente e Elegante)
    linha_thresh = ax1.axvline(threshold_otimizado, color='#555555', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2, label=f'Threshold ({threshold_otimizado:.4f})')

    # Anotação Refinada focada no Lucro
    max_y = ax1.get_ylim()[1]
    ax1.annotate(
        f'Lucro Estimado:\n$ {lucro_op:,.0f}',
        xy=(threshold_otimizado, max_y * 0.45),
        xytext=(threshold_otimizado + 0.05, max_y * 0.55),
        arrowprops=dict(arrowstyle="-|>", color='#7F8C8D', lw=1.2, connectionstyle="arc3,rad=.2", alpha=0.8),
        fontsize=10.5, fontweight='bold', color='#1A1A1A', alpha=0.9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=cor_fundo, edgecolor='#E5E7E9', alpha=0.8),
        zorder=5
    )

    # Formatação de Eixos
    ax1.set_title('Trade-off Estratégico: Taxa de Concessão vs. Risco da Carteira', fontsize=14, fontweight='bold', color='#1A1A1A', pad=20, loc='left')
    ax1.set_xlabel('Ponto de Corte de Risco (Score Threshold)', fontsize=11, color=cor_texto, labelpad=12)
    
    ax1.set_ylabel('Taxa de Aprovação (%)', color=cor_aprov, fontsize=11, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Taxa de Inadimplência Aprovada (%)', color=cor_risco, fontsize=11, fontweight='bold', labelpad=12)
    
    ax1.tick_params(axis='y', colors=cor_texto, labelsize=10)
    ax2.tick_params(axis='y', colors=cor_texto, labelsize=10)
    ax1.tick_params(axis='x', colors=cor_texto, labelsize=10)

    # Grid e Legenda Despoluída (Jogada para baixo do gráfico para não obstruir as linhas)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7', zorder=0)
    ax1.set_axisbelow(True)
    
    linhas = linha1 + linha2 + [linha_thresh]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10.5, labelcolor=cor_texto)

    # Decluttering (Spines)
    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, right=False)
    for ax in [ax1, ax2]:
        ax.spines['left'].set_color('#E5E7E9')
        ax.spines['bottom'].set_color('#E5E7E9')
        ax.spines['right'].set_color('#E5E7E9')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)

    plt.tight_layout()
    
    if filepath:
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    plt.show()

    df_tradeoff = pd.DataFrame({
        'threshold': thresholds,
        'taxa_aprovacao_pct': taxas_aprovacao,
        'taxa_risco_pct': riscos_carteira,
        'lucro_esperado': lucros
    })

    return df_tradeoff, lucro_op
# =============================================================================
# 15. ENSEMBLE CUSTOMIZADO E EXPORTAÇÃO DE ARTEFATOS
# =============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin


class HardcodedStackingClassifier(ClassifierMixin, BaseEstimator):
    """
    Classificador de Stacking pré-ajustado (zero overhead computacional).
    Combina linearmente as probabilidades previstas por estimadores base pré-treinados
    utilizando pesos estáticos (coeficientes) e intercepto de uma Regressão Logística.
    """
    def __init__(self, estimators, weights, intercept=0.0):
        self.estimators = estimators
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        # Estimadores base e coeficientes já estão definidos estaticamente
        return self

    def predict_proba(self, X):
        prob_cols = []
        for name, est in self.estimators:
            p = est.predict_proba(X)[:, 1]
            prob_cols.append(p)
        
        P = np.column_stack(prob_cols)
        z = self.intercept + P.dot(self.weights)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    # =========================================================================
    # MOCK PARA COMPATIBILIDADE COM PLOTAGEM (Resolve o AttributeError)
    # =========================================================================
    @property
    def final_estimator_(self):
        """
        Cria um objeto falso que simula um meta-learner do Scikit-Learn,
        permitindo que funções de plotagem acessem os pesos e intercepto.
        """
        mock_estimator = types.SimpleNamespace()
        # A função de plotagem espera um array 2D para coef_ (ex: shape (1, n_features))
        mock_estimator.coef_ = np.array([self.weights])
        mock_estimator.intercept_ = np.array([self.intercept])
        return mock_estimator


def extrair_metricas_avaliacao(modelo, X, y, threshold=0.5):
    """
    Calcula dicionário com métricas completas de avaliação em uma partição:
    accuracy, precision, recall, f1_score, roc_auc, pr_auc e ks.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    from scipy.stats import ks_2samp

    y_arr = np.asarray(y)
    probs = modelo.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    pos_probs = probs[y_arr == 1]
    neg_probs = probs[y_arr == 0]
    ks_val = float(ks_2samp(pos_probs, neg_probs).statistic) if len(pos_probs) > 0 and len(neg_probs) > 0 else 0.0

    metricas = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_arr, preds)),
        'precision': float(precision_score(y_arr, preds, zero_division=0)),
        'recall': float(recall_score(y_arr, preds, zero_division=0)),
        'f1_score': float(f1_score(y_arr, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_arr, probs)),
        'pr_auc': float(average_precision_score(y_arr, probs)),
        'ks': float(ks_val)
    }
    return metricas


def exportar_artefatos_modelo(
    nome_modelo,
    modelo,
    diretorio_destino,
    threshold_otimo,
    features_utilizadas,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Exporta o modelo em .joblib e grava os arquivos estruturados
    metadata.json e metrics.json no diretório especificado.
    """
    import joblib
    import json
    from pathlib import Path

    if diretorio_destino is None:
        dest = ARTIFACTS_DIR
    else:
        dest = Path(diretorio_destino)
        if not dest.is_absolute():
            dest = BASE_DIR / dest
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Serialização do estimador
    arquivo_modelo = dest / f"{nome_modelo}.joblib"
    joblib.dump(modelo, arquivo_modelo)

    # 2. Extração dos hiperparâmetros
    hiperparametros = {}
    if hasattr(modelo, 'get_params'):
        try:
            params_raw = modelo.get_params(deep=False)
            for k, v in params_raw.items():
                if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                    hiperparametros[k] = v
                else:
                    hiperparametros[k] = str(v)
        except Exception:
            hiperparametros = {'info': 'Nao foi possivel serializar get_params()'}

    # 3. Geração do metadata.json
    metadata = {
        'nome_modelo': nome_modelo,
        'classe': f"{modelo.__class__.__module__}.{modelo.__class__.__name__}",
        'hiperparametros': hiperparametros,
        'features_utilizadas': list(features_utilizadas),
        'total_features': len(features_utilizadas),
        'tamanho_historico_treino': {
            'linhas': int(len(X_train)),
            'taxa_inadimplencia': float(y_train.mean())
        },
        'tamanho_historico_teste': {
            'linhas': int(len(X_test)),
            'taxa_inadimplencia': float(y_test.mean())
        }
    }

    arquivo_metadata = dest / f"{nome_modelo}_metadata.json"
    with open(arquivo_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 4. Geração do metrics.json
    metricas_treino = extrair_metricas_avaliacao(modelo, X_train, y_train, threshold=threshold_otimo)
    metricas_teste = extrair_metricas_avaliacao(modelo, X_test, y_test, threshold=threshold_otimo)

    metrics_payload = {
        'nome_modelo': nome_modelo,
        'threshold_otimo': float(threshold_otimo),
        'metricas_treino': metricas_treino,
        'metricas_teste': metricas_teste
    }

    arquivo_metrics = dest / f"{nome_modelo}_metrics.json"
    with open(arquivo_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"Artefatos de {nome_modelo} exportados com sucesso em {dest}:")
    print(f" - {arquivo_modelo.name}")
    print(f" - {arquivo_metadata.name}")
    print(f" - {arquivo_metrics.name}")

def encontrar_threshold_otimo_oof(probs_oof, y_true, valores_credito, margem_lucro=0.15):
    """
    Encontra o threshold de corte que maximiza o lucro economico nas previsoes Out-Of-Fold (OOF).
    """
    thresholds_teste = np.linspace(0.01, 0.99, 100)
    y_true_arr = np.asarray(y_true)
    valores_arr = np.asarray(valores_credito)
    lucros = []

    for t in thresholds_teste:
        aprovados_bons = (probs_oof < t) & (y_true_arr == 0)
        aprovados_maus = (probs_oof < t) & (y_true_arr == 1)
        lucro_bons = np.sum(valores_arr[aprovados_bons] * margem_lucro)
        prejuizo_maus = np.sum(valores_arr[aprovados_maus])
        lucros.append(lucro_bons - prejuizo_maus)

    idx_otimo = np.argmax(lucros)
    return thresholds_teste[idx_otimo], lucros[idx_otimo]


class RankAveragingEnsemble(ClassifierMixin, BaseEstimator):
    """
    Ensemble baseado na media de ranks das probabilidades dos estimadores base.
    """
    _estimator_type = "classifier"

    def __init__(self, estimators=None, weights=None, refit_base=False):
        self.estimators = estimators
        self.weights = weights
        self.refit_base = refit_base
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self):
        from sklearn.base import ClassifierTags, TargetTags
        tags = super().__sklearn_tags__() if hasattr(super(), '__sklearn_tags__') else None
        if tags is not None:
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            tags.target_tags = TargetTags(single_output=True)
            return tags
        return tags

    def fit(self, X, y):
        from sklearn.base import clone
        if self.refit_base:
            # Clona e retreina cada estimador base nos dados do fold (CV honesto)
            self.estimators_ = [
                (nome, clone(est).fit(X, y)) for nome, est in self.estimators
            ]
        else:
            # Usa os modelos ja treinados
            self.estimators_ = list(self.estimators)
        return self

    def _rank_normalize(self, p):
        from scipy.stats import rankdata
        return rankdata(p) / len(p)

    def predict_proba(self, X):
        probs = [est.predict_proba(X)[:, 1] for _, est in self.estimators_]
        ranks = np.column_stack([self._rank_normalize(p) for p in probs])

        if self.weights is None:
            w = np.ones(ranks.shape[1]) / ranks.shape[1]
        else:
            w = np.asarray(self.weights, dtype=float)
            w = w / w.sum()

        score = np.average(ranks, axis=1, weights=w)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class EnsembleCalibradoWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper que encapsula um Ensemble e uma Regressão Isotônica já ajustada.
    Garante que as previsões do ensemble passem pela curva de calibração sem Data Leakage.
    """
    def __init__(self, ensemble_base, reg_isotonica):
        self.ensemble_base = ensemble_base
        self.iso = reg_isotonica
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y=None):
        return self
        
    def predict_proba(self, X):
        p_raw = self.ensemble_base.predict_proba(X)[:, 1]
        p_cal = np.clip(self.iso.predict(p_raw), 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


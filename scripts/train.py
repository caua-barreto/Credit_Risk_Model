"""
train.py — Treinamento, otimizacao de thresholds, poda SHAP e serializacao de artefatos.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Resolucao dinamica da raiz do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_utils import (
    HardcodedStackingClassifier,
    exportar_artefatos_modelo,
    podar_features_por_shap,
    otimizar_thresholds_por_lucro,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

# Hiperparametros campeoes de referencia (validados no 03_Modeling.ipynb)
DEFAULT_LGBM_PARAMS = {
    'n_estimators': 182,
    'learning_rate': 0.07930064881300683,
    'num_leaves': 23,
    'max_depth': 5,
    'subsample': 0.8707367972152678,
    'colsample_bytree': 0.6364467981453202,
    'reg_alpha': 0.010433124771633521,
    'reg_lambda': 1.2092712649790147,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 312,
    'learning_rate': 0.09986552891346542,
    'max_depth': 3,
    'subsample': 0.7705613519160045,
    'colsample_bytree': 0.847731022252971,
    'reg_alpha': 6.30507511527427,
    'reg_lambda': 0.03202991450856441,
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

DEFAULT_CAT_PARAMS = {
    'iterations': 243,
    'learning_rate': 0.09698383256576704,
    'depth': 5,
    'l2_leaf_reg': 6.969367542097534,
    'auto_class_weights': 'Balanced',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1,
}

# Features identificadas para poda SHAP (top_n=8)
DEFAULT_FEATURES_PODA = [
    'Flag_Sobre_Endividamento_Aberto',
    'Flag_First_Payment_Default',
    'credacc_transactions_402L_sum',
    'applicationscnt_629L',
    'Mes_do_Ano_3',
    'Mes_do_Ano_4',
    'Mes_do_Ano_9',
    'Mes_do_Ano_2',
]


def carregar_dados(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega as bases processadas em formato Parquet."""
    train_path = data_dir / "df_train.parquet"
    test_path = data_dir / "df_test.parquet"
    oot_path = data_dir / "df_oot.parquet"

    logger.info("Carregando datasets processados de: %s", data_dir)
    df_train = pd.read_parquet(train_path, engine="pyarrow")
    df_test = pd.read_parquet(test_path, engine="pyarrow")
    df_oot = pd.read_parquet(oot_path, engine="pyarrow")
    return df_train, df_test, df_oot


def separar_features_target(
    df: pd.DataFrame,
    colunas_ignorar: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variaveis explicativas e target."""
    if colunas_ignorar is None:
        colunas_ignorar = ['target', 'case_id', 'date_decision', 'MONTH', 'WEEK_NUM']
    features = [c for c in df.columns if c not in colunas_ignorar]
    return df[features], df['target']


def main():
    parser = argparse.ArgumentParser(description="Treinamento e exportacao de modelos de risco de credito.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Diretorio contendo os arquivos df_train.parquet e df_test.parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretorio para persistencia dos modelos e metadados.",
    )
    parser.add_argument(
        "--executar-shap",
        action="store_true",
        help="Executa rotina dinamica de poda SHAP antes do retreino.",
    )
    parser.add_argument(
        "--top-n-remover",
        type=int,
        default=8,
        help="Quantidade de variaveis mais fracas a remover via SHAP.",
    )
    parser.add_argument(
        "--threshold-lgbm",
        type=float,
        default=0.7425,
        help="Threshold otimizado de negocio para LightGBM.",
    )
    parser.add_argument(
        "--threshold-stacking",
        type=float,
        default=0.7920,
        help="Threshold otimizado para Stacking Ensemble.",
    )
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test, _ = carregar_dados(args.data_dir)
    X_train, y_train = separar_features_target(df_train)
    X_test, y_test = separar_features_target(df_test)

    taxa_desbalanceamento = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("Taxa de desbalanceamento calculada: %.2f", taxa_desbalanceamento)

    # 1. Poda dimensional via SHAP
    features_a_remover = DEFAULT_FEATURES_PODA
    if args.executar_shap:
        logger.info("Executando analise SHAP dinamica em amostra de teste...")
        lgbm_pre = LGBMClassifier(**DEFAULT_LGBM_PARAMS)
        lgbm_pre.fit(X_train, y_train)
        X_amostra_shap = X_test.sample(n=min(10000, len(X_test)), random_state=42)
        cortes = podar_features_por_shap(
            {'LightGBM_Opt': lgbm_pre},
            X_train, y_train, X_test, y_test, X_amostra_shap,
            top_n_remover=args.top_n_remover,
        )
        features_a_remover = cortes.get('LightGBM_Opt', DEFAULT_FEATURES_PODA)

    logger.info("Features podadas via SHAP: %s", features_a_remover)
    features_lite = [c for c in X_train.columns if c not in features_a_remover]
    logger.info("Total de features selecionadas: %d", len(features_lite))

    X_train_lite = X_train[features_lite]
    X_test_lite = X_test[features_lite]

    # 2. Treino LightGBM_Opt_Lite (Modelo Campeao de Negocio)
    logger.info("Treinando LightGBM_Opt_Lite...")
    modelo_lgbm_lite = LGBMClassifier(**DEFAULT_LGBM_PARAMS)
    modelo_lgbm_lite.fit(X_train_lite, y_train)

    # 3. Treino Estimadores Base e Meta-Learner para Stacking Ensemble
    logger.info("Treinando estimadores base para Stacking Ensemble...")
    xgb_params = DEFAULT_XGB_PARAMS.copy()
    xgb_params['scale_pos_weight'] = taxa_desbalanceamento
    modelo_xgb_lite = XGBClassifier(**xgb_params)
    modelo_xgb_lite.fit(X_train_lite, y_train)

    modelo_cat_lite = CatBoostClassifier(**DEFAULT_CAT_PARAMS)
    modelo_cat_lite.fit(X_train_lite, y_train)

    estimadores_base_lite = [
        ('lgbm', modelo_lgbm_lite),
        ('xgb',  modelo_xgb_lite),
        ('cat',  modelo_cat_lite),
    ]

    P_train_lite = np.column_stack([est.predict_proba(X_train_lite)[:, 1] for _, est in estimadores_base_lite])
    meta_learner_lite = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    meta_learner_lite.fit(P_train_lite, y_train)

    modelo_stacking_lite = HardcodedStackingClassifier(
        estimators=estimadores_base_lite,
        weights=meta_learner_lite.coef_.ravel(),
        intercept=meta_learner_lite.intercept_[0],
    )

    # 4. Exportacao dos Artefatos Oficiais
    logger.info("Exportando artefatos para %s...", args.artifacts_dir)
    exportar_artefatos_modelo(
        nome_modelo='LightGBM_Opt_Lite',
        modelo=modelo_lgbm_lite,
        diretorio_destino=args.artifacts_dir,
        threshold_otimo=args.threshold_lgbm,
        features_utilizadas=features_lite,
        X_train=X_train_lite,
        y_train=y_train,
        X_test=X_test_lite,
        y_test=y_test,
    )

    exportar_artefatos_modelo(
        nome_modelo='Stacking_Ensemble_Lite',
        modelo=modelo_stacking_lite,
        diretorio_destino=args.artifacts_dir,
        threshold_otimo=args.threshold_stacking,
        features_utilizadas=features_lite,
        X_train=X_train_lite,
        y_train=y_train,
        X_test=X_test_lite,
        y_test=y_test,
    )

    logger.info("Pipeline de treinamento finalizado com sucesso.")


if __name__ == "__main__":
    main()

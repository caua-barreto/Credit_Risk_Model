"""
train.py — Treinamento, calibração isotônica, otimização de threshold e exportação de artefatos.

Reproduz estritamente a metodologia validada em 03_Modeling.ipynb:
  1. Carrega dados de Treino e Teste em data/processed/
  2. Aplica a seleção/poda das 33 variáveis homologadas (LITE)
  3. Treina o LGBMClassifier com os hiperparâmetros campeões do Optuna
  4. Realiza a Calibração Isotônica via CalibratedClassifierCV (cv=3 estratificado)
  5. Determina o Threshold Ótimo por matriz de ganho/perda financeira
  6. Exporta artefatos oficiais padronizados:
     - artifacts/modelo.joblib
     - artifacts/modelo_metadata.json
     - artifacts/modelo_metrics.json
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
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Resolução dinâmica da raiz do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_utils import (
    exportar_artefatos_modelo,
    encontrar_threshold_otimo_oof,
    podar_features_por_shap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

# Hiperparâmetros campeões validados em 03_Modeling.ipynb (Optuna)
DEFAULT_LGBM_PARAMS = {
    'n_estimators': 168,
    'learning_rate': 0.04061350322079583,
    'num_leaves': 51,
    'max_depth': 9,
    'subsample': 0.9486424242315596,
    'colsample_bytree': 0.878224877398664,
    'reg_alpha': 6.032518729548217,
    'reg_lambda': 0.017424838010587338,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
}

# 10 features podadas pelo SHAP conforme Seção 7.4 do 03_Modeling.ipynb
DEFAULT_FEATURES_PODA = [
    'Flag_Sobre_Endividamento_Aberto',
    'Flag_First_Payment_Default',
    'Mes_do_Ano_4',
    'credacc_transactions_402L_sum',
    'Mes_do_Ano_7',
    'Mes_do_Ano_9',
    'Mes_do_Ano_3',
    'Mes_do_Ano_2',
    'clientscnt_887L',
    'numinstlswithdpd5_4187116L',
]

COLUNAS_IGNORAR = ['target', 'case_id', 'date_decision', 'MONTH', 'WEEK_NUM']


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
    """Separa variáveis explicativas e target."""
    if colunas_ignorar is None:
        colunas_ignorar = COLUNAS_IGNORAR
    features = [c for c in df.columns if c not in colunas_ignorar]
    return df[features], df['target'].astype('int8')


def treinar_e_exportar(
    data_dir: Path = BASE_DIR / "data" / "processed",
    artifacts_dir: Path = BASE_DIR / "artifacts",
    executar_shap: bool = False,
    top_n_remover: int = 10,
    margem_lucro: float = 0.15,
) -> tuple[CalibratedClassifierCV, float, List[str]]:
    """
    Executa o treinamento oficial do LightGBM LITE Calibrado,
    otimiza o threshold financeiro OOF e exporta os artefatos.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test, _ = carregar_dados(data_dir)
    X_train, y_train = separar_features_target(df_train)
    X_test, y_test = separar_features_target(df_test)

    todas_features = list(X_train.columns)
    logger.info("Total de features iniciais pós-eng: %d", len(todas_features))

    # 1. Poda dimensional SHAP
    features_a_remover = DEFAULT_FEATURES_PODA
    if executar_shap:
        logger.info("Executando análise SHAP dinâmica em amostra de teste...")
        lgbm_pre = LGBMClassifier(**DEFAULT_LGBM_PARAMS)
        lgbm_pre.fit(X_train, y_train)
        X_amostra_shap = X_test.sample(n=min(20000, len(X_test)), random_state=42)
        cortes = podar_features_por_shap(
            {'LightGBM_Opt': lgbm_pre},
            X_train, y_train, X_test, y_test, X_amostra_shap,
            top_n_remover=top_n_remover,
        )
        features_a_remover = cortes.get('LightGBM_Opt', DEFAULT_FEATURES_PODA)

    features_lite = [c for c in todas_features if c not in features_a_remover]
    logger.info("Features podadas (%d): %s", len(features_a_remover), features_a_remover)
    logger.info("Features homologadas LITE: %d variáveis", len(features_lite))

    X_train_lite = X_train[features_lite]
    X_test_lite = X_test[features_lite]

    # 2. Treino do estimador base LightGBM LITE
    logger.info("Treinando LightGBM_Opt_Lite base...")
    base_lgbm = LGBMClassifier(**DEFAULT_LGBM_PARAMS)
    base_lgbm.fit(X_train_lite, y_train)

    # 3. Calibração Isotônica via Validação Cruzada (cv=3 estratificado)
    logger.info("Calibrando LightGBM_Opt_Lite via Regressão Isotônica (CalibratedClassifierCV, cv=3)...")
    cv_calib = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    modelo_campeao = CalibratedClassifierCV(
        estimator=base_lgbm,
        method='isotonic',
        cv=cv_calib,
    )
    modelo_campeao.fit(X_train_lite, y_train)

    # 4. Busca do Threshold Ótimo por Lucro Máximo (Out-of-Fold)
    logger.info("Otimizando threshold financeiro com base nas probabilidades calibradas OOF...")
    probs_oof_calibradas = cross_val_predict(
        modelo_campeao,
        X_train_lite,
        y_train,
        cv=cv_calib,
        method='predict_proba',
        n_jobs=-1,
    )[:, 1]

    valores_credito_train = X_train['credamount_770A'].values
    threshold_otimo, lucro_oof_max = encontrar_threshold_otimo_oof(
        probs_oof=probs_oof_calibradas,
        y_true=y_train,
        valores_credito=valores_credito_train,
        margem_lucro=margem_lucro,
    )
    logger.info("Threshold ótimo encontrado: %.4f | Lucro OOF Max: $ %s", threshold_otimo, f"{lucro_oof_max:,.2f}")

    # 5. Exportação dos Artefatos Oficiais
    logger.info("Exportando artefatos oficiais para: %s...", artifacts_dir)
    exportar_artefatos_modelo(
        nome_modelo='modelo',
        modelo=modelo_campeao,
        diretorio_destino=artifacts_dir,
        threshold_otimo=float(threshold_otimo),
        features_utilizadas=features_lite,
        X_train=X_train_lite,
        y_train=y_train,
        X_test=X_test_lite,
        y_test=y_test,
    )

    logger.info("Pipeline de treinamento e serialização concluído com sucesso.")
    return modelo_campeao, float(threshold_otimo), features_lite


def main():
    parser = argparse.ArgumentParser(description="Treinamento e exportação do modelo campeão de risco de crédito.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Diretório contendo os arquivos df_train.parquet e df_test.parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretório para persistência dos modelos e metadados.",
    )
    parser.add_argument(
        "--executar-shap",
        action="store_true",
        help="Executa rotina dinâmica de poda SHAP antes do retreino.",
    )
    parser.add_argument(
        "--top-n-remover",
        type=int,
        default=10,
        help="Quantidade de variáveis mais fracas a remover via SHAP.",
    )
    parser.add_argument(
        "--margem-lucro",
        type=float,
        default=0.15,
        help="Margem de ganho líquido sobre o crédito adimplente.",
    )
    args = parser.parse_args()

    treinar_e_exportar(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        executar_shap=args.executar_shap,
        top_n_remover=args.top_n_remover,
        margem_lucro=args.margem_lucro,
    )


if __name__ == "__main__":
    main()

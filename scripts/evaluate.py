"""
evaluate.py — Avaliacao em bases Teste e OOT, metricas de performance e extracao de drift (PSI).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Resolucao dinamica da raiz do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.drift import (
    calcular_psi,
    relatorio_feature_drift_psi,
    calcular_psi_scores,
    teste_qui_quadrado_temporal,
)
from src.model_utils import extrair_metricas_avaliacao

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def carregar_modelo_e_metadados(artifacts_dir: Path, nome_modelo: str):
    """Carrega o estimador serializado e o respectivo arquivo metadata.json."""
    caminho_modelo = artifacts_dir / f"{nome_modelo}.joblib"
    caminho_metadata = artifacts_dir / f"{nome_modelo}_metadata.json"

    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {caminho_modelo}")
    if not caminho_metadata.exists():
        raise FileNotFoundError(f"Metadados nao encontrados: {caminho_metadata}")

    modelo = joblib.load(caminho_modelo)
    with open(caminho_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, metadata


def main():
    parser = argparse.ArgumentParser(description="Avaliacao de modelos e extracao de drift temporal.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Diretorio contendo os arquivos df_test.parquet e df_oot.parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretorio de artefatos onde o modelo e metadados estao salvos.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=BASE_DIR / "reports",
        help="Diretorio para salvar relatorios e metricas.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="LightGBM_Opt_Lite",
        help="Nome base do modelo a avaliar (ex: LightGBM_Opt_Lite).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold de decisao (se omitido, le do metrics.json ou usa 0.5).",
    )
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carga do Modelo e Metadados
    modelo, metadata = carregar_modelo_e_metadados(args.artifacts_dir, args.model_name)
    features = metadata.get("features_utilizadas", [])
    logger.info("Modelo %s carregado com %d features mapeadas.", args.model_name, len(features))

    # Threshold de corte
    threshold = args.threshold
    if threshold is None:
        caminho_metrics = args.artifacts_dir / f"{args.model_name}_metrics.json"
        if caminho_metrics.exists():
            with open(caminho_metrics, "r", encoding="utf-8") as f:
                dados_metrics = json.load(f)
                threshold = dados_metrics.get("threshold_otimo", 0.5)
        else:
            threshold = 0.5
    logger.info("Threshold adotado para avaliacao: %.4f", threshold)

    # 2. Carga dos dados de Teste e OOT
    df_test = pd.read_parquet(args.data_dir / "df_test.parquet", engine="pyarrow")
    df_oot = pd.read_parquet(args.data_dir / "df_oot.parquet", engine="pyarrow")

    X_test = df_test[features]
    y_test = df_test["target"]

    X_oot = df_oot[features]
    y_oot = df_oot["target"]

    # 3. Metricas de Performance
    logger.info("Calculando metricas na base Teste...")
    metricas_teste = extrair_metricas_avaliacao(modelo, X_test, y_test, threshold=threshold)
    logger.info("Calculando metricas na base OOT...")
    metricas_oot = extrair_metricas_avaliacao(modelo, X_oot, y_oot, threshold=threshold)

    # 4. Extracao de Drift via src.drift
    logger.info("Gerando relatorio de Feature Drift (PSI) entre Teste e OOT...")
    df_drift_features = relatorio_feature_drift_psi(X_test, X_oot, bins=10)

    prob_test = modelo.predict_proba(X_test)[:, 1]
    prob_oot = modelo.predict_proba(X_oot)[:, 1]
    psi_scores = calcular_psi_scores(prob_test, prob_oot, bins=10)
    logger.info("PSI dos Scores Preditos (Teste vs OOT): %.4f", psi_scores)

    # 5. Exportacao do relatorio consolidado em reports/drift_reports.csv
    caminho_drift = args.reports_dir / "drift_reports.csv"
    df_drift_features.to_csv(caminho_drift, index=False)
    logger.info("Relatorio de drift salvo com sucesso em: %s", caminho_drift)

    # 6. Sumario de Avaliacao
    caminho_eval_summary = args.reports_dir / f"evaluation_{args.model_name}.json"
    resumo_avaliacao = {
        "modelo": args.model_name,
        "threshold": threshold,
        "metricas_teste": metricas_teste,
        "metricas_oot": metricas_oot,
        "psi_score_distribuicao": float(psi_scores),
    }
    with open(caminho_eval_summary, "w", encoding="utf-8") as f:
        json.dump(resumo_avaliacao, f, indent=2, ensure_ascii=False)
    logger.info("Sumario de avaliacao salvo em: %s", caminho_eval_summary)


if __name__ == "__main__":
    main()

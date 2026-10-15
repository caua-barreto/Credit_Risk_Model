"""
predict.py — Script de inferencia em lote (batch scoring).
Recebe um CSV com dados de entrada via CLI, aplica transformacoes, carrega modelo
e gera predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Resolucao dinamica da raiz do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import (
    criar_shadow_features,
    criar_ratios,
    criar_dpd_momentum,
    criar_intensidade,
    criar_flags_risco_critico,
    criar_features_linha,
    imputar_zero_logico,
    aplicar_ohe,
)
from scripts.build_features import CreditRiskPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")


def carregar_modelo_e_features(artifacts_dir: Path, nome_modelo: str):
    """Carrega o estimador salvo e os metadados contendo as features obrigatorias."""
    caminho_modelo = artifacts_dir / f"{nome_modelo}.joblib"
    caminho_metadata = artifacts_dir / f"{nome_modelo}_metadata.json"

    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Arquivo do modelo nao encontrado: {caminho_modelo}")

    modelo = joblib.load(caminho_modelo)

    features = None
    if caminho_metadata.exists():
        with open(caminho_metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            features = metadata.get("features_utilizadas", None)

    return modelo, features


def carregar_ou_aplicar_pipeline(
    df: pd.DataFrame,
    pipeline_path: Optional[Path],
) -> pd.DataFrame:
    """
    Aplica o pipeline serializado CreditRiskPipeline caso disponivel;
    caso contrario, aplica transformacoes deterministicas stateless.
    """
    if pipeline_path is not None and pipeline_path.exists():
        logger.info("Carregando pipeline de transformacao salvo em %s...", pipeline_path)
        pipe = CreditRiskPipeline.load(pipeline_path)
        return pipe.transform(df)

    logger.info("Aplicando funcoes deterministicas de feature engineering (stateless)...")
    df_out = df.copy()
    df_out = criar_shadow_features(df_out)
    df_out = criar_ratios(df_out)
    df_out = criar_intensidade(df_out)
    df_out = criar_flags_risco_critico(df_out)
    df_out = criar_features_linha(df_out)
    df_out = imputar_zero_logico(df_out)
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Inferencia em lote (Credit Risk Predictions).")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Caminho do arquivo CSV de entrada com os dados dos proponentes.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "predictions.csv",
        help="Caminho do arquivo CSV de saida com as predicoes geradas.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretorio contendo os modelos serializados.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="modelo",
        help="Nome do modelo a ser utilizado para inferencia (padrao: modelo).",
    )
    parser.add_argument(
        "--pipeline-path",
        type=Path,
        default=BASE_DIR / "artifacts" / "feature_pipeline.joblib",
        help="Caminho do pipeline de features serializado.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold de corte para decisao binaria (se omitido, le do metrics.json ou usa 0.5).",
    )
    args = parser.parse_args()

    # 1. Leitura dos dados de entrada
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {args.input_csv}")

    logger.info("Lendo arquivo de entrada: %s", args.input_csv)
    df_raw = pd.read_csv(args.input_csv)
    case_ids = df_raw["case_id"] if "case_id" in df_raw.columns else pd.Series(range(len(df_raw)), name="case_id")

    # 2. Carga do Modelo e Metadados
    modelo, features_esperadas = carregar_modelo_e_features(args.artifacts_dir, args.model_name)

    # Threshold de decisao
    threshold = args.threshold
    if threshold is None:
        caminho_metrics = args.artifacts_dir / f"{args.model_name}_metrics.json"
        if caminho_metrics.exists():
            with open(caminho_metrics, "r", encoding="utf-8") as f:
                dados_metrics = json.load(f)
                threshold = dados_metrics.get("threshold_otimo", 0.5)
        else:
            threshold = 0.5
    logger.info("Utilizando threshold: %.4f", threshold)

    # 3. Aplicacao do pipeline de transformacao
    df_proc = carregar_ou_aplicar_pipeline(df_raw, args.pipeline_path)

    # 4. Alinhamento de colunas com o modelo treinado
    if features_esperadas is not None:
        # Garante presenca de todas as colunas necessarias
        for col in features_esperadas:
            if col not in df_proc.columns:
                df_proc[col] = 0.0
        X = df_proc[features_esperadas]
    else:
        colunas_ignorar = ['target', 'case_id', 'date_decision', 'MONTH', 'WEEK_NUM']
        features = [c for c in df_proc.columns if c not in colunas_ignorar]
        X = df_proc[features]

    # 5. Inferencia
    logger.info("Executando inferencia para %d proponentes...", len(X))
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X)[:, 1]
    else:
        probabilidades = modelo.predict(X)

    decisoes = (probabilidades >= threshold).astype(int)

    # 6. Montagem e Exportacao das Predicoes
    df_resultado = pd.DataFrame({
        "case_id": case_ids,
        "score_inadimplencia": probabilidades,
        "previsao_inadimplente": decisoes,
        "decisao_credito": np.where(decisoes == 1, "Reprovado", "Aprovado"),
    })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.output_csv, index=False)
    logger.info("Predicoes geradas com sucesso em: %s (shape: %s)", args.output_csv, df_resultado.shape)


if __name__ == "__main__":
    main()

"""
evaluate.py — Avaliação completa do modelo em bases Teste e OOT, geração de relatórios e métricas MLOps.

Reproduz estritamente todas as análises de 04_Model_Avaliation.ipynb:
  1. Performance estática: Relatório Treino x OOT (Accuracy, Precision, Recall, F1-SCORE, ROC-AUC, PR-AUC, KS)
  2. Gráficos corporativos:
     - Curva ROC isolada
     - Distribuição KDE dos scores com threshold ótimo vertical
     - CDF empírica K-S com ponto de separação máxima
  3. Análise de Drift:
     - Target Drift safra a safra (CSV + gráfico)
     - Feature Drift PSI (CSV + gráfico do Top 3)
     - Score PSI e teste KS de deslocamento
  4. Estabilidade Temporal (GINI por safra) com axvspan na zona OOT + Teste Qui-Quadrado (chi2)
  5. Tabela de decis e K-S máximo (Cumulative Gains e Lift)
  6. Simulação financeira de trade-off de concessão e rentabilidade safra a safra
  7. Exportação automatizada de métricas contínuas para artifacts/monitoring_metrics_oot.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score

# Resolução dinâmica da raiz do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.drift import (
    calcular_psi,
    calcular_psi_scores,
    relatorio_feature_drift_psi,
    teste_qui_quadrado_temporal,
)
from src.plots import (
    plot_curva_roc,
    plot_distribuicao_kde_scores,
    plot_cdf_ks,
    plot_target_drift_mensal,
    plot_comparativo_psi_features,
    plot_matriz_confusao,
    plot_estabilidade_gini_temporal,
)
from src.model_utils import (
    gerar_tabela_decis_ks,
    simular_lucro_e_curva_tradeoff,
    extrair_metricas_avaliacao,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def carregar_modelo_e_metadados(artifacts_dir: Path, nome_modelo: str = "modelo"):
    """Carrega o estimador serializado e o respectivo arquivo metadata.json."""
    caminho_modelo = artifacts_dir / f"{nome_modelo}.joblib"
    caminho_metadata = artifacts_dir / f"{nome_modelo}_metadata.json"
    caminho_metrics = artifacts_dir / f"{nome_modelo}_metrics.json"

    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")
    if not caminho_metadata.exists():
        raise FileNotFoundError(f"Metadados não encontrados em: {caminho_metadata}")

    modelo = joblib.load(caminho_modelo)
    with open(caminho_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metrics = {}
    if caminho_metrics.exists():
        with open(caminho_metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return modelo, metadata, metrics


def executar_avaliacao_completa(
    data_dir: Path = BASE_DIR / "data" / "processed",
    artifacts_dir: Path = BASE_DIR / "artifacts",
    reports_dir: Path = BASE_DIR / "reports",
    nome_modelo: str = "modelo",
    threshold: Optional[float] = None,
    margem_lucro: float = 0.15,
    gerar_graficos: bool = True,
) -> Dict:
    """
    Executa a suíte completa de avaliação OOT, exportando relatórios CSV,
    figuras PNG e o JSON de monitoramento MLOps.
    """
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carga de Artefatos
    modelo, metadata, metrics_armazenadas = carregar_modelo_e_metadados(artifacts_dir, nome_modelo)
    features = metadata.get("features_utilizadas", [])
    logger.info("Modelo carregado com %d features homologadas.", len(features))

    if threshold is None:
        threshold = float(metrics_armazenadas.get("threshold_otimo", 0.1189))
    logger.info("Threshold financeiro congelado: %.4f", threshold)

    # 2. Carga dos Conjuntos de Dados
    df_train = pd.read_parquet(data_dir / "df_train.parquet", engine="pyarrow")
    df_test = pd.read_parquet(data_dir / "df_test.parquet", engine="pyarrow")
    df_oot = pd.read_parquet(data_dir / "df_oot.parquet", engine="pyarrow")

    X_train = df_train[features]
    y_train = df_train["target"].astype("int8")
    X_test = df_test[features]
    y_test = df_test["target"].astype("int8")
    X_oot = df_oot[features]
    y_oot = df_oot["target"].astype("int8")

    # Escoragem
    df_test["score_prob"] = modelo.predict_proba(X_test)[:, 1]
    df_oot["score_prob"] = modelo.predict_proba(X_oot)[:, 1]
    df_train["score_prob"] = modelo.predict_proba(X_train)[:, 1]

    # =========================================================================
    # 2.0 PERFORMANCE ESTÁTICA: RELATÓRIO TREINO X OOT
    # =========================================================================
    logger.info("Computando Relatório Treino x OOT...")
    if "metricas_treino" in metrics_armazenadas:
        metricas_treino = metrics_armazenadas["metricas_treino"]
    else:
        metricas_treino = extrair_metricas_avaliacao(modelo, X_train, y_train, threshold=threshold)

    metricas_oot = extrair_metricas_avaliacao(modelo, X_oot, y_oot, threshold=threshold)

    nomes_metricas = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-SCORE"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("ks", "KS"),
    ]

    linhas_relatorio = []
    for chave, nome_display in nomes_metricas:
        val_tr = metricas_treino[chave]
        val_oot = metricas_oot[chave]
        delta = val_oot - val_tr
        pct_var = (delta / val_tr * 100) if val_tr != 0 else 0.0
        linhas_relatorio.append({
            "Metric": nome_display,
            "Treino": val_tr,
            "OOT": val_oot,
            "Delta": delta,
            "Delta (%)": pct_var,
        })

    df_treino_oot = pd.DataFrame(linhas_relatorio)
    caminho_csv_relatorio = metrics_dir / "relatorio_treino_x_oot.csv"
    df_treino_oot.to_csv(caminho_csv_relatorio, index=False)
    logger.info("Relatório Treino x OOT salvo em: %s", caminho_csv_relatorio)

    if gerar_graficos:
        logger.info("Gerando gráficos de ROC, KDE de Scores e CDF K-S...")
        plot_curva_roc(
            y_test=y_test,
            prob_test=df_test["score_prob"],
            y_oot=y_oot,
            prob_oot=df_oot["score_prob"],
            filepath=figures_dir / "curva_roc_isolada.png",
        )
        plot_distribuicao_kde_scores(
            y_test=y_test,
            prob_test=df_test["score_prob"],
            y_oot=y_oot,
            prob_oot=df_oot["score_prob"],
            threshold=threshold,
            filepath=figures_dir / "distribuicao_kde_comparativo_threshold.png",
        )
        plot_cdf_ks(
            y_test=y_test,
            prob_test=df_test["score_prob"],
            y_oot=y_oot,
            prob_oot=df_oot["score_prob"],
            filepath=figures_dir / "cdf_ks_comparativo.png",
        )

    # =========================================================================
    # 3.0 ANÁLISE DE DRIFT (ESTABILIDADE POPULACIONAL)
    # =========================================================================
    logger.info("Analisando Target Drift e Feature Drift (PSI)...")
    df_target_drift = plot_target_drift_mensal(
        df_test=df_test,
        df_oot=df_oot,
        col_mes="MONTH",
        col_target="target",
        filepath=figures_dir / "target_drift_safra.png" if gerar_graficos else None,
    )
    df_target_drift.to_csv(metrics_dir / "target_drift_safra.csv", index=False)

    df_psi_features = relatorio_feature_drift_psi(X_test, X_oot, bins=10)
    df_psi_features.to_csv(metrics_dir / "psi_features_report.csv", index=False)

    df_psi_scores_tab, psi_geral = calcular_psi_scores(df_test["score_prob"], df_oot["score_prob"], bins=10)
    ks_shift_res = ks_2samp(df_test["score_prob"], df_oot["score_prob"])
    ks_shift_stat = float(ks_shift_res.statistic)
    p_val_shift = float(ks_shift_res.pvalue)

    psi_por_feature = {}
    for _, row in df_psi_features.iterrows():
        try:
            psi_por_feature[str(row["Variável"])] = float(row["PSI"])
        except Exception:
            pass

    if gerar_graficos:
        df_psi_features_num = df_psi_features.copy()
        df_psi_features_num["PSI_NUM"] = pd.to_numeric(df_psi_features_num["PSI"], errors="coerce")
        top3_features = df_psi_features_num.sort_values(by="PSI_NUM", ascending=False).head(3)["Variável"].tolist()
        plot_comparativo_psi_features(
            X_test=X_test,
            X_oot=X_oot,
            features_list=top3_features,
            filepath=figures_dir / "feature_drift_top3_psi.png",
        )

    # =========================================================================
    # 4.0 ESTABILIDADE TEMPORAL (GINI POR SAFRA) + TESTE QUI-QUADRADO
    # =========================================================================
    logger.info("Computando série temporal de GINI e Teste Qui-Quadrado...")
    df_hist_scores = pd.concat([
        df_train[["MONTH", "target", "score_prob"]],
        df_test[["MONTH", "target", "score_prob"]],
    ], axis=0, ignore_index=True)
    df_oot_scores = df_oot[["MONTH", "target", "score_prob"]].copy()

    safras_hist = sorted(df_hist_scores["MONTH"].unique())
    safras_oot = sorted(df_oot_scores["MONTH"].unique())

    registros_gini = []
    for mes in safras_hist:
        df_m = df_hist_scores[df_hist_scores["MONTH"] == mes]
        if df_m["target"].nunique() > 1:
            auc = roc_auc_score(df_m["target"], df_m["score_prob"])
            registros_gini.append({
                "MONTH": int(mes),
                "Partição": "Histórico (Treino + Teste)",
                "Volume": len(df_m),
                "Inadimplência (%)": df_m["target"].mean() * 100,
                "GINI": (2 * auc) - 1,
            })

    for mes in safras_oot:
        df_m = df_oot_scores[df_oot_scores["MONTH"] == mes]
        if df_m["target"].nunique() > 1:
            auc = roc_auc_score(df_m["target"], df_m["score_prob"])
            registros_gini.append({
                "MONTH": int(mes),
                "Partição": "Out-of-Time (OOT)",
                "Volume": len(df_m),
                "Inadimplência (%)": df_m["target"].mean() * 100,
                "GINI": (2 * auc) - 1,
            })

    df_gini_temporal = pd.DataFrame(registros_gini)
    df_gini_temporal.to_csv(metrics_dir / "estabilidade_gini_temporal.csv", index=False)

    media_gini_oot = df_gini_temporal.loc[df_gini_temporal["Partição"] == "Out-of-Time (OOT)", "GINI"].mean()
    if gerar_graficos:
        plot_estabilidade_gini_temporal(
            df_gini=df_gini_temporal[["MONTH", "GINI"]],
            meses_oot=safras_oot,
            media_oot=media_gini_oot,
            filepath=figures_dir / "04_estabilidade_gini_temporal.png",
        )

    # Teste Qui-Quadrado
    df_contingencia = df_gini_temporal.copy()
    df_contingencia["MONTH"] = df_contingencia["MONTH"].astype(str)
    resultados_chi2 = teste_qui_quadrado_temporal(
        taxa_mensal_display=df_contingencia,
        col_taxa="Inadimplência (%)",
        col_volume="Volume",
    )

    # =========================================================================
    # 5.0 TABELA DE DECIS E SEPARAÇÃO K-S
    # =========================================================================
    logger.info("Computando Tabela de Decis e Estatística K-S no OOT...")
    tabela_decis = gerar_tabela_decis_ks(y_true=y_oot, prob=df_oot["score_prob"], n_decis=10)
    ks_max_row = tabela_decis.loc[tabela_decis["ks"].idxmax()]
    decil_ks_max = int(ks_max_row["decil"])
    valor_ks_max = float(ks_max_row["ks"])

    df_oot_sorted = df_oot.sort_values("score_prob", ascending=False).reset_index(drop=True)
    top10_n = int(0.10 * len(df_oot_sorted))
    top10_bads = df_oot_sorted.head(top10_n)["target"].sum()
    total_bads = df_oot_sorted["target"].sum()
    top10_bad_rate = df_oot_sorted.head(top10_n)["target"].mean()
    overall_bad_rate = df_oot_sorted["target"].mean()
    lift_top10 = float(top10_bad_rate / overall_bad_rate)
    pct_capturada_top10 = float((top10_bads / total_bads) * 100)

    # =========================================================================
    # 6.0 SIMULAÇÃO FINANCEIRA DE CONCESSÃO
    # =========================================================================
    logger.info("Simulando resultado financeiro e trade-off de concessão...")
    valores_credito_oot = df_oot["credamount_770A"].values
    df_tradeoff, lucro_realizado = simular_lucro_e_curva_tradeoff(
        y_true=y_oot,
        prob=df_oot["score_prob"],
        valores_credito=valores_credito_oot,
        margem_lucro=margem_lucro,
        threshold_otimizado=threshold,
        filepath=figures_dir / "simulacao_tradeoff_financeiro.png" if gerar_graficos else None,
    )
    df_tradeoff.to_csv(metrics_dir / "simulacao_tradeoff_financeiro.csv", index=False, float_format="%.4f")

    # Lucro por safra
    lucros_por_safra = []
    for mes in sorted(df_oot["MONTH"].unique()):
        mask = df_oot["MONTH"] == mes
        y_m = y_oot[mask].values
        p_m = df_oot.loc[mask, "score_prob"].values
        v_m = df_oot.loc[mask, "credamount_770A"].values
        aprov_m = p_m < threshold
        lucro_m = np.sum(v_m[aprov_m & (y_m == 0)] * margem_lucro) - np.sum(v_m[aprov_m & (y_m == 1)])
        tx_aprov_m = (np.sum(aprov_m) / len(p_m)) * 100
        lucros_por_safra.append({
            "Safra": str(mes),
            "Contratos": int(len(p_m)),
            "Aprovação (%)": float(tx_aprov_m),
            "Lucro ($)": float(lucro_m),
        })
    df_lucro_safra = pd.DataFrame(lucros_por_safra)

    # Matriz de Confusão e Custos
    y_pred_oot = (df_oot["score_prob"] >= threshold).astype(int)
    fn_mask = (y_oot == 1) & (y_pred_oot == 0)
    fp_mask = (y_oot == 0) & (y_pred_oot == 1)
    perda_fn = float(df_oot.loc[fn_mask, "credamount_770A"].sum())
    oportunidade_fp = float(df_oot.loc[fp_mask, "credamount_770A"].sum() * margem_lucro)

    if gerar_graficos:
        plot_matriz_confusao(
            y_true=y_oot,
            y_prob=df_oot["score_prob"],
            threshold=threshold,
            filepath=figures_dir / "matriz_confusao_oot.png",
        )

    # =========================================================================
    # 8.0 EXPORTAÇÃO DE MÉTRICAS CONTÍNUAS (MLOps JSON)
    # =========================================================================
    logger.info("Exportando artifacts/monitoring_metrics_oot.json...")
    monitoring_metrics = {
        "timestamp_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_homologado": "LightGBM_Opt_Lite_Calibrado",
        "total_features": len(features),
        "threshold_congelado": float(threshold),
        "discriminacao_oot": {
            "roc_auc": float(roc_auc_score(y_oot, df_oot["score_prob"])),
            "gini": float(2 * roc_auc_score(y_oot, df_oot["score_prob"]) - 1),
            "pr_auc": float(average_precision_score(y_oot, df_oot["score_prob"])),
            "ks_max": float(valor_ks_max),
            "decil_ks_max": int(decil_ks_max),
            "lift_top10_pct": float(lift_top10),
            "pct_inadimplentes_top10": float(pct_capturada_top10),
        },
        "drift_populacional": {
            "psi_score_geral": float(psi_geral),
            "status_psi_score": "Estável" if psi_geral < 0.10 else ("Alerta" if psi_geral < 0.25 else "Crítico"),
            "ks_distribuicao_scores": float(ks_shift_stat),
            "p_valor_ks_shift": float(p_val_shift),
            "psi_features": {k: float(v) for k, v in psi_por_feature.items()},
        },
        "financeiro_negocio": {
            "lucro_total_realizado": float(lucro_realizado),
            "taxa_aprovacao_pct": float((df_oot["score_prob"] < threshold).mean() * 100),
            "exposicao_fn_dolar": float(perda_fn),
            "custo_oportunidade_fp_dolar": float(oportunidade_fp),
            "lucro_por_safra": df_lucro_safra.to_dict(orient="records"),
        },
    }

    caminho_metricas_oot = artifacts_dir / "monitoring_metrics_oot.json"
    with open(caminho_metricas_oot, "w", encoding="utf-8") as f:
        json.dump(monitoring_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métricas MLOps exportadas com sucesso em: %s", caminho_metricas_oot)

    return monitoring_metrics


def main():
    parser = argparse.ArgumentParser(description="Avaliação de modelos OOT e geração de métricas MLOps.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Diretório contendo os arquivos df_train.parquet, df_test.parquet e df_oot.parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretório onde os modelos e metadados estão salvos.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=BASE_DIR / "reports",
        help="Diretório raiz para salvar relatórios (metrics e figures).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="modelo",
        help="Nome do modelo serializado em artifacts/ (padrão: modelo).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold de decisão (se omitido, lê do modelo_metrics.json).",
    )
    parser.add_argument(
        "--margem-lucro",
        type=float,
        default=0.15,
        help="Margem de ganho líquido sobre crédito adimplente.",
    )
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Desativa a geração de gráficos PNG pesados.",
    )
    args = parser.parse_args()

    executar_avaliacao_completa(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        nome_modelo=args.model_name,
        threshold=args.threshold,
        margem_lucro=args.margem_lucro,
        gerar_graficos=not args.sem_graficos,
    )


if __name__ == "__main__":
    main()

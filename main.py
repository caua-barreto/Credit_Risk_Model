"""
main.py — Orquestrador do Pipeline de Risco de Crédito (Home Credit).

Permite executar o pipeline completo de ponta a ponta a partir de dados brutos
(raw parquet files) ou etapas individuais (features, train, evaluate):

Comandos:
  python main.py --all
  python main.py --features
  python main.py --train
  python main.py --evaluate
  python main.py --predict --input-csv data/sample.csv

Parâmetros Customizáveis:
  --data-dir: Diretório de dados brutos ou processados
  --output-dir: Diretório para salvar bases processadas
  --artifacts-dir: Diretório para modelos e metadados
  --reports-dir: Diretório para relatórios, tabelas e figuras
  --executar-shap: Ativa cálculo dinâmico de poda SHAP
  --sem-graficos: Desativa renderização de figuras PNG pesadas
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Resolução da raiz do projeto
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Orquestrador End-to-End do Pipeline de Risco de Crédito.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Modos de Execução
    group_mode = parser.add_argument_group("Modos de Execução")
    group_mode.add_argument(
        "--all",
        action="store_true",
        help="Executa o pipeline completo: build_features -> train -> evaluate.",
    )
    group_mode.add_argument(
        "--features",
        action="store_true",
        help="Executa o pipeline de feature engineering a partir dos dados brutos.",
    )
    group_mode.add_argument(
        "--train",
        action="store_true",
        help="Executa o treinamento, calibração e exportação do modelo campeão.",
    )
    group_mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Executa a avaliação estatística, temporal (Drift/GINI) e financeira no OOT.",
    )
    group_mode.add_argument(
        "--predict",
        action="store_true",
        help="Executa inferência em lote em um arquivo CSV.",
    )

    # Diretórios e Caminhos
    group_paths = parser.add_argument_group("Caminhos e Diretórios")
    group_paths.add_argument(
        "--raw-train-dir",
        type=Path,
        default=None,
        help="Diretório contendo as tabelas brutas da competição (parquet_files/train).",
    )
    group_paths.add_argument(
        "--processed-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Diretório onde os arquivos processados (df_train, df_test, df_oot) são salvos.",
    )
    group_paths.add_argument(
        "--artifacts-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
        help="Diretório para salvar/carregar os modelos serializados e metadados.",
    )
    group_paths.add_argument(
        "--reports-dir",
        type=Path,
        default=BASE_DIR / "reports",
        help="Diretório para salvar tabelas de métricas e figuras executivas.",
    )
    group_paths.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Arquivo CSV de entrada para predição (obrigatório se --predict for usado).",
    )
    group_paths.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "predictions.csv",
        help="Arquivo CSV de saída das predições.",
    )

    # Opções de Modelagem e Avaliação
    group_options = parser.add_argument_group("Opções de Modelagem e Avaliação")
    group_options.add_argument(
        "--executar-shap",
        action="store_true",
        help="Calcula a importância SHAP dinamicamente para poda de variáveis.",
    )
    group_options.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold financeiro customizado (se omitido, usa o ótimo do modelo_metrics.json).",
    )
    group_options.add_argument(
        "--margem-lucro",
        type=float,
        default=0.15,
        help="Margem de ganho operacional sobre créditos adimplentes (padrão: 0.15).",
    )
    group_options.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Desativa a geração de figuras PNG pesadas durante a avaliação.",
    )

    args = parser.parse_args()

    # Se nenhum modo explícito for passado, exibe o help
    if not (args.all or args.features or args.train or args.evaluate or args.predict):
        parser.print_help()
        sys.exit(0)

    t_inicio_total = time.time()
    logger.info("=" * 80)
    logger.info(" INICIANDO PIPELINE DE RISCO DE CRÉDITO — HOME CREDIT")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Feature Engineering
    # -------------------------------------------------------------------------
    if args.all or args.features:
        logger.info("\n>>> [ETAPA 1/3] FEATURE ENGINEERING & PREPARAÇÃO DE DADOS <<<")
        t0 = time.time()
        from scripts.build_features import executar_pipeline

        executar_pipeline(
            train_dir=args.raw_train_dir,
            output_dir=args.processed_dir,
            caminho_pipeline=args.artifacts_dir / "feature_pipeline.joblib",
            incluir_dpd_momentum=True,
            escalonar=True,
        )
        logger.info("Etapa de Feature Engineering concluída em %.1f segundos.", time.time() - t0)

    # -------------------------------------------------------------------------
    # 2. Treinamento e Calibração
    # -------------------------------------------------------------------------
    if args.all or args.train:
        logger.info("\n>>> [ETAPA 2/3] TREINAMENTO, CALIBRAÇÃO & OTIMIZAÇÃO DE THRESHOLD <<<")
        t0 = time.time()
        from scripts.train import treinar_e_exportar

        treinar_e_exportar(
            data_dir=args.processed_dir,
            artifacts_dir=args.artifacts_dir,
            executar_shap=args.executar_shap,
            top_n_remover=10,
            margem_lucro=args.margem_lucro,
        )
        logger.info("Etapa de Treinamento concluída em %.1f segundos.", time.time() - t0)

    # -------------------------------------------------------------------------
    # 3. Avaliação Estatística, Drift e Financeira (OOT)
    # -------------------------------------------------------------------------
    if args.all or args.evaluate:
        logger.info("\n>>> [ETAPA 3/3] VALIDAÇÃO OOT, MONITORAMENTO DRIFT & RESULTADO FINANCEIRO <<<")
        t0 = time.time()
        from scripts.evaluate import executar_avaliacao_completa

        executar_avaliacao_completa(
            data_dir=args.processed_dir,
            artifacts_dir=args.artifacts_dir,
            reports_dir=args.reports_dir,
            nome_modelo="modelo",
            threshold=args.threshold,
            margem_lucro=args.margem_lucro,
            gerar_graficos=not args.sem_graficos,
        )
        logger.info("Etapa de Avaliação concluída em %.1f segundos.", time.time() - t0)

    # -------------------------------------------------------------------------
    # 4. Inferência em Lote
    # -------------------------------------------------------------------------
    if args.predict:
        if args.input_csv is None:
            logger.error("Erro: Parâmetro --input-csv é obrigatório quando --predict for solicitado.")
            sys.exit(1)

        logger.info("\n>>> INFERÊNCIA EM LOTE (PREDICTIONS) <<<")
        t0 = time.time()
        from scripts.predict import carregar_modelo_e_features, carregar_ou_aplicar_pipeline
        import pandas as pd
        import numpy as np

        modelo, features = carregar_modelo_e_features(args.artifacts_dir, "modelo")
        df_raw = pd.read_csv(args.input_csv)
        case_ids = df_raw["case_id"] if "case_id" in df_raw.columns else pd.Series(range(len(df_raw)), name="case_id")

        df_proc = carregar_ou_aplicar_pipeline(df_raw, args.artifacts_dir / "feature_pipeline.joblib")
        if features:
            for col in features:
                if col not in df_proc.columns:
                    df_proc[col] = 0.0
            X = df_proc[features]
        else:
            X = df_proc

        thresh = args.threshold
        if thresh is None:
            caminho_metrics = args.artifacts_dir / "modelo_metrics.json"
            if caminho_metrics.exists():
                import json
                with open(caminho_metrics) as f:
                    thresh = float(json.load(f).get("threshold_otimo", 0.1189))
            else:
                thresh = 0.1189

        probs = modelo.predict_proba(X)[:, 1]
        decisoes = (probs >= thresh).astype(int)

        df_out = pd.DataFrame({
            "case_id": case_ids,
            "score_inadimplencia": probs,
            "previsao_inadimplente": decisoes,
            "decisao_credito": np.where(decisoes == 1, "Reprovado", "Aprovado"),
        })
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output_csv, index=False)
        logger.info("Predições salvas em: %s (shape: %s) em %.1f segundos.", args.output_csv, df_out.shape, time.time() - t0)

    logger.info("\n" + "=" * 80)
    logger.info(" EXECUÇÃO DO PIPELINE FINALIZADA EM %.1f SEGUNDOS COM SUCESSO!", time.time() - t_inicio_total)
    logger.info("=" * 80)


if __name__ == "__main__":
    run_pipeline()


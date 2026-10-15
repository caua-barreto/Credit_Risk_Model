"""
pipeline.py — Pipeline de feature engineering do projeto Home Credit
(Credit Risk Model Stability).

Reproduz, fora do notebook, o fluxo do ``02_Feature_Engineering.ipynb``:

     1. Carga da train_base + merge Depth 0 (static_0 / static_cb_0)
     2. Agregacoes Depth 1/2 (historico) + merge por case_id
     3. Divisao temporal: OOT (MONTH >= 202007) + split 70/30 estratificado
     4. Shadow features (padrao de nulos -> Flag_Thin_File)
     5. Engenharia de features (racios, intensidade, flags, sazonalidade,
        benchmark por profissao)
     6. One-Hot Encoding de Mes_do_Ano (categorias aprendidas no treino)
     7. Remocao de esparsidade (>70% nulos no treino)
     8. Encoding de categoricas residuais (descarte de ruido / OHE)
     9. Imputacao de zero logico (contagens e flags)
    10. Poda de multicolinearidade (lista validada no notebook)
    11. Escalonamento RobustScaler (fit no treino, NaNs preservados)
    12. Exportacao em Parquet (df_train, df_test, df_oot, master_table)

Principios
----------
* Toda estatistica (lista de colunas esparsas, medias por profissao,
  mediana de DPD, categorias do OHE e parametros do RobustScaler) e
  aprendida APENAS no treino (``fit``) e reaplicada em teste/OOT
  (``transform``), evitando data leakage.
* Todas as transformacoes sao aplicadas aos 3 conjuntos, garantindo que
  treino, teste e OOT terminem com exatamente as mesmas colunas.
* O OHE de ``Mes_do_Ano`` usa as categorias do treino. Fazer
  ``get_dummies(drop_first=True)`` direto no OOT (meses 7-10) descartaria
  julho como se fosse a categoria de referencia (janeiro).

Uso
---
Linha de comando (a partir da raiz do projeto)::

    python scripts/pipeline.py --train-dir /caminho/parquet_files/train
    python scripts/pipeline.py              # baixa os dados via kagglehub

Notebook / outro script::

    import sys; sys.path.append('/content/Projeto_06/scripts')
    from pipeline import executar_pipeline
    df_train, df_test, df_oot, pipe = executar_pipeline(train_dir=train_dir)

    # Reaplicar o pipeline ja treinado em novos dados
    from pipeline import CreditRiskPipeline
    pipe = CreditRiskPipeline.load('artifacts/feature_pipeline.joblib')
    df_novo = pipe.transform(df_novo_master)
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

# Resolução da raiz do projeto para permitir execução como script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    get_table_groups,
    carregar_familia,
    downcast_train_base,
    merge_depth_0,
    descobrir_colunas_depth_1,
    merge_depth_1,
)


# =============================================================================
# Constantes (validadas no notebook 02_Feature_Engineering)
# =============================================================================

TARGET: str = "target"
COLUNAS_ID: List[str] = ["case_id", "date_decision", "MONTH", "WEEK_NUM", TARGET]

OOT_START: int = 202007
TEST_SIZE: float = 0.30
RANDOM_STATE: int = 42
N_FOLDS: int = 5

LIMIAR_ESPARSIDADE: float = 0.70
LIMIAR_CONCENTRACAO_MODA: float = 0.90
MAX_CARDINALIDADE_OHE: int = 20

# Secao 1.0 — Top 50 colunas (estritamente pre-decisao / historicas)
COLUNAS_SELECIONADAS: List[str] = [
    # Scores de bureau
    "score_940", "riskassesment_302T", "riskassesment_940T",
    # Historico passado de atrasos (bureau)
    "maxdpdlast12m_727P", "maxdpdlast24m_143P", "avgdbddpdlast24m_3658932P",
    "maxdbddpdlast1m_3658939P", "pctinstlsallpaidlat10d_839L",
    "pctinstlsallpaidlate6d_3546844L", "numinstlswithdpd5_4187116L",
    "actualdpd_943P", "dpdmax_139P", "dpdmax_757P", "pmts_dpd_1073P",
    # Endividamento pre-existente
    "totaldebt_9A", "currdebt_22A", "totaldebtoverduevalue_178A",
    "totaldebtoverduevalue_718A", "totaloutstanddebtvalue_39A",
    "totaloutstanddebtvalue_668A", "debtoutstand_525A",
    "overdueamountmax_155A", "overdueamountmax_35A", "residualamount_1093A",
    "credamount_770A", "annuity_780A", "amount_1115A", "credlmt_935A",
    "outstandingdebt_522A", "monthlyinstlamount_332A", "pmts_overdue_1140A",
    # Rendimento e perfil socio-economico
    "mainoccupationinc_384A", "incometype_1044T", "empl_employedfrom_271D",
    "empl_employedtotal_800L", "commnoinclast6m_3546845L", "clientscnt_887L",
    # Comportamento de busca e rejeicoes
    "numrejects9m_859L", "days30_165L", "applicationscnt_867L",
    "applicationscnt_629L", "forquarter_462L", "days180_256L",
    # Historico de bons pagamentos
    "numinstunpaidmaxest_4493212L", "numinstpaid_4499208L",
    "posfpd30lastmonth_3976960P", "numactiverelcontr_750L", "pmtnum_8L",
    "tenor_203L", "credacc_transactions_402L",
]

# Secao 3.0 — Agregacoes Depth 1/2
AGG_MAP_DEPTH_1: Dict[str, List[str]] = {
    # Atrasos e comportamento de pagamento passado
    "dpdmax_139P": ["max"],
    "dpdmax_757P": ["max"],
    "pmts_dpd_1073P": ["max", "mean"],
    "pmts_overdue_1140A": ["sum", "max"],
    "numinstunpaidmaxest_4493212L": ["max"],
    "posfpd30lastmonth_3976960P": ["max", "mean"],
    # Endividamento e limites pre-existentes no bureau
    "amount_1115A": ["sum", "mean"],
    "credlmt_935A": ["sum", "max"],
    "outstandingdebt_522A": ["sum"],
    "monthlyinstlamount_332A": ["sum"],
    "numactiverelcontr_750L": ["sum"],
    # Maturidade e volume de contratos anteriores
    "credamount_770A": ["sum", "max"],
    "annuity_780A": ["sum", "mean"],
    "pmtnum_8L": ["sum", "max"],
    "numinstpaid_4499208L": ["sum", "mean"],
    "tenor_203L": ["mean", "max"],
    "credacc_transactions_402L": ["sum", "mean"],
}

# Secao 6.0 — Identificadores anonimizados sem poder discriminativo
COLUNAS_RUIDO: List[str] = ["lastrejectreasonclient_4145040M"]

# Secao 7.2 — Ausencia de registo == zero eventos
COLUNAS_ZERO_LOGICO: List[str] = [
    # Flags e contagens de bureau
    "numrejects9m_859L", "applicationscnt_867L", "applicationscnt_629L",
    "forquarter_462L", "days180_256L", "clientscnt_887L",
    "empl_employedtotal_800L",
    # Contagens de contratos
    "numinstunpaidmaxest_4493212L", "numinstpaid_4499208L",
    "numinstlswithdpd5_4187116L", "numactiverelcontr_750L",
    "pmtnum_8L_max", "pmtnum_8L_sum",
    "credacc_transactions_402L_sum", "credacc_transactions_402L_mean",
    # Flags criados
    "Flag_Has_RiskAssessment_302T", "Flag_Has_RiskAssessment_940T",
    "Flag_Thin_File", "Flag_First_Payment_Default",
    "Flag_Sobre_Endividamento_Aberto", "Flag_Pico_Atraso_Recente",
    "Flag_Pandemia",
]

# Secoes 8.1 / 8.2 — Multicolinearidade (correlacao > 0.80 e VIF > 10).
# Entre cada par, mantida a feature de maior IV.
COLUNAS_MULTICOLINEARIDADE: List[str] = [
    # --- Correlacao entre pares ---
    "posfpd30lastmonth_3976960P",    # substituida por Flag_First_Payment_Default
    "Flag_Has_RiskAssessment_302T",  # representada por Flag_Thin_File
    "Flag_Has_RiskAssessment_940T",  # representada por Flag_Thin_File
    "tenor_203L_max",                # substituida por pmtnum_8L_max (maior IV)
    "numrejects9m_859L",             # substituida por Frequencia_Rejeicoes_Mensal
    "dpdmax_139P_max",               # identica a pmts_dpd_1073P_max
    "totaldebt_9A",                  # identica a currdebt_22A
    "outstandingdebt_522A_sum",      # quase identica a currdebt_22A (menor IV)
    "Termometro_Busca_Bureau",       # substituida por applicationscnt_867L (maior IV)
    "pctinstlsallpaidlat10d_839L",   # pior IV vs pctinstlsallpaidlate6d_3546844L
    "WEEK_NUM",                      # variavel de controlo, nao preditiva
    "maxdpdlast12m_727P",            # pior IV vs maxdpdlast24m_143P
    # --- VIF > 10 (drop efetivamente executado na celula 8.2) ---
    "credlmt_935A_max",
    "pmts_overdue_1140A_sum",        # mantida a 'max' (maior IV)
]


# =============================================================================
# 1. Carga e merges (Depth 0 + Depth 1/2)
# =============================================================================

def baixar_dados() -> Path:
    """
    Baixa a competicao via kagglehub (credenciais lidas do ``.env`` da raiz
    do projeto) e retorna o diretorio ``parquet_files/train``.
    """
    from dotenv import load_dotenv
    from src.pipeline import download_competition

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    base_dir = download_competition()
    return base_dir / "parquet_files" / "train"


def construir_master_table(
    grupos_tabelas: Dict[str, List[Path]],
    colunas_selecionadas: Optional[List[str]] = None,
    agg_map: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Secoes 2.0 e 3.0 do notebook: carrega a train_base, faz o merge das
    tabelas estaticas (Depth 0) e das agregacoes de historico (Depth 1/2).

    Parameters
    ----------
    grupos_tabelas : dict
        Output de ``get_table_groups``.
    colunas_selecionadas : list[str], optional
        Default: ``COLUNAS_SELECIONADAS``.
    agg_map : dict, optional
        Default: ``AGG_MAP_DEPTH_1``.

    Returns
    -------
    pd.DataFrame
        Master table (1 linha por case_id).
    """
    colunas_selecionadas = colunas_selecionadas or COLUNAS_SELECIONADAS
    agg_map = agg_map or AGG_MAP_DEPTH_1

    df_base = carregar_familia(grupos_tabelas, "train_base")
    df_base = downcast_train_base(df_base)

    df_master = merge_depth_0(df_base, grupos_tabelas, colunas_selecionadas)
    del df_base
    gc.collect()
    print(f"\nShape final Depth 0: {df_master.shape}")

    col_to_familia, familias = descobrir_colunas_depth_1(grupos_tabelas, agg_map=agg_map)
    if familias:
        df_master = merge_depth_1(
            df_master, grupos_tabelas, col_to_familia, familias, agg_map=agg_map
        )
    print(f"\nShape final Depth 1/2: {df_master.shape}")
    return df_master


# =============================================================================
# 2. Divisao temporal (OOT + split 70/30 estratificado)
# =============================================================================

def dividir_temporal(
    df: pd.DataFrame,
    oot_start: int = OOT_START,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Secao 4.0 do notebook. Isola os meses >= ``oot_start`` como OOT e
    divide o historico em treino/teste estratificado pelo target.

    Returns
    -------
    tuple
        (df_train, df_test, df_oot)
    """
    df_oot = df[df["MONTH"] >= oot_start].copy()
    df_hist = df[df["MONTH"] < oot_start]

    df_train, df_test = train_test_split(
        df_hist,
        test_size=test_size,
        random_state=random_state,
        stratify=df_hist[TARGET],
    )
    df_train, df_test = df_train.copy(), df_test.copy()

    if verbose:
        def _resumo(nome: str, d: pd.DataFrame) -> None:
            print(
                f"{nome:<5}: {d.shape[0]:>9,} linhas | periodo "
                f"{d['MONTH'].min()} a {d['MONTH'].max()} | "
                f"default rate {d[TARGET].mean() * 100:.2f}%"
            )

        print("\n=== DIVISAO TEMPORAL ===")
        _resumo("Train", df_train)
        _resumo("Test", df_test)
        _resumo("OOT", df_oot)

    return df_train, df_test, df_oot


def criar_stratified_kfold(
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """StratifiedKFold usado na validacao cruzada do 03_Modeling."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# =============================================================================
# 3. Feature engineering linha-a-linha (sem estado, sem risco de leakage)
# =============================================================================

def criar_shadow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Secao 5.0 — Flags binarias a partir da presenca/ausencia dos scores de bureau."""
    if "riskassesment_302T" in df.columns:
        df["Flag_Has_RiskAssessment_302T"] = (~df["riskassesment_302T"].isna()).astype("int8")

    if "riskassesment_940T" in df.columns:
        df["Flag_Has_RiskAssessment_940T"] = (~df["riskassesment_940T"].isna()).astype("int8")

    # Thin File: sem nenhum dos dois scores de bureau
    cols_bureau = [c for c in ["riskassesment_302T", "riskassesment_940T"] if c in df.columns]
    if len(cols_bureau) == 2:
        df["Flag_Thin_File"] = (
            df[cols_bureau[0]].isna() & df[cols_bureau[1]].isna()
        ).astype("int8")
    elif len(cols_bureau) == 1:
        df["Flag_Thin_File"] = df[cols_bureau[0]].isna().astype("int8")

    return df


def criar_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Secao 5.1 — Racios de alavancagem e esforco financeiro (NaN onde divisor <= 0)."""
    # Racio Anuidade vs Credito
    if "annuity_780A" in df.columns and "credamount_770A" in df.columns:
        df["Racio_Anuidade_Credito"] = np.where(
            df["credamount_770A"].fillna(0) > 0,
            df["annuity_780A"] / df["credamount_770A"],
            np.nan,
        ).astype("float32")

    # Taxa de Utilizacao Global: divida atual / limite total
    if "currdebt_22A" in df.columns and "credlmt_935A_sum" in df.columns:
        df["Taxa_Utilizacao_Global"] = np.where(
            df["credlmt_935A_sum"].fillna(0) > 0,
            df["currdebt_22A"] / (df["credlmt_935A_sum"].fillna(0) + 1),
            np.nan,
        ).astype("float32")

    # Severidade da Divida Vencida: divida vencida / divida total
    if "totaldebtoverduevalue_178A" in df.columns and "totaldebt_9A" in df.columns:
        df["Severidade_Divida_Vencida"] = np.where(
            df["totaldebt_9A"].fillna(0) > 0,
            df["totaldebtoverduevalue_178A"] / (df["totaldebt_9A"].fillna(0) + 1),
            np.nan,
        ).astype("float32")

    # Ticket Medio Historico: montante medio por prestacao
    if "monthlyinstlamount_332A_sum" in df.columns and "pmtnum_8L_sum" in df.columns:
        df["Ticket_Medio_Historico"] = np.where(
            df["pmtnum_8L_sum"].fillna(0) > 0,
            df["monthlyinstlamount_332A_sum"].fillna(0) / (df["pmtnum_8L_sum"].fillna(0) + 1),
            np.nan,
        ).astype("float32")

    # Endividamento vs Rendimento (DTI)
    if "currdebt_22A" in df.columns and "mainoccupationinc_384A" in df.columns:
        df["Endividamento_vs_Rendimento"] = np.where(
            df["mainoccupationinc_384A"].fillna(0) > 0,
            df["currdebt_22A"] / (df["mainoccupationinc_384A"].fillna(0) + 1),
            np.nan,
        ).astype("float32")

    return df


def criar_dpd_momentum(df: pd.DataFrame, mediana_dpd: Optional[float] = None) -> pd.DataFrame:
    """
    Secao 5.2 — Dinamica de atrasos (descrita no notebook, mas sem celula de
    codigo; por isso e OPCIONAL no pipeline: ``incluir_dpd_momentum=True``).

    ``mediana_dpd`` deve ser aprendida no treino (ver ``CreditRiskPipeline``).
    """
    if "actualdpd_943P" in df.columns and "maxdpdlast24m_143P" in df.columns:
        df["Aceleracao_Atraso_Ativo"] = (
            df["actualdpd_943P"] - df["maxdpdlast24m_143P"]
        ).astype("float32")

    if "actualdpd_943P" in df.columns and mediana_dpd is not None:
        df["Flag_Pico_Atraso_Recente"] = (
            df["actualdpd_943P"].fillna(0) > mediana_dpd
        ).astype("int8")

    if "numinstlswithdpd5_4187116L" in df.columns and "numinstpaid_4499208L" in df.columns:
        df["Racio_Falhas_Prestracoes"] = (
            df["numinstlswithdpd5_4187116L"] / (df["numinstpaid_4499208L"].fillna(0) + 1)
        ).astype("float32")

    if "maxdpdlast24m_143P" in df.columns and "maxdpdlast12m_727P" in df.columns:
        df["Diferencial_Degradacao_Cliente"] = (
            df["maxdpdlast24m_143P"] - df["maxdpdlast12m_727P"]
        ).astype("float32")

    return df


def criar_intensidade(df: pd.DataFrame) -> pd.DataFrame:
    """Secao 5.3 — Intensidade de busca e sinais anomalos."""
    if "numrejects9m_859L" in df.columns:
        df["Frequencia_Rejeicoes_Mensal"] = (
            df["numrejects9m_859L"].fillna(0) / 9
        ).astype("float32")

    if "applicationscnt_867L" in df.columns and "applicationscnt_629L" in df.columns:
        df["Termometro_Busca_Bureau"] = (
            df["applicationscnt_867L"].fillna(0) / (df["applicationscnt_629L"].fillna(0) + 1)
        ).astype("float32")

    if "mainoccupationinc_384A" in df.columns and "commnoinclast6m_3546845L" in df.columns:
        df["Probabilidade_Rendimento_Instavel"] = (
            (df["mainoccupationinc_384A"].fillna(0) - df["commnoinclast6m_3546845L"].fillna(0)).abs()
            / (df["mainoccupationinc_384A"].fillna(0) + 1)
        ).astype("float32")

    return df


def criar_flags_risco_critico(df: pd.DataFrame) -> pd.DataFrame:
    """Secao 5.4 — Flags binarios de risco critico."""
    col_fpd = "posfpd30lastmonth_3976960P"
    if col_fpd in df.columns:
        df["Flag_First_Payment_Default"] = (df[col_fpd].fillna(0) > 0).astype("int8")

    col_limite = "credlmt_935A_sum"
    if "currdebt_22A" in df.columns and col_limite in df.columns:
        divida = df["currdebt_22A"].fillna(0)
        limite = df[col_limite].fillna(0)
        df["Flag_Sobre_Endividamento_Aberto"] = (
            (limite > 0) & (divida > 0.8 * limite)
        ).astype("int8")

    return df


def criar_features_linha(df: pd.DataFrame) -> pd.DataFrame:
    """Secao 5.5 — Features sazonais / macroeconomicas de linha unica."""
    if "MONTH" in df.columns:
        df["Mes_do_Ano"] = (df["MONTH"] % 100).astype("int16")
        df["Flag_Pandemia"] = (df["MONTH"] >= 202003).astype("int8")

    overdue_cols = [c for c in df.columns if "totaldebtoverduevalue" in c]
    if len(overdue_cols) >= 2:
        df["Volatilidade_Divida_Vencida"] = df[overdue_cols].std(axis=1).astype("float32")

    if (
        "annuity_780A" in df.columns
        and "mainoccupationinc_384A" in df.columns
        and "Flag_Pandemia" in df.columns
    ):
        df["Stress_Nova_Prestacao"] = (
            (df["annuity_780A"].fillna(0) / (df["mainoccupationinc_384A"].fillna(0) + 1))
            * (1 + df["Flag_Pandemia"] * 0.5)
        ).astype("float32")

    return df


def imputar_zero_logico(df: pd.DataFrame, colunas: Optional[List[str]] = None) -> pd.DataFrame:
    """Secao 7.2 — NaN -> 0 onde ausencia de registo significa 'zero eventos'."""
    colunas = COLUNAS_ZERO_LOGICO if colunas is None else colunas
    for col in colunas:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


# =============================================================================
# 4. Diagnostico (Information Value)
# =============================================================================

def calcular_iv_dataset(df: pd.DataFrame, target_col: str = TARGET, bins: int = 10) -> pd.DataFrame:
    """
    Secao 8.0 — Information Value de todas as features, com faixas de Siddiqi.
    Deve ser executado APENAS sobre o treino.
    """
    iv_resultados = []

    df_valid = df.dropna(subset=[target_col])
    total_bons = (df_valid[target_col] == 0).sum()
    total_maus = (df_valid[target_col] == 1).sum()
    eps = 1e-6

    features = [c for c in df_valid.columns if c not in COLUNAS_ID + [target_col]]

    for col in features:
        serie = df_valid[col]

        if pd.api.types.is_numeric_dtype(serie):
            try:
                categorias = pd.qcut(serie, q=bins, duplicates="drop").astype(str)
            except Exception:
                categorias = pd.cut(serie, bins=bins).astype(str)
            categorias = categorias.fillna("Missing")
        else:
            categorias = serie.fillna("Missing").astype(str)

        tabela = df_valid.groupby(categorias)[target_col].agg(total="count", maus="sum").reset_index()
        tabela["bons"] = tabela["total"] - tabela["maus"]
        tabela["dist_bons"] = (tabela["bons"] + eps) / total_bons
        tabela["dist_maus"] = (tabela["maus"] + eps) / total_maus
        tabela["woe"] = np.log(tabela["dist_bons"] / tabela["dist_maus"])
        iv_total = ((tabela["dist_bons"] - tabela["dist_maus"]) * tabela["woe"]).sum()

        if iv_total < 0.02:
            status = "Sem Poder Preditivo"
        elif iv_total < 0.10:
            status = "Poder Fraco"
        elif iv_total < 0.30:
            status = "Poder Medio"
        elif iv_total <= 0.50:
            status = "Poder Forte"
        else:
            status = "Alerta: Suspeito / Leakage"

        iv_resultados.append({"Feature": col, "Information_Value": iv_total, "Diagnostico": status})

    return (
        pd.DataFrame(iv_resultados)
        .sort_values(by="Information_Value", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# 5. Pipeline com estado (fit no treino -> transform em teste/OOT)
# =============================================================================

class CreditRiskPipeline:
    """
    Pipeline de feature engineering com semantica ``fit``/``transform``.

    Recebe DataFrames ja mergeados (output de ``construir_master_table`` e
    ``dividir_temporal``). Tudo o que depende de estatisticas da amostra e
    aprendido em ``fit_transform(df_train)`` e reaplicado em ``transform``.

    Parameters
    ----------
    limiar_esparsidade : float, default 0.70
        Colunas com fracao de nulos (no treino) acima disto sao removidas.
    colunas_zero_logico : list[str], optional
        Default: ``COLUNAS_ZERO_LOGICO``.
    colunas_multicolinearidade : list[str], optional
        Default: ``COLUNAS_MULTICOLINEARIDADE``.
    colunas_ruido : list[str], optional
        Default: ``COLUNAS_RUIDO``.
    incluir_dpd_momentum : bool, default False
        Cria as features da secao 5.2 (nao executadas no notebook).
    escalonar : bool, default True
        Aplica RobustScaler nas variaveis continuas.
    verbose : bool, default True
    """

    def __init__(
        self,
        limiar_esparsidade: float = LIMIAR_ESPARSIDADE,
        colunas_zero_logico: Optional[List[str]] = None,
        colunas_multicolinearidade: Optional[List[str]] = None,
        colunas_ruido: Optional[List[str]] = None,
        incluir_dpd_momentum: bool = False,
        escalonar: bool = True,
        verbose: bool = True,
    ) -> None:
        self.limiar_esparsidade = limiar_esparsidade
        self.colunas_zero_logico = list(colunas_zero_logico or COLUNAS_ZERO_LOGICO)
        self.colunas_multicolinearidade = list(colunas_multicolinearidade or COLUNAS_MULTICOLINEARIDADE)
        self.colunas_ruido = list(colunas_ruido or COLUNAS_RUIDO)
        self.incluir_dpd_momentum = incluir_dpd_momentum
        self.escalonar = escalonar
        self.verbose = verbose

        # Estado aprendido no treino
        self.mediana_dpd_: Optional[float] = None
        self.benchmark_profissao_: Optional[Dict] = None
        self.categorias_ohe_: Dict[str, list] = {}
        self.colunas_esparsas_: List[str] = []
        self.colunas_cat_descartar_: List[str] = []
        self.colunas_cat_ohe_: List[str] = []
        self.colunas_escalonar_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.colunas_finais_: List[str] = []
        self.fitted_: bool = False

    # -------------------------------------------------------------------------
    # API publica
    # -------------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "CreditRiskPipeline":
        self.fit_transform(df_train)
        return self

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Aprende o estado no treino e devolve o treino transformado."""
        df = self._executar(df_train, fit=True)
        self.fitted_ = True
        if self.verbose:
            print(f"\nPipeline ajustado. Shape treino: {df.shape} | "
                  f"{len(self.feature_names_)} features preditoras")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica o estado aprendido no treino (teste, OOT ou novos dados)."""
        if not self.fitted_:
            raise RuntimeError("Pipeline nao ajustado. Execute fit/fit_transform no treino primeiro.")
        return self._executar(df, fit=False)

    @property
    def feature_names_(self) -> List[str]:
        """Colunas preditoras finais (exclui ids, datas e target)."""
        return [c for c in self.colunas_finais_ if c not in COLUNAS_ID]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        if self.verbose:
            print(f"Pipeline salvo em: {path}")
        return path

    @staticmethod
    def load(path: Path) -> "CreditRiskPipeline":
        return joblib.load(Path(path))

    # -------------------------------------------------------------------------
    # Orquestracao
    # -------------------------------------------------------------------------

    def _executar(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        # Secoes 5.0 a 5.5 — features linha-a-linha
        df = criar_shadow_features(df)
        df = criar_ratios(df)
        if self.incluir_dpd_momentum:
            if fit and "actualdpd_943P" in df.columns:
                self.mediana_dpd_ = float(df["actualdpd_943P"].median())
            df = criar_dpd_momentum(df, self.mediana_dpd_)
        df = criar_intensidade(df)
        df = criar_flags_risco_critico(df)
        df = criar_features_linha(df)
        df = self._benchmark_profissao(df, fit)

        # Secao 6.0 — OHE de Mes_do_Ano
        if fit:
            self.categorias_ohe_ = {}
            if "Mes_do_Ano" in df.columns:
                self.categorias_ohe_["Mes_do_Ano"] = sorted(df["Mes_do_Ano"].dropna().unique().tolist())
        df = self._aplicar_ohe(df, ["Mes_do_Ano"])

        # Secao 7.1 — esparsidade
        df = self._remover_esparsas(df, fit)

        # Secao 6.0 — categoricas residuais (apos a esparsidade, para nao
        # codificar colunas object que ja seriam descartadas)
        df = self._encoding_categorico(df, fit)

        # Secao 7.2 — zero logico
        df = imputar_zero_logico(df, self.colunas_zero_logico)

        # Secoes 8.1 / 8.2 — multicolinearidade (lista validada)
        df = self._podar_multicolinearidade(df, fit)

        # Secao 7.4 — RobustScaler (independente por coluna; a ordem em
        # relacao a poda nao altera o resultado)
        if self.escalonar:
            df = self._escalonar(df, fit)

        return self._alinhar_colunas(df, fit)

    # -------------------------------------------------------------------------
    # Etapas com estado
    # -------------------------------------------------------------------------

    def _benchmark_profissao(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Secao 5.5 — Rendimento/Divida vs media da profissao (medias do treino)."""
        cols = ["mainoccupationinc_384A", "incometype_1044T", "currdebt_22A"]

        if fit:
            self.benchmark_profissao_ = None
            if all(c in df.columns for c in cols):
                self.benchmark_profissao_ = {
                    "media_renda": df.groupby("incometype_1044T")["mainoccupationinc_384A"].mean().to_dict(),
                    "media_divida": df.groupby("incometype_1044T")["currdebt_22A"].mean().to_dict(),
                    "mediana_renda": df["mainoccupationinc_384A"].median(),
                    "mediana_divida": df["currdebt_22A"].median(),
                }

        bm = self.benchmark_profissao_
        if bm is None or not all(c in df.columns for c in cols):
            return df

        # Profissao inexistente no treino -> mediana global do treino
        renda_proj = df["incometype_1044T"].map(bm["media_renda"]).astype(float).fillna(bm["mediana_renda"])
        divida_proj = df["incometype_1044T"].map(bm["media_divida"]).astype(float).fillna(bm["mediana_divida"])

        df["Rendimento_vs_Media_Profissao"] = (
            df["mainoccupationinc_384A"].fillna(0) / (renda_proj + 1)
        ).astype("float32")
        df["Divida_vs_Media_Profissao"] = (
            df["currdebt_22A"].fillna(0) / (divida_proj + 1)
        ).astype("float32")
        return df

    def _aplicar_ohe(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """OHE (drop_first) usando SEMPRE as categorias aprendidas no treino."""
        for col in cols:
            if col not in df.columns or col not in self.categorias_ohe_:
                continue
            categorias = self.categorias_ohe_[col]
            valores = df[col]
            if col != "Mes_do_Ano":
                valores = valores.map(lambda v: str(v) if pd.notna(v) else np.nan)
            dummies = pd.get_dummies(
                pd.Categorical(valores, categories=categorias),
                prefix=col, drop_first=True, dtype="int8",
            )
            dummies.index = df.index
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _remover_esparsas(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            pct_nulos = df.isnull().mean()
            self.colunas_esparsas_ = [
                c for c in pct_nulos[pct_nulos > self.limiar_esparsidade].index
                if c not in COLUNAS_ID
            ]
            if self.verbose:
                print(f"\nColunas com >{self.limiar_esparsidade * 100:.0f}% de nulos no treino "
                      f"({len(self.colunas_esparsas_)}):")
                for c in self.colunas_esparsas_:
                    print(f"  {c}: {pct_nulos[c] * 100:.1f}%")
        return df.drop(columns=self.colunas_esparsas_, errors="ignore")

    def _encoding_categorico(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Remove identificadores anonimizados e decide, no treino, entre descartar
        (moda > 90% ou cardinalidade > 20) ou aplicar OHE nas categoricas.
        """
        df = df.drop(columns=[c for c in self.colunas_ruido if c in df.columns])

        if fit:
            cat_cols = [
                c for c in df.select_dtypes(include=["object", "category", "string"]).columns
                if c not in COLUNAS_ID
            ]
            self.colunas_cat_descartar_, self.colunas_cat_ohe_ = [], []
            for col in cat_cols:
                contagens = df[col].value_counts()
                moda_pct = contagens.iloc[0] / len(df) if not contagens.empty else 1.0
                if moda_pct > LIMIAR_CONCENTRACAO_MODA or df[col].nunique() > MAX_CARDINALIDADE_OHE:
                    self.colunas_cat_descartar_.append(col)
                else:
                    self.colunas_cat_ohe_.append(col)
                    self.categorias_ohe_[col] = sorted(df[col].dropna().astype(str).unique().tolist())
            if self.verbose and cat_cols:
                print(f"\nCategoricas descartadas: {self.colunas_cat_descartar_}")
                print(f"Categoricas com OHE: {self.colunas_cat_ohe_}")

        df = df.drop(columns=[c for c in self.colunas_cat_descartar_ if c in df.columns])
        return self._aplicar_ohe(df, self.colunas_cat_ohe_)

    def _podar_multicolinearidade(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cols = [c for c in self.colunas_multicolinearidade if c in df.columns]
        if fit and self.verbose:
            print(f"\nMulticolinearidade: {len(cols)} coluna(s) removida(s): {cols}")
        return df.drop(columns=cols)

    def _escalonar(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """RobustScaler (mediana/IQR) ajustado no treino; NaNs sao ignorados no fit e preservados."""
        if fit:
            binarias = {c for c in df.columns if "Flag" in c or df[c].nunique() == 2}
            self.colunas_escalonar_ = [
                c for c in df.columns
                if c not in COLUNAS_ID
                and c not in binarias
                and pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])
            ]
            self.scaler_ = RobustScaler().fit(df[self.colunas_escalonar_]) if self.colunas_escalonar_ else None
            if self.verbose:
                print(f"\nRobustScaler ajustado no treino em {len(self.colunas_escalonar_)} colunas continuas.")

        if self.scaler_ is None:
            return df

        faltantes = [c for c in self.colunas_escalonar_ if c not in df.columns]
        for c in faltantes:
            df[c] = np.nan

        valores = self.scaler_.transform(df[self.colunas_escalonar_])
        for i, col in enumerate(self.colunas_escalonar_):
            dtype = df[col].dtype if pd.api.types.is_float_dtype(df[col]) else "float32"
            df[col] = valores[:, i].astype(dtype)
        return df

    def _alinhar_colunas(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Garante que teste/OOT/novos dados tenham as mesmas colunas (e ordem) do treino."""
        if fit:
            self.colunas_finais_ = list(df.columns)
            return df

        # target pode nao existir em dados de inferencia
        colunas = [c for c in self.colunas_finais_ if c != TARGET or TARGET in df.columns]
        faltantes = [c for c in colunas if c not in df.columns]
        if faltantes and self.verbose:
            print(f"Aviso: colunas ausentes preenchidas com NaN: {faltantes}")
        return df.reindex(columns=colunas)


# =============================================================================
# 6. Exportacao e execucao ponta-a-ponta
# =============================================================================

def exportar_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_oot: pd.DataFrame,
    output_dir: Path,
    exportar_master_table: bool = True,
) -> None:
    """Secao 9.1 — Salva os conjuntos em Parquet (master_table = treino final)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arquivos = {
        "df_oot.parquet": df_oot,
        "df_train.parquet": df_train,
        "df_test.parquet": df_test,
    }
    if exportar_master_table:
        arquivos["master_table.parquet"] = df_train

    print("\n=== EXPORTACAO ===")
    for nome, df in arquivos.items():
        caminho = output_dir / nome
        df.to_parquet(caminho, index=False, engine="pyarrow")
        print(f"  {nome}: {caminho.stat().st_size / (1024 ** 2):.1f} MB | shape {df.shape}")
    print(f"Exportacao concluida em: {output_dir}")


def executar_pipeline(
    train_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    caminho_pipeline: Optional[Path] = None,
    grupos_tabelas: Optional[Dict[str, List[Path]]] = None,
    df_master: Optional[pd.DataFrame] = None,
    incluir_dpd_momentum: bool = False,
    escalonar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CreditRiskPipeline]:
    """
    Executa o fluxo completo do notebook 02: merges -> split -> FE -> export.

    Parameters
    ----------
    train_dir : Path, optional
        Diretorio ``parquet_files/train``. Se None (e sem ``grupos_tabelas``
        / ``df_master``), baixa os dados via kagglehub.
    output_dir : Path or None
        Destino dos Parquets. ``None`` desativa a exportacao.
    caminho_pipeline : Path or None
        Destino do pipeline ajustado (joblib). ``None`` desativa.
    grupos_tabelas : dict, optional
        Mapeamento de familias ja calculado (pula ``get_table_groups``).
    df_master : pd.DataFrame, optional
        Master table ja mergeada (pula a carga/merges).

    Returns
    -------
    tuple
        (df_train, df_test, df_oot, pipeline_ajustado)
    """
    if df_master is None:
        if grupos_tabelas is None:
            train_dir = Path(train_dir) if train_dir is not None else baixar_dados()
            grupos_tabelas = get_table_groups(train_dir)
        df_master = construir_master_table(grupos_tabelas)

    df_train, df_test, df_oot = dividir_temporal(df_master)
    del df_master
    gc.collect()

    pipe = CreditRiskPipeline(incluir_dpd_momentum=incluir_dpd_momentum, escalonar=escalonar)
    df_train = pipe.fit_transform(df_train)
    df_test = pipe.transform(df_test)
    df_oot = pipe.transform(df_oot)

    assert list(df_train.columns) == list(df_test.columns) == list(df_oot.columns), \
        "Train/Test/OOT com colunas divergentes."

    print("\n=== SHAPES FINAIS ===")
    print(f"  Train: {df_train.shape}")
    print(f"  Test:  {df_test.shape}")
    print(f"  OOT:   {df_oot.shape}")

    out_dir = output_dir if output_dir is not None else (PROJECT_ROOT / 'data' / 'processed')
    pipe_path = caminho_pipeline if caminho_pipeline is not None else (PROJECT_ROOT / 'artifacts' / 'feature_pipeline.joblib')
    exportar_datasets(df_train, df_test, df_oot, out_dir)
    pipe.save(pipe_path)

    return df_train, df_test, df_oot, pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de feature engineering — Home Credit.")
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="Diretorio parquet_files/train. Se omitido, baixa via kagglehub.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--pipeline-path", type=Path,
                        default=PROJECT_ROOT / "artifacts" / "feature_pipeline.joblib")
    parser.add_argument("--dpd-momentum", action="store_true",
                        help="Inclui as features de dinamica de DPD (secao 5.2).")
    parser.add_argument("--sem-escalonamento", action="store_true",
                        help="Nao aplica o RobustScaler.")
    args = parser.parse_args()

    executar_pipeline(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        caminho_pipeline=args.pipeline_path,
        incluir_dpd_momentum=args.dpd_momentum,
        escalonar=not args.sem_escalonamento,
    )


if __name__ == "__main__":
    main()

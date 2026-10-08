"""
pipeline.py — Pipeline de feature engineering para o projeto Home Credit.

Encapsula todo o fluxo validado no notebook 02_Feature_Engineering:
merge depth 0/1, shadow features, engenharia avancada, limpeza de
esparsidade, encoding categorico, imputacao logica, normalizacao
RobustScaler e poda de multicolinearidade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from utils_2 import (
    merge_depth_0,
    descobrir_colunas_depth_1,
    merge_depth_1,
    aplicar_ohe,
    prune_multicollinearity,
)


# =============================================================================
# Constantes
# =============================================================================

LIMIAR_ESPARSIDADE: float = 0.70

COLUNAS_ZERO_FILL: List[str] = [
    # Flags e contagens de bureau
    "numrejects9m_859L",
    "applicationscnt_867L",
    "applicationscnt_629L",
    "forquarter_462L",
    "days180_256L",
    "clientscnt_887L",
    "empl_employedtotal_800L",
    # Contagens de contratos
    "numinstunpaidmaxest_4493212L",
    "numinstpaid_4499208L",
    "numinstlswithdpd5_4187116L",
    "numactiverelcontr_750L",
    "pmtnum_8L_max",
    "pmtnum_8L_sum",
    "credacc_transactions_402L_sum",
    "credacc_transactions_402L_mean",
    # Flags criados
    "Flag_Has_RiskAssessment_302T",
    "Flag_Has_RiskAssessment_940T",
    "Flag_Thin_File",
    "Flag_First_Payment_Default",
    "Flag_Sobre_Endividamento_Aberto",
    "Flag_Pico_Atraso_Recente",
    "Flag_Pandemia",
]

# Colunas removidas por multicolinearidade severa (correlacao > 0.95 + VIF > 10),
# lista validada no notebook 02_Feature_Engineering.
COLUNAS_MULTICOLINEARIDADE: List[str] = [
    # --- Pares com correlacao > 0.95 ---
    "posfpd30lastmonth_3976960P",     # corr 1.000 com Flag_First_Payment_Default
    "Flag_Has_RiskAssessment_302T",   # corr 1.000 com Flag_Thin_File
    "Flag_Has_RiskAssessment_940T",   # corr 1.000 com Flag_Thin_File
    "pmtnum_8L_max",                  # corr 1.000 com tenor_203L_max
    "numrejects9m_859L",              # corr 1.000 com Frequencia_Rejeicoes_Mensal
    "dpdmax_139P_max",                # corr 1.000 com pmts_dpd_1073P_max
    "totaldebt_9A",                   # corr 1.000 com currdebt_22A
    "outstandingdebt_522A_sum",       # corr 0.998 com currdebt_22A
    "applicationscnt_867L",           # corr 0.989 com Termometro_Busca_Bureau
    "pctinstlsallpaidlat10d_839L",    # corr 0.965 com pctinstlsallpaidlate6d
    "WEEK_NUM",                       # variavel de controlo, corr 0.835 com MONTH
    "maxdpdlast24m_143P",             # corr 0.830; resolve VIF infinito do Diferencial
    # --- VIF > 10 ---
    "credlmt_935A_max",               # VIF = 16.31
    "pmts_overdue_1140A_sum",         # VIF = 11.39
]


# =============================================================================
# Pipeline
# =============================================================================


class CreditRiskPipeline:
    """
    Pipeline de feature engineering para o desafio Home Credit.

    Replica fielmente o fluxo validado no notebook 02_Feature_Engineering,
    organizado em 9 etapas sequenciais.

    Parameters
    ----------
    grupos_tabelas : dict
        Mapeamento de familias de tabelas (output de ``get_table_groups``).
    colunas_selecionadas : list[str]
        Colunas finais selecionadas para extracao.
    """

    def __init__(
        self,
        grupos_tabelas: Dict[str, List[str]],
        colunas_selecionadas: List[str],
    ) -> None:
        self.grupos_tabelas = grupos_tabelas
        self.colunas_selecionadas = colunas_selecionadas
        self.fitted: bool = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit_transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Executa todas as 9 etapas do pipeline e retorna a master table final.

        Etapas
        ------
        1. merge_depth_0
        2. merge_depth_1
        3. create_shadow_features
        4. create_engineered_features
        5. clean_sparse_columns
        6. categorical_encoding
        7. impute_nulls
        8. normalize
        9. prune_multicollinearity
        """
        df = self.merge_depth_0(df_base)
        df = self.merge_depth_1(df)
        df = self.create_shadow_features(df)
        df = self.create_engineered_features(df)
        df = self.clean_sparse_columns(df)
        df = self.categorical_encoding(df)
        df = self.impute_nulls(df)
        df = self.normalize(df)
        df = self.prune_multicollinearity(df)
        self.fitted = True
        return df

    def split_temporal(
        self,
        df: pd.DataFrame,
        oot_start: int = 202007,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide o DataFrame em historico e out-of-time com base na coluna MONTH.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com coluna MONTH no formato YYYYMM.
        oot_start : int, default 202007
            Primeiro mes (inclusivo) do conjunto out-of-time.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (df_hist, df_oot)
        """
        mask_oot = df["MONTH"] >= oot_start
        df_hist = df[~mask_oot].copy()
        df_oot = df[mask_oot].copy()
        return df_hist, df_oot

    def export(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Salva o DataFrame em formato Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a ser exportado.
        output_path : Path
            Caminho de destino do arquivo .parquet.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")

    # -------------------------------------------------------------------------
    # Step 1 — Merge Depth 0
    # -------------------------------------------------------------------------

    def merge_depth_0(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Merge das tabelas estaticas (train_static_0 e train_static_cb_0)
        com a train_base via LEFT JOIN por case_id.
        """
        return merge_depth_0(df_base, self.grupos_tabelas, self.colunas_selecionadas)

    # -------------------------------------------------------------------------
    # Step 2 — Merge Depth 1/2
    # -------------------------------------------------------------------------

    def merge_depth_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agregacoes de historico (groupby case_id) e merge ao DataFrame principal.
        Usa o AGG_MAP_DEPTH_1 padrao de utils_2.
        """
        col_to_familia, familias = descobrir_colunas_depth_1(self.grupos_tabelas)
        if familias:
            df = merge_depth_1(
                df, self.grupos_tabelas, col_to_familia, familias
            )
        return df

    # -------------------------------------------------------------------------
    # Step 3 — Shadow Features
    # -------------------------------------------------------------------------

    def create_shadow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flags binarias que capturam o perfil *Thin File* (cliente desconhecido
        pelo bureau) a partir do padrao de nulos dos scores de risco.
        """
        # Flag de presenca de RiskAssessment 302T
        if "riskassesment_302T" in df.columns:
            df["Flag_Has_RiskAssessment_302T"] = (
                ~df["riskassesment_302T"].isna()
            ).astype("int8")

        # Flag de presenca de RiskAssessment 940T
        if "riskassesment_940T" in df.columns:
            df["Flag_Has_RiskAssessment_940T"] = (
                ~df["riskassesment_940T"].isna()
            ).astype("int8")

        # Flag Thin File: sem nenhum dos dois scores de bureau
        cols_bureau = [
            c for c in ["riskassesment_302T", "riskassesment_940T"]
            if c in df.columns
        ]
        if len(cols_bureau) == 2:
            df["Flag_Thin_File"] = (
                df[cols_bureau[0]].isna() & df[cols_bureau[1]].isna()
            ).astype("int8")
        elif len(cols_bureau) == 1:
            df["Flag_Thin_File"] = df[cols_bureau[0]].isna().astype("int8")
        else:
            df["Flag_Thin_File"] = np.int8(1)

        return df

    # -------------------------------------------------------------------------
    # Step 4 — Engineered Features
    # -------------------------------------------------------------------------

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sintese de conhecimento de negocio em variaveis cruzadas:
        racios de esforco, dinamica de DPD, intensidade de busca,
        flags de risco critico e contexto macroeconomico.
        """
        df = self._eng_ratios(df)
        df = self._eng_dpd(df)
        df = self._eng_intensity(df)
        df = self._eng_flags(df)
        df = self._eng_advanced(df)
        return df

    # ---- 4a. Ratios de Alavancagem e Esforco Financeiro ---------------------

    def _eng_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Racios financeiros com np.where para preservar NaN quando divisor = 0."""

        # Racio Anuidade vs Credito
        if "annuity_780A" in df.columns and "credamount_770A" in df.columns:
            df["Racio_Anuidade_Credito"] = np.where(
                df["credamount_770A"].fillna(0) > 0,
                df["annuity_780A"] / df["credamount_770A"],
                np.nan,
            ).astype("float32")

        # Taxa de Utilizacao Global
        if "currdebt_22A" in df.columns and "credlmt_935A_sum" in df.columns:
            df["Taxa_Utilizacao_Global"] = np.where(
                df["credlmt_935A_sum"].fillna(0) > 0,
                df["currdebt_22A"] / (df["credlmt_935A_sum"].fillna(0) + 1),
                np.nan,
            ).astype("float32")

        # Severidade da Divida Vencida
        if "totaldebtoverduevalue_178A" in df.columns and "totaldebt_9A" in df.columns:
            df["Severidade_Divida_Vencida"] = np.where(
                df["totaldebt_9A"].fillna(0) > 0,
                df["totaldebtoverduevalue_178A"] / (df["totaldebt_9A"].fillna(0) + 1),
                np.nan,
            ).astype("float32")

        # Ticket Medio Historico
        if "monthlyinstlamount_332A_sum" in df.columns and "pmtnum_8L_sum" in df.columns:
            df["Ticket_Medio_Historico"] = np.where(
                df["pmtnum_8L_sum"].fillna(0) > 0,
                df["monthlyinstlamount_332A_sum"].fillna(0)
                / (df["pmtnum_8L_sum"].fillna(0) + 1),
                np.nan,
            ).astype("float32")

        # Endividamento vs Rendimento
        if "currdebt_22A" in df.columns and "mainoccupationinc_384A" in df.columns:
            df["Endividamento_vs_Rendimento"] = np.where(
                df["mainoccupationinc_384A"].fillna(0) > 0,
                df["currdebt_22A"] / (df["mainoccupationinc_384A"].fillna(0) + 1),
                np.nan,
            ).astype("float32")

        return df

    # ---- 4b. Dinamica de DPD ------------------------------------------------

    def _eng_dpd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum temporal dos atrasos (DPD)."""

        # Aceleracao de Atraso Ativo
        if "actualdpd_943P" in df.columns and "maxdpdlast24m_143P" in df.columns:
            df["Aceleracao_Atraso_Ativo"] = (
                df["actualdpd_943P"].fillna(0)
                - df["maxdpdlast24m_143P"].fillna(0)
            ).astype("float32")

        # Flag Pico de Atraso Recente
        if "actualdpd_943P" in df.columns:
            mediana_dpd = df["actualdpd_943P"].median()
            df["Flag_Pico_Atraso_Recente"] = (
                df["actualdpd_943P"].fillna(0) > mediana_dpd
            ).astype("int8")

        # Racio de Falhas em Prestacoes
        if "numinstlswithdpd5_4187116L" in df.columns and "numinstpaid_4499208L" in df.columns:
            df["Racio_Falhas_Prestracoes"] = (
                df["numinstlswithdpd5_4187116L"]
                / (df["numinstpaid_4499208L"].fillna(0) + 1)
            ).astype("float32")

        # Diferencial de Degradacao do Cliente
        if "maxdpdlast12m_727P" in df.columns and "maxdpdlast24m_143P" in df.columns:
            df["Diferencial_Degradacao_Cliente"] = (
                df["maxdpdlast12m_727P"].fillna(0)
                - df["maxdpdlast24m_143P"].fillna(0)
            ).astype("float32")

        return df

    # ---- 4c. Intensidade de Busca e Sinais Anomalos -------------------------

    def _eng_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de urgencia financeira e sinais de fraude."""

        # Frequencia de Rejeicoes Mensal
        if "numrejects9m_859L" in df.columns:
            df["Frequencia_Rejeicoes_Mensal"] = (
                df["numrejects9m_859L"].fillna(0) / 9
            ).astype("float32")

        # Termometro de Busca no Bureau
        if "applicationscnt_867L" in df.columns and "applicationscnt_629L" in df.columns:
            df["Termometro_Busca_Bureau"] = (
                df["applicationscnt_867L"].fillna(0)
                / (df["applicationscnt_629L"].fillna(0) + 1)
            ).astype("float32")

        # Probabilidade de Rendimento Instavel
        if "mainoccupationinc_384A" in df.columns and "commnoinclast6m_3546845L" in df.columns:
            df["Probabilidade_Rendimento_Instavel"] = (
                (
                    df["mainoccupationinc_384A"].fillna(0)
                    - df["commnoinclast6m_3546845L"].fillna(0)
                ).abs()
                / (df["mainoccupationinc_384A"].fillna(0) + 1)
            ).astype("float32")

        return df

    # ---- 4d. Flags Binarios de Risco Critico --------------------------------

    def _eng_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags binarios de comportamentos extremos de inadimplencia."""

        # Flag_First_Payment_Default
        col_fpd = "posfpd30lastmonth_3976960P"
        if col_fpd in df.columns:
            df["Flag_First_Payment_Default"] = (
                df[col_fpd].fillna(0) > 0
            ).astype("int8")

        # Flag_Sobre_Endividamento_Aberto
        col_limite = "credlmt_935A_sum"
        if "currdebt_22A" in df.columns and col_limite in df.columns:
            divida = df["currdebt_22A"].fillna(0)
            limite = df[col_limite].fillna(0)
            df["Flag_Sobre_Endividamento_Aberto"] = (
                (limite > 0) & (divida > 0.8 * limite)
            ).astype("int8")

        return df

    # ---- 4e. Features Avancadas, Sazonais e Macroeconomicas -----------------

    def _eng_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sazonalidade, contexto macro e benchmarking por profissao."""

        # Mes do Ano (ciclicidade sazonal) — sera convertido em OHE no step 6
        if "MONTH" in df.columns:
            df["Mes_do_Ano"] = (df["MONTH"] % 100).astype("int16")

        # Flag Pandemia (marco 2020 em diante)
        if "MONTH" in df.columns:
            df["Flag_Pandemia"] = (df["MONTH"] >= 202003).astype("int8")

        # Rendimento vs Media da Profissao
        if "mainoccupationinc_384A" in df.columns and "incometype_1044T" in df.columns:
            media_prof = df.groupby("incometype_1044T")[
                "mainoccupationinc_384A"
            ].transform("mean")
            df["Rendimento_vs_Media_Profissao"] = (
                df["mainoccupationinc_384A"].fillna(0)
                / (media_prof.fillna(0) + 1)
            ).astype("float32")

        # Divida vs Media da Profissao
        if "currdebt_22A" in df.columns and "incometype_1044T" in df.columns:
            media_div_prof = df.groupby("incometype_1044T")[
                "currdebt_22A"
            ].transform("mean")
            df["Divida_vs_Media_Profissao"] = (
                df["currdebt_22A"].fillna(0) / (media_div_prof.fillna(0) + 1)
            ).astype("float32")

        # Volatilidade da Divida Vencida (desvio padrao dos overdue)
        overdue_cols = [c for c in df.columns if "totaldebtoverduevalue" in c]
        if len(overdue_cols) >= 2:
            df["Volatilidade_Divida_Vencida"] = (
                df[overdue_cols].std(axis=1).astype("float32")
            )

        # Stress Nova Prestacao (anuidade relativa ao rendimento, ponderada por pandemia)
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

    # -------------------------------------------------------------------------
    # Step 5 — Clean Sparse Columns (>70% nulos)
    # -------------------------------------------------------------------------

    def clean_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas com mais de 70% de valores nulos."""
        pct_nulos = df.isnull().mean()
        cols_remover = pct_nulos[pct_nulos > LIMIAR_ESPARSIDADE].index.tolist()
        if cols_remover:
            print(f"Colunas com >{LIMIAR_ESPARSIDADE*100:.0f}% de nulos ({len(cols_remover)}):")
            for c in cols_remover:
                print(f"  {c}: {pct_nulos[c]*100:.1f}%")
            df = df.drop(columns=cols_remover)
        return df

    # -------------------------------------------------------------------------
    # Step 6 — Categorical Encoding
    # -------------------------------------------------------------------------

    def categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding de variaveis categoricas:
        1. Remove lastrejectreasonclient (codigo anonimizado de alta cardinalidade).
        2. Remove categoricas com >90% de nulos.
        3. Aplica OHE (drop_first=True) nas candidatas validas.
        """
        # 6.1 Remocao explicita de identificador anonimizado
        if "lastrejectreasonclient_4145040M" in df.columns:
            df = df.drop(columns=["lastrejectreasonclient_4145040M"])

        # 6.2 Identificar colunas object/categoricas
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Descarte de categoricas com extrema esparsidade (>90% nulos)
        cols_esparsas = [c for c in cat_cols if df[c].isna().mean() > 0.90]
        if cols_esparsas:
            df = df.drop(columns=cols_esparsas)
            cat_cols = [c for c in cat_cols if c not in cols_esparsas]

        # 6.3 Diagnostico e criterios
        THRESHOLD_CONCENTRACAO = 90.0
        THRESHOLD_CARDINALIDADE_OHE = 20

        cols_descartar: List[str] = []
        cols_ohe: List[str] = []

        for col in cat_cols:
            nunique = df[col].nunique()
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            mode_count = (df[col] == mode_val).sum() if mode_val is not None else 0
            mode_pct = mode_count / len(df) * 100

            if mode_pct > THRESHOLD_CONCENTRACAO:
                cols_descartar.append(col)
            elif nunique <= THRESHOLD_CARDINALIDADE_OHE:
                cols_ohe.append(col)
            else:
                cols_descartar.append(col)

        # Descartar
        cols_descartar_existentes = [c for c in cols_descartar if c in df.columns]
        if cols_descartar_existentes:
            df = df.drop(columns=cols_descartar_existentes)

        # OHE
        if cols_ohe:
            df = aplicar_ohe(df, cols_ohe, drop_first=True)

        return df

    # -------------------------------------------------------------------------
    # Step 7 — Impute Nulls
    # -------------------------------------------------------------------------

    def impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputacao logica baseada na semantica da variavel:
        - Zero para contagens/flags (ausencia = zero eventos).
        - Mantem NaN para DPD/historico (ausencia = cliente desconhecido).
        """
        cols_zero = [c for c in COLUNAS_ZERO_FILL if c in df.columns]
        for col in cols_zero:
            nulos_antes = df[col].isna().sum()
            if nulos_antes > 0:
                df[col] = df[col].fillna(0)
        return df

    # -------------------------------------------------------------------------
    # Step 8 — Normalize (RobustScaler)
    # -------------------------------------------------------------------------

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Escalonamento robusto (mediana/IQR) nas variaveis continuas.
        Preserva NaNs (nativo para XGBoost/LightGBM/CatBoost).
        Ignora colunas de identificacao, binarias/flags e temporais.
        """
        scaler = RobustScaler()

        colunas_identificacao = ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]
        colunas_binarias = [
            c for c in df.columns
            if "Flag" in c or df[c].nunique() == 2
        ]
        colunas_temporais = ["Mes_do_Ano"]

        excecoes = colunas_identificacao + colunas_binarias + colunas_temporais
        cols_to_scale = [
            c for c in df.columns
            if c not in excecoes and pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in cols_to_scale:
            mask = df[col].notna()
            if mask.sum() > 0:
                valores_transformados = scaler.fit_transform(df.loc[mask, [col]])
                df.loc[mask, col] = valores_transformados.flatten()

        return df

    # -------------------------------------------------------------------------
    # Step 9 — Prune Multicollinearity
    # -------------------------------------------------------------------------

    def prune_multicollinearity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas com multicolinearidade severa:
        - Correlacao > 0.95 (pares validados).
        - VIF > 10 (validado via statsmodels).
        Usa a lista consolidada em ``COLUNAS_MULTICOLINEARIDADE``.
        """
        return prune_multicollinearity(df, COLUNAS_MULTICOLINEARIDADE)

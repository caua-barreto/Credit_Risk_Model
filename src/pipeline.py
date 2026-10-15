"""
pipeline.py — Modulo de ingestao, processamento e engenharia de atributos (Feature Engineering).
"""
from __future__ import annotations

import os
import re
import gc
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Union
import locale
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import kagglehub
from dotenv import load_dotenv
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

BASE_DIR = Path(__file__).resolve().parent.parent


def setup_environment(project_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Configura o diretório de trabalho, instala dependências se necessário
    e carrega credenciais Kaggle a partir do .env.

    Parameters
    ----------
    project_root : Optional[Union[str, Path]], default=None
        Caminho raiz do projeto. Se None, resolve dinamicamente a raiz a partir
        da estrutura do repositório (duas pastas acima deste módulo).

    Returns
    -------
    Path
        Caminho raiz do projeto.
    """
    if project_root is None:
        root = Path(__file__).resolve().parent.parent
    else:
        root = Path(project_root)

    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)

    try:
        import kagglehub
    except ImportError:
        print("Instalando dependencias...")
        os.system(f"pip install -r {root}/requirements.txt -q")
        print("Reiniciando runtime...")
        time.sleep(2)
        os._exit(0)

    env_path = root / '.env'
    if not load_dotenv(env_path, override=True):
        raise FileNotFoundError(
            f"Arquivo .env nao encontrado em {env_path}. "
            "Crie o arquivo com KAGGLE_USERNAME e KAGGLE_KEY."
        )

    api_token = os.environ.get('KAGGLE_API_TOKEN')
    if not api_token or not api_token.startswith('KGAT_'):
        raise ValueError("Token KAGGLE_API_TOKEN nao encontrado ou invalido no .env")

    print("Ambiente configurado. Token Kaggle carregado.")
    return root


def download_competition(competition: str = 'home-credit-credit-risk-model-stability') -> Path:
    """
    Baixa a competição via kagglehub e retorna o caminho base dos dados.

    Returns
    -------
    Path
        Diretório raiz dos dados extraídos.
    """
    locale.setlocale(locale.LC_TIME, 'C')
    path_kagglehub = kagglehub.competition_download(competition)
    base_dir = Path(path_kagglehub)
    print(f"Dados localizados em: {base_dir}")
    return base_dir


def get_table_groups(train_dir: Path) -> Dict[str, List[Path]]:
    """
    Mapeia todos os arquivos .parquet do diretório de treino em famílias,
    usando regex para agrupar partições por nome e depth.

    Parameters
    ----------
    train_dir : Path
        Diretório contendo os arquivos .parquet de treino.

    Returns
    -------
    Dict[str, List[Path]]
        Dicionário {nome_familia: [lista_de_arquivos]}.
    """
    grupos: Dict[str, List[Path]] = defaultdict(list)

    for file_path in train_dir.glob("*.parquet"):
        nome = file_path.stem
        if nome == 'train_base':
            grupo = nome
        elif re.search(r'_\d+_\d+$', nome):
            grupo = re.sub(r'_\d+$', '', nome)
        else:
            grupo = nome
        grupos[grupo].append(file_path)

    print(f"Foram mapeadas {len(grupos)} familias de tabelas.")
    return dict(grupos)


# =============================================================================
# 3. CARREGAMENTO DE DADOS
# =============================================================================


def carregar_familia(
    grupos_tabelas: Dict[str, List[Path]],
    nome_grupo: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Carrega todos os partições de uma família via PyArrow nativo
    e retorna um DataFrame unificado.

    Parameters
    ----------
    grupos_tabelas : Dict[str, List[Path]]
        Mapeamento de famílias (output de get_table_groups).
    nome_grupo : str
        Nome da família a carregar.
    columns : list, optional
        Subconjunto de colunas a ler (projeção de colunas no PyArrow).

    Returns
    -------
    pd.DataFrame
    """
    if nome_grupo not in grupos_tabelas:
        raise ValueError(f"Familia '{nome_grupo}' nao encontrada.")

    arquivos = sorted(grupos_tabelas[nome_grupo])
    print(f"Carregando '{nome_grupo}' ({len(arquivos)} arquivo(s))...")

    tabelas = [pq.read_table(arq, columns=columns) for arq in arquivos]
    return pa.concat_tables(tabelas).to_pandas()


def downcast_train_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica downcast de tipos na train_base para otimizar memória.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame da train_base recém-carregado.

    Returns
    -------
    pd.DataFrame
        DataFrame com tipos otimizados.
    """
    df['case_id'] = df['case_id'].astype('int32')
    df['date_decision'] = pd.to_datetime(df['date_decision'])
    df['MONTH'] = df['MONTH'].astype('int32')
    df['WEEK_NUM'] = df['WEEK_NUM'].astype('int16')
    df['target'] = df['target'].astype('int8')
    return df


# =============================================================================
# 4. ANÁLISE DE VALORES NULOS
# =============================================================================


def merge_depth_0(
    df_base: pd.DataFrame,
    grupos_tabelas: Dict[str, List[Path]],
    colunas_selecionadas: List[str]
) -> pd.DataFrame:
    """
    Carrega train_static_0 e train_static_cb_0, filtra colunas selecionadas
    e faz LEFT JOIN com df_base. Auditoria de granularidade inclusa.

    Parameters
    ----------
    df_base : pd.DataFrame
    grupos_tabelas : Dict[str, List[Path]]
    colunas_selecionadas : List[str]

    Returns
    -------
    pd.DataFrame
        Master table apos merge depth 0.
    """
    print("Carregando tabelas de Depth 0...")
    df_static_0 = carregar_familia(grupos_tabelas, 'train_static_0')
    df_static_cb_0 = carregar_familia(grupos_tabelas, 'train_static_cb_0')

    df_static_0['case_id'] = df_static_0['case_id'].astype('int32')
    df_static_cb_0['case_id'] = df_static_cb_0['case_id'].astype('int32')

    cols_s0 = [c for c in colunas_selecionadas if c in df_static_0.columns]
    cols_scb0 = [c for c in colunas_selecionadas if c in df_static_cb_0.columns]

    print(f"Colunas em static_0: {len(cols_s0)}")
    print(f"Colunas em static_cb_0: {len(cols_scb0)}")

    df_s0_sel = df_static_0[['case_id'] + cols_s0]
    df_scb0_sel = df_static_cb_0[['case_id'] + cols_scb0]

    linhas_originais = df_base.shape[0]
    df_master = df_base.merge(df_s0_sel, on='case_id', how='left')
    df_master = df_master.merge(df_scb0_sel, on='case_id', how='left')

    linhas_finais = df_master.shape[0]
    unicos = df_master['case_id'].nunique()

    print(f"\nAuditoria Depth 0: {linhas_originais:,} -> {linhas_finais:,} linhas")
    assert linhas_originais == linhas_finais, "Merge alterou quantidade de linhas."
    assert linhas_finais == unicos, "Regra de unicidade violada."
    print(f"Granularidade preservada. Shape: {df_master.shape}")

    del df_static_0, df_static_cb_0, df_s0_sel, df_scb0_sel
    gc.collect()

    return df_master


# =============================================================================
# 7. AGREGAÇÕES DEPTH 1/2
# =============================================================================

AGG_MAP_DEPTH_1: Dict[str, List[str]] = {
    'dpdmax_139P': ['max'],
    'dpdmax_757P': ['max'],
    'pmts_dpd_1073P': ['max', 'mean'],
    'amount_1115A': ['sum', 'mean'],
    'credamount_770A': ['sum', 'max'],
    'annuity_780A': ['sum', 'mean'],
    'credlmt_935A': ['max', 'sum'],
    'outstandingdebt_522A': ['sum'],
    'monthlyinstlamount_332A': ['sum'],
    'pmts_overdue_1140A': ['sum', 'max'],
    'pmtnum_8L': ['max', 'sum'],
    'numinstunpaidmaxest_4493212L': ['max'],
    'numinstpaid_4499208L': ['sum', 'mean'],
    'tenor_203L': ['max', 'mean'],
    'credacc_transactions_402L': ['sum', 'mean'],
    'posfpd30lastmonth_3976960P': ['max', 'mean'],
    'numactiverelcontr_750L': ['sum'],
}

SKIP_FAMILIES = {'train_base', 'train_static_0', 'train_static_cb_0'}


def descobrir_colunas_depth_1(
    grupos_tabelas: Dict[str, List[Path]],
    agg_map: Optional[Dict[str, List[str]]] = None
) -> Tuple[Dict[str, str], set]:
    """
    Descobre em qual familia cada coluna depth 1/2 existe,
    lendo apenas o schema do primeiro arquivo de cada familia.

    Returns
    -------
    Tuple[Dict[str, str], set]
        (col_to_familia, familias_depth_1)
    """
    agg_map = agg_map or AGG_MAP_DEPTH_1
    colunas = list(agg_map.keys())
    col_to_familia: Dict[str, str] = {}
    familias: set = set()

    for familia in sorted(grupos_tabelas.keys()):
        if familia in SKIP_FAMILIES:
            continue
        primeiro_arq = sorted(grupos_tabelas[familia])[0]
        schema_cols = pq.read_schema(primeiro_arq).names
        for col in colunas:
            if col in schema_cols and col not in col_to_familia:
                col_to_familia[col] = familia
                familias.add(familia)

    nao_encontradas = [c for c in colunas if c not in col_to_familia]
    print(f"Familias depth 1: {sorted(familias)}")
    print(f"Colunas mapeadas: {len(col_to_familia)} de {len(colunas)}")
    if nao_encontradas:
        print(f"Ja na Depth 0 ou ausentes: {nao_encontradas}")

    return col_to_familia, familias


def merge_depth_1(
    df_master: pd.DataFrame,
    grupos_tabelas: Dict[str, List[Path]],
    col_to_familia: Dict[str, str],
    familias: set,
    agg_map: Optional[Dict[str, List[str]]] = None,
    date_col_map: Optional[Dict[str, str]] = None,
    date_lookup_map: Optional[Dict[str, Dict]] = None,
    strict_temporal: bool = True,
) -> pd.DataFrame:
    """
    Carrega colunas Depth 1/2, FILTRA EVENTOS DO FUTURO comparando a data
    do evento com 'date_decision' do cliente, agrega por case_id e faz LEFT JOIN.
    """
    agg_map = agg_map or AGG_MAP_DEPTH_1
    date_lookup_map = date_lookup_map or {}

    if date_col_map is None:
        date_col_map = {
            'applprev_1':       'creationdate_885D',
            'credit_bureau_a':  'dateofcredstart_181D',
            'credit_bureau_b':  'contractdate_551D',
            'tax_registry':     'recorddate_4527225D',
            'person_1':         'birthdate_509D',
        }

    linhas_originais = df_master.shape[0]
    total_cols = 0

    # Referencia temporal de cada cliente
    df_dates = df_master[['case_id', 'date_decision']].copy()

    # Pre-construcao dos lookups
    lookups_cache: Dict[str, pd.DataFrame] = {}
    for fam_target, cfg in date_lookup_map.items():
        if fam_target not in familias:
            continue
        print(f"\n[LOOKUP] Construindo lookup de datas para '{fam_target}' "
              f"a partir de '{cfg['source_family']}'...")
        lookups_cache[fam_target] = _construir_lookup_data(
            grupos_tabelas,
            cfg['source_family'],
            cfg['join_keys'],
            cfg['date_col_source'],
        )
        print(f"[LOOKUP] '{fam_target}': {len(lookups_cache[fam_target]):,} registros.")

    for familia in sorted(familias):
        cols_familia = [c for c, f in col_to_familia.items() if f == familia]
        arquivos = sorted(grupos_tabelas[familia])

        print(f"\n-> {familia} ({len(cols_familia)} colunas, {len(arquivos)} arquivos)")

        usa_lookup = familia in lookups_cache
        date_col_in_parquet = None
        if not usa_lookup:
            for substring, col in date_col_map.items():
                if substring in familia:
                    date_col_in_parquet = col
                    break

        if usa_lookup:
            cfg = date_lookup_map[familia]
            print(f"   [OK] Filtro Temporal via LOOKUP: "
                  f"'{cfg['source_family']}.{cfg['date_col_source']}' "
                  f"join em {cfg['join_keys']}.")

        elif date_col_in_parquet is None:
            msg = (f"Familia '{familia}' sem mapeamento de data e sem lookup. "
                   f"Filtro temporal NAO sera aplicado.")
            if strict_temporal:
                raise ValueError(msg + " Configure date_col_map ou date_lookup_map.")
            print(f"   [AVISO] {msg}")

        else:
            faltantes = [
                a.name for a in arquivos
                if date_col_in_parquet not in pq.read_schema(a).names
            ]
            if faltantes:
                msg = (f"Coluna '{date_col_in_parquet}' ausente em "
                       f"{len(faltantes)}/{len(arquivos)} particoes de '{familia}': "
                       f"{faltantes[:3]}{'...' if len(faltantes) > 3 else ''}")
                if strict_temporal:
                    raise ValueError(msg + "  -> Corrija o mapeamento.")
                print(f"   [AVISO] {msg} -> filtro sera PULADO (leakage!)")
                date_col_in_parquet = None
            else:
                print(f"   [OK] Filtro Temporal ATIVO: '{date_col_in_parquet}' "
                      f"presente em {len(arquivos)}/{len(arquivos)} particoes.")

        agg_dict = {col: agg_map[col] for col in cols_familia}
        lista_agg = []

        for arq in arquivos:
            cols_leitura = ['case_id'] + cols_familia

            # ---- Leitura ----
            if usa_lookup:
                cfg = date_lookup_map[familia]
                extra_keys = [k for k in cfg['join_keys'] if k != 'case_id']
                df_part = pq.read_table(
                    arq, columns=cols_leitura + extra_keys
                ).to_pandas()
            elif date_col_in_parquet:
                df_part = pq.read_table(
                    arq, columns=cols_leitura + [date_col_in_parquet]
                ).to_pandas()
            else:
                df_part = pq.read_table(arq, columns=cols_leitura).to_pandas()

            # ---- Filtro temporal ----
            if usa_lookup:
                cfg = date_lookup_map[familia]
                df_part = df_part.merge(
                    lookups_cache[familia],
                    on=cfg['join_keys'],
                    how='left',
                )
                df_part = df_part.merge(df_dates, on='case_id', how='left')
                df_part[cfg['date_col_source']] = pd.to_datetime(
                    df_part[cfg['date_col_source']], errors='coerce'
                )
                mask = (
                    (df_part[cfg['date_col_source']] < df_part['date_decision'])
                    | (df_part[cfg['date_col_source']].isna())
                )
                n_antes = len(df_part)
                df_part = df_part[mask].drop(
                    columns=['date_decision', cfg['date_col_source']]
                )
                n_depois = len(df_part)
                if n_antes > 0:
                    pct = (n_antes - n_depois) / n_antes * 100
                    print(f"     {arq.name}: {n_antes:,} -> {n_depois:,} "
                          f"({pct:.2f}% eventos futuros removidos)")

            elif date_col_in_parquet and date_col_in_parquet in df_part.columns:
                df_part = df_part.merge(df_dates, on='case_id', how='left')
                df_part[date_col_in_parquet] = pd.to_datetime(
                    df_part[date_col_in_parquet], errors='coerce'
                )
                mask = (
                    (df_part[date_col_in_parquet] < df_part['date_decision'])
                    | (df_part[date_col_in_parquet].isna())
                )
                n_antes = len(df_part)
                df_part = df_part[mask].drop(
                    columns=['date_decision', date_col_in_parquet]
                )
                n_depois = len(df_part)
                if n_antes > 0:
                    pct = (n_antes - n_depois) / n_antes * 100
                    print(f"     {arq.name}: {n_antes:,} -> {n_depois:,} "
                          f"({pct:.2f}% eventos futuros removidos)")

            # ---- Agregacao ----
            df_part_agg = df_part.groupby('case_id').agg(agg_dict)
            df_part_agg.columns = [
                f"{col}_{func}" for col, func in df_part_agg.columns
            ]
            df_part_agg = df_part_agg.reset_index()
            lista_agg.append(df_part_agg)

            del df_part
            gc.collect()

        df_agg_final = pd.concat(lista_agg, ignore_index=True)
        del lista_agg
        gc.collect()

        print(f"  Shape agregado: {df_agg_final.shape}")
        df_master = df_master.merge(df_agg_final, on='case_id', how='left')
        total_cols += df_agg_final.shape[1] - 1

        del df_agg_final
        gc.collect()

    linhas_finais = df_master.shape[0]
    unicos = df_master['case_id'].nunique()
    print(f"\nAuditoria Depth 1: {linhas_originais:,} -> {linhas_finais:,} linhas")
    print(f"Colunas adicionadas: {total_cols}")
    assert linhas_originais == linhas_finais, "Merge depth 1 alterou linhas."
    assert linhas_finais == unicos, "Unicidade violada no depth 1."
    print(f"Granularidade preservada. Shape: {df_master.shape}")

    return df_master
# =============================================================================
# 8. PREPARAÇÃO E DIVISÃO TEMPORAL
# =============================================================================


def divisao_temporal_oot(df: pd.DataFrame, oot_start: int = 202007) -> tuple:
    """
    Divide a master table em conjuntos de Treino e OOT (Out-of-Time).
    
    Parâmetros:
    ----------
    df : pd.DataFrame
        O DataFrame master table.
    oot_start : int
        O mês de corte no formato YYYYMM. Tudo a partir daqui (inclusive) é OOT.
        
    Retorna:
    -------
    tuple
        (df_train, df_oot)
    """
    df_train = df[df['MONTH'] < oot_start].copy()
    df_oot = df[df['MONTH'] >= oot_start].copy()
    
    print(f"Train: {df_train.shape[0]:,} linhas  |  {df_train.shape[1]} colunas")
    print(f"OOT:   {df_oot.shape[0]:,} linhas  |  {df_oot.shape[1]} colunas")
    
    return df_train, df_oot


def aplicar_ohe(
    df: pd.DataFrame,
    cols: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding às colunas categóricas especificadas,
    substituindo-as pelas dummies no DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as colunas a codificar.
    cols : list[str]
        Colunas categóricas a converter.
    drop_first : bool, default True
        Se True, remove a primeira categoria (evita multicolinearidade).

    Returns
    -------
    pd.DataFrame
        DataFrame com as colunas originais substituídas pelas dummies.
    """
    cols_existentes = [c for c in cols if c in df.columns]
    if not cols_existentes:
        return df

    df_ohe = pd.get_dummies(
        df[cols_existentes], drop_first=drop_first, dtype='int8'
    )
    df = pd.concat([df.drop(columns=cols_existentes), df_ohe], axis=1)
    print(f"OHE aplicado em {len(cols_existentes)} coluna(s): {cols_existentes}")
    return df


def prune_multicollinearity(
    df: pd.DataFrame,
    cols_remover: List[str],
) -> pd.DataFrame:
    """
    Remove colunas com multicolinearidade severa (lista validada).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    cols_remover : list[str]
        Lista de colunas a remover (derivada de análise de correlação > 0.95
        e VIF > 10).

    Returns
    -------
    pd.DataFrame
        DataFrame sem as colunas removidas.
    """
    cols_existentes = [c for c in cols_remover if c in df.columns]
    if cols_existentes:
        df = df.drop(columns=cols_existentes)
        print(f"Multicolinearidade: {len(cols_existentes)} coluna(s) removida(s):")
        for c in cols_existentes:
            print(f"  - {c}")
    else:
        print("Multicolinearidade: nenhuma coluna da lista encontrada para remoção.")
    return df


def analisar_multicolinearidade_pares(df, target_col='target', limite=0.80):
    """
    Identifica pares de features numéricas com correlação acima do limite.
    Calcula dinamicamente o Information Value (IV) em relação ao target e 
    imprime a feature de maior poder preditivo sempre à esquerda.
    """
    
    # ---------------------------------------------------------
    # Função Auxiliar: Cálculo rápido de IV
    # ---------------------------------------------------------
    def calcular_iv_interno(feature):
        # Isola a coluna e o target, removendo nulos apenas para o cálculo estatístico
        temp = df[[feature, target_col]].dropna()
        
        # Variáveis binárias ou com poucos valores únicos (agrupamento direto)
        if temp[feature].nunique() <= 10:
            binned = temp[feature]
        # Variáveis contínuas (quebra em até 10 decis)
        else:
            binned = pd.qcut(temp[feature], q=10, duplicates='drop')
            
        # Calcula eventos (Maus/1) e não-eventos (Bons/0) por faixa
        agrupado = temp.groupby(binned)[target_col].agg(['count', 'sum'])
        agrupado['eventos'] = agrupado['sum']
        agrupado['nao_eventos'] = agrupado['count'] - agrupado['sum']
        
        total_eventos = agrupado['eventos'].sum()
        total_nao_eventos = agrupado['nao_eventos'].sum()
        
        # Proteção matemática caso uma classe não exista
        if total_eventos == 0 or total_nao_eventos == 0:
            return 0.0
            
        # Substitui 0 por 0.0001 para evitar erro matemático de log(0)
        taxa_eventos = (agrupado['eventos'] / total_eventos).replace(0, 0.0001)
        taxa_nao_eventos = (agrupado['nao_eventos'] / total_nao_eventos).replace(0, 0.0001)
        
        woe = np.log(taxa_nao_eventos / taxa_eventos)
        iv = (taxa_nao_eventos - taxa_eventos) * woe
        
        return iv.sum()

    # ---------------------------------------------------------
    # 1. Identificar Correlações
    # ---------------------------------------------------------
    # Remove a coluna target do cálculo de correlação, se estiver no df
    cols_numericas = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number])
    matriz_corr = cols_numericas.corr().abs()
    
    # Pega o triângulo superior para evitar pares espelhados
    triangulo_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    
    # Filtra os pares acima do limite estipulado
    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna]) 
        for coluna in triangulo_sup.columns 
        for linha in triangulo_sup.index 
        if triangulo_sup.loc[linha, coluna] > limite
    ]
    
    # ---------------------------------------------------------
    # 2. Calcular IV apenas para as features colineares (otimização de tempo)
    # ---------------------------------------------------------
    features_colineares = set([p[0] for p in pares_altos] + [p[1] for p in pares_altos])
    dict_iv_dinamico = {f: calcular_iv_interno(f) for f in features_colineares}
    
    # ---------------------------------------------------------
    # 3. Ordenação e Print
    # ---------------------------------------------------------
    print(f"--- Pares com Correlação > {limite} ---")
    
    # Ordena para exibir as maiores correlações (próximas de 1.0) primeiro
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        feat1 = par[0]
        feat2 = par[1]
        corr = par[2]
        
        iv1 = dict_iv_dinamico[feat1]
        iv2 = dict_iv_dinamico[feat2]
        
        # A de maior IV vai para a esquerda (Candidata a MANTER)
        if iv1 >= iv2:
            feat_manter, feat_descartar = feat1, feat2
            iv_manter, iv_descartar = iv1, iv2
        else:
            feat_manter, feat_descartar = feat2, feat1
            iv_manter, iv_descartar = iv2, iv1
            
        print(f"{feat_manter} (IV: {iv_manter:.4f}) & {feat_descartar} (IV: {iv_descartar:.4f}) | Corr: {corr:.3f}")


def analisar_vif(df, limite=10.0, imputar_nulos=True):
    """
    Calcula o Fator de Inflação da Variância (VIF) para diagnosticar 
    multicolinearidade multivariada entre as variáveis numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis preditoras.
    limite : float, default=10.0
        Valor de corte para considerar a multicolinearidade alta. 
        (Normalmente, VIF > 5 ou VIF > 10 indica problemas graves).
    imputar_nulos : bool, default=True
        Se True, imputa NaNs com a mediana antes do cálculo 
        (o statsmodels não suporta NaNs). Se False, descarta as linhas com NaNs.

    Retorna
    -------
    pd.DataFrame
        Tabela ordenada com todas as features e os seus respetivos valores de VIF.
    """
    # 1. Isolar variáveis numéricas
    df_num = df.select_dtypes(include=[np.number]).copy()
    
    # 2. Tratamento estrito de NaNs para o statsmodels
    if imputar_nulos:
        df_num = df_num.fillna(df_num.median())
    else:
        df_num = df_num.dropna()
        if df_num.empty:
            raise ValueError("O DataFrame ficou vazio após remover os NaNs. Use imputar_nulos=True.")
            
    # 3. Adicionar constante (intercepto) essencial para o cálculo correto do VIF
    X = add_constant(df_num)
    
    # 4. Calcular o VIF iterativamente
    vif_resultados = []
    colunas = X.columns
    valores = X.values
    
    for i in range(len(colunas)):
        # Ignorar erro de divisão por zero caso haja features perfeitamente colineares
        try:
            vif_val = variance_inflation_factor(valores, i)
        except Exception:
            vif_val = np.inf
            
        vif_resultados.append({
            'Feature': colunas[i],
            'VIF': vif_val
        })
        
    df_vif = pd.DataFrame(vif_resultados)
    
    # 5. Limpar a constante e ordenar
    df_vif = df_vif[df_vif['Feature'] != 'const'].sort_values(by='VIF', ascending=False).reset_index(drop=True)
    
    # 6. Exibir o diagnóstico
    print(f"--- Variáveis com VIF > {limite} (Alta Multicolinearidade) ---")
    altos = df_vif[df_vif['VIF'] > limite]
    
    if altos.empty:
        print("Nenhuma variável ultrapassou o limite de VIF estipulado.")
    else:
        for _, row in altos.iterrows():
            print(f"{row['Feature']}: {row['VIF']:.2f}")
            
    return df_vif


# =============================================================================
# 12. FUNCOES OBJETIVO PARA OPTUNA (TUNING DE HIPERPARAMETROS)
# =============================================================================


def criar_ratios(df):
    """
    Cria rácios financeiros linha a linha.
    O uso de np.where preserva NaNs onde a divisão não for possível.
    """
    # Rácio Anuidade vs Crédito: anuidade / montante solicitado
    if 'annuity_780A' in df.columns and 'credamount_770A' in df.columns:
        df['Racio_Anuidade_Credito'] = np.where(
            df['credamount_770A'].fillna(0) > 0,
            df['annuity_780A'] / df['credamount_770A'],
            np.nan
        ).astype('float32')

    # Taxa de Utilização Global: dívida atual / limite total
    if 'currdebt_22A' in df.columns and 'credlmt_935A_sum' in df.columns:
        df['Taxa_Utilizacao_Global'] = np.where(
            df['credlmt_935A_sum'].fillna(0) > 0,
            df['currdebt_22A'] / (df['credlmt_935A_sum'].fillna(0) + 1),
            np.nan
        ).astype('float32')

    # Severidade da Dívida Vencida: dívida vencida / dívida total
    if 'totaldebtoverduevalue_178A' in df.columns and 'totaldebt_9A' in df.columns:
        df['Severidade_Divida_Vencida'] = np.where(
            df['totaldebt_9A'].fillna(0) > 0,
            df['totaldebtoverduevalue_178A'] / (df['totaldebt_9A'].fillna(0) + 1),
            np.nan
        ).astype('float32')

    # Ticket Médio Histórico: montante médio por prestação
    if 'monthlyinstlamount_332A_sum' in df.columns and 'pmtnum_8L_sum' in df.columns:
        df['Ticket_Medio_Historico'] = np.where(
            df['pmtnum_8L_sum'].fillna(0) > 0,
            df['monthlyinstlamount_332A_sum'].fillna(0) / (df['pmtnum_8L_sum'].fillna(0) + 1),
            np.nan
        ).astype('float32')

    # Endividamento vs Rendimento: dívida atual / rendimento principal
    if 'currdebt_22A' in df.columns and 'mainoccupationinc_384A' in df.columns:
        df['Endividamento_vs_Rendimento'] = np.where(
            df['mainoccupationinc_384A'].fillna(0) > 0,
            df['currdebt_22A'] / (df['mainoccupationinc_384A'].fillna(0) + 1),
            np.nan
        ).astype('float32')

    return df


def criar_dpd_momentum(df):
    """
    Features que capturam a tendência temporal do comportamento de pagamento.

    NOTA: 'Aceleracao_Atraso_Ativo' e 'Flag_Pico_Atraso_Recente' foram descontinuadas
    porque dependiam de 'actualdpd_943P', variável que não passou no merge Depth 1/2.
    """
    # Diferencial de Degradação: pior atraso do último ano vs. dos últimos 2 anos.
    # Valores positivos => o atraso anual piorou em relação à média de 2 anos.
    if 'maxdpdlast24m_143P' in df.columns and 'maxdpdlast12m_727P' in df.columns:
        df['Diferencial_Degradacao_Cliente'] = (
            df['maxdpdlast12m_727P'].fillna(0) - df['maxdpdlast24m_143P'].fillna(0)
        ).astype('float32')

    # Rácio de Falhas: prestações com DPD > 5 / prestações pagas totais.
    if 'numinstlswithdpd5_4187116L' in df.columns and 'numinstpaid_4499208L' in df.columns:
        df['Racio_Falhas_Prestracoes'] = (
            df['numinstlswithdpd5_4187116L'].fillna(0)
            / (df['numinstpaid_4499208L'].fillna(0) + 1)
        ).astype('float32')

    return df


def criar_intensidade(df):
    """
    Cria features que medem urgência financeira e distorções nos pedidos.
    Cálculo efetuado linha a linha, garantindo total isolamento dos dados.
    """
    # Frequência de Rejeições Mensal: rejeições / meses de janela
    if 'numrejects9m_859L' in df.columns:
        df['Frequencia_Rejeicoes_Mensal'] = (
            df['numrejects9m_859L'].fillna(0) / 9
        ).astype('float32')

    # Termômetro de Busca no Bureau: aplicações recentes normalizadas
    if 'applicationscnt_867L' in df.columns and 'applicationscnt_629L' in df.columns:
        df['Termometro_Busca_Bureau'] = (
            df['applicationscnt_867L'].fillna(0) / (df['applicationscnt_629L'].fillna(0) + 1)
        ).astype('float32')

    # Probabilidade de Rendimento Instável: variação entre fontes de rendimento
    if 'mainoccupationinc_384A' in df.columns and 'commnoinclast6m_3546845L' in df.columns:
        df['Probabilidade_Rendimento_Instavel'] = (
            (df['mainoccupationinc_384A'].fillna(0) - df['commnoinclast6m_3546845L'].fillna(0)).abs()
            / (df['mainoccupationinc_384A'].fillna(0) + 1)
        ).astype('float32')

    return df


def criar_flags_risco_critico(df):
    """
    Isola comportamentos extremos que determinam alta probabilidade de default.
    A operação é feita linha a linha, sem risco de data leakage.
    """
    # 1. Flag First Payment Default (FPD)
    col_fpd = 'posfpd30lastmonth_3976960P'
    if col_fpd in df.columns:
        df['Flag_First_Payment_Default'] = (
            df[col_fpd].fillna(0) > 0
        ).astype('int8')

    # 2. Sobre-endividamento Aberto: dívida atual > 80% do limite total consolidado
    col_limite = 'credlmt_935A_sum'
    if 'currdebt_22A' in df.columns and col_limite in df.columns:
        divida = df['currdebt_22A'].fillna(0)
        limite = df[col_limite].fillna(0)
        
        # Marcamos 1 se a dívida superar 80% do limite (e houver limite positivo)
        df['Flag_Sobre_Endividamento_Aberto'] = (
            (limite > 0) & (divida > 0.8 * limite)
        ).astype('int8')
        
    return df


def criar_features_linha(df):
    """
    Cria as features de linha única (sem vazamento de dados).
    """
    # Mês do Ano (ciclicidade sazonal)
    if 'MONTH' in df.columns:
        df['Mes_do_Ano'] = (df['MONTH'] % 100).astype('int16')

    # Flag Pandemia (março 2020 em diante)
    if 'MONTH' in df.columns:
        df['Flag_Pandemia'] = (df['MONTH'] >= 202003).astype('int8')

    # Volatilidade da Dívida Vencida (desvio padrão do overdue ao longo do histórico)
    overdue_cols = [c for c in df.columns if 'totaldebtoverduevalue' in c]
    if len(overdue_cols) >= 2:
        df['Volatilidade_Divida_Vencida'] = df[overdue_cols].std(axis=1).astype('float32')

    # Stress Nova Prestação: anuidade relativa ao rendimento, ponderada por pandemia
    if 'annuity_780A' in df.columns and 'mainoccupationinc_384A' in df.columns and 'Flag_Pandemia' in df.columns:
        df['Stress_Nova_Prestacao'] = (
            (df['annuity_780A'].fillna(0) / (df['mainoccupationinc_384A'].fillna(0) + 1))
            * (1 + df['Flag_Pandemia'] * 0.5)  # Nota: Corrigido de '-' para '*' para manter a lógica de multiplicador de stress
        ).astype('float32')

    return df


def imputar_zero_logico(df, colunas):
    """
    Substitui NaNs por 0 em colunas onde a ausência de informação 
    significa matematicamente 'zero eventos' (ex: zero rejeições).
    """
    for col in colunas:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def calcular_iv_dataset(df: pd.DataFrame, target_col: str, bins: int = 10) -> pd.DataFrame:
    """
    Calcula o Information Value (IV) para todas as features preditoras do DataFrame,
    classificando a relevância preditiva segundo as faixas de Siddiqi.
    """
    iv_resultados = []
    
    df_valid = df.dropna(subset=[target_col]).copy()
    total_bons = (df_valid[target_col] == 0).sum()
    total_maus = (df_valid[target_col] == 1).sum()
    eps = 1e-6 
    
    colunas_ignorar = ['case_id', 'date_decision', 'MONTH', 'WEEK_NUM', target_col]
    features = [c for c in df_valid.columns if c not in colunas_ignorar]
    
    for col in features:
        serie = df_valid[col]
        
        if pd.api.types.is_numeric_dtype(serie):
            try:
                categorias = pd.qcut(serie, q=bins, duplicates='drop').astype(str)
            except Exception:
                categorias = pd.cut(serie, bins=bins).astype(str)
            categorias = categorias.fillna('Missing')
        else:
            categorias = serie.fillna('Missing').astype(str)
            
        tabela = df_valid.groupby(categorias)[target_col].agg(
            total='count',
            maus='sum'
        ).reset_index()
        tabela['bons'] = tabela['total'] - tabela['maus']
        
        tabela['dist_bons'] = (tabela['bons'] + eps) / total_bons
        tabela['dist_maus'] = (tabela['maus'] + eps) / total_maus
        
        tabela['woe'] = np.log(tabela['dist_bons'] / tabela['dist_maus'])
        iv_total = ((tabela['dist_bons'] - tabela['dist_maus']) * tabela['woe']).sum()
        
        if iv_total < 0.02:
            status = 'Sem Poder Preditivo'
        elif iv_total < 0.10:
            status = 'Poder Fraco'
        elif iv_total < 0.30:
            status = 'Poder Médio'
        elif iv_total <= 0.50:
            status = 'Poder Forte'
        else:
            status = 'Alerta: Suspeito / Leakage'
            
        iv_resultados.append({
            'Feature': col,
            'Information_Value': iv_total,
            'Diagnóstico': status
        })
        
    df_iv = pd.DataFrame(iv_resultados).sort_values(by='Information_Value', ascending=False).reset_index(drop=True)
    return df_iv


def validar_esquema(df_train, df_test, df_oot, target_col='target'):
    """Garante que Test/OOT têm exatamente as mesmas colunas que Train."""
    cols_train = set(df_train.columns)
    cols_test  = set(df_test.columns)
    cols_oot   = set(df_oot.columns)

    faltando_test = cols_train - cols_test
    faltando_oot  = cols_train - cols_oot
    extras_test   = cols_test - cols_train
    extras_oot    = cols_oot - cols_train

    print("=== VALIDAÇÃO DE ESQUEMA ===")
    print(f"Train: {len(cols_train)} colunas")
    print(f"Test : {len(cols_test)} colunas | faltando: {len(faltando_test)} | extras: {len(extras_test)}")
    print(f"OOT  : {len(cols_oot)} colunas | faltando: {len(faltando_oot)} | extras: {len(extras_oot)}")

    if faltando_test:
        print(f"   Faltando em Test: {sorted(faltando_test)}")
    if faltando_oot:
        print(f"   Faltando em OOT: {sorted(faltando_oot)}")
    if extras_test:
        print(f"   Extras em Test: {sorted(extras_test)}")
    if extras_oot:
        print(f"   Extras em OOT: {sorted(extras_oot)}")

    assert not faltando_test, "Test tem colunas em falta vs. Train!"
    assert not faltando_oot,  "OOT tem colunas em falta vs. Train!"
    assert not extras_test,   "Test tem colunas extras vs. Train!"
    assert not extras_oot,    "OOT tem colunas extras vs. Train!"

    print("Esquemas alinhados. Pronto para exportar.")


def criar_shadow_features(df):
    """
    Cria flags binárias (1/0) a partir da presença ou ausência de dados vitais.
    A operação é feita linha a linha, sem risco de data leakage.
    """
    # Flag de presença de RiskAssessment 302T
    if 'riskassesment_302T' in df.columns:
        df['Flag_Has_RiskAssessment_302T'] = (~df['riskassesment_302T'].isna()).astype('int8')

    # Flag de presença de RiskAssessment 940T
    if 'riskassesment_940T' in df.columns:
        df['Flag_Has_RiskAssessment_940T'] = (~df['riskassesment_940T'].isna()).astype('int8')

    # Flag Thin File: sem nenhum dos dois scores de bureau
    cols_bureau = [c for c in ['riskassesment_302T', 'riskassesment_940T'] if c in df.columns]
    if len(cols_bureau) == 2:
        df['Flag_Thin_File'] = (
            df[cols_bureau[0]].isna() & df[cols_bureau[1]].isna()
        ).astype('int8')
    elif len(cols_bureau) == 1:
        df['Flag_Thin_File'] = df[cols_bureau[0]].isna().astype('int8')
        
    return df

# =============================================================================
# VALIDACAO PRE-MERGE: INTEGRIDADE DAS COLUNAS DE DATA
# =============================================================================


def validar_colunas_data(
    grupos_tabelas: Dict[str, List[Path]],
    date_col_map: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Verifica, em todas as particoes de cada familia Depth 1,
    se a coluna de data necessaria para o filtro anti-leakage existe.

    Deve ser rodada ANTES de merge_depth_1. Sem esta validacao,
    particoes sem a coluna de data passam pelo filtro em silencio,
    gerando data leakage invisivel.

    Parameters
    ----------
    grupos_tabelas : Dict[str, List[Path]]
        Mapeamento de familias (output de get_table_groups).
    date_col_map : Dict[str, str]
        Mapeamento {substring_familia: nome_coluna_data}.

    Returns
    -------
    Dict[str, Dict]
        Auditoria por familia, com contagem de particoes OK/faltantes
        e exemplos de arquivos problematicos.
    """
    auditoria: Dict[str, Dict] = {}
    problemas_criticos: List[str] = []

    for familia, arquivos in grupos_tabelas.items():
        if familia in SKIP_FAMILIES:
            continue

        # Descobre se esta familia tem mapeamento de data
        date_col = None
        for substring, col in date_col_map.items():
            if substring in familia:
                date_col = col
                break

        if date_col is None:
            auditoria[familia] = {
                'date_col': None,
                'total_particoes': len(arquivos),
                'particoes_com_data': 0,
                'particoes_sem_data': 0,
                'arquivos_faltantes': [],
                'status': 'SEM MAPEAMENTO - filtro temporal NAO sera aplicado',
            }
            continue

        arquivos_ok = 0
        arquivos_faltantes = []

        for arq in arquivos:
            schema_cols = pq.read_schema(arq).names
            if date_col in schema_cols:
                arquivos_ok += 1
            else:
                arquivos_faltantes.append(arq.name)

        status = 'OK'
        if arquivos_faltantes:
            status = 'CRITICO - particoes sem coluna de data'
            problemas_criticos.append(
                f"  [X] {familia} | coluna '{date_col}' ausente em "
                f"{len(arquivos_faltantes)}/{len(arquivos)} particoes: "
                f"{arquivos_faltantes[:3]}{'...' if len(arquivos_faltantes) > 3 else ''}"
            )

        auditoria[familia] = {
            'date_col': date_col,
            'total_particoes': len(arquivos),
            'particoes_com_data': arquivos_ok,
            'particoes_sem_data': len(arquivos_faltantes),
            'arquivos_faltantes': arquivos_faltantes,
            'status': status,
        }

    # -------- Relatorio --------
    print("=" * 78)
    print(" VALIDACAO DE COLUNAS DE DATA (PRE-MERGE DEPTH 1)")
    print("=" * 78)
    for familia, info in auditoria.items():
        if info['status'] == 'OK':
            icone = "[OK]"
        elif 'SEM MAPEAMENTO' in info['status']:
            icone = "[--]"
        else:
            icone = "[X] "
        col = info['date_col'] or "-"
        print(f"{icone} {familia:<32} | data: {col:<28} | "
              f"OK: {info['particoes_com_data']}/{info['total_particoes']} | "
              f"{info['status']}")

    if problemas_criticos:
        print("\n" + "!" * 78)
        print(" PROBLEMAS CRITICOS - RESOLVA ANTES DE SEGUIR")
        print("!" * 78)
        for p in problemas_criticos:
            print(p)
        print(
            "\nAcao: descubra a coluna de data correta para as familias afetadas,\n"
            "ou remova essas familias de 'familias' no merge_depth_1.\n"
            "Caso contrario, eventos do FUTURO entrarao nas agregacoes."
        )
    else:
        print("\n[OK] Todas as particoes possuem a coluna de data. Filtro garantido.")

    print("=" * 78)
    return auditoria

# =============================================================================
# LOOKUP DE DATAS PARA FAMILIAS SEM COLUNA DE DATA PROPRIA
# =============================================================================


def _construir_lookup_data(
    grupos_tabelas: Dict[str, List[Path]],
    familia_fonte: str,
    join_keys: List[str],
    date_col: str,
) -> pd.DataFrame:
    """
    Constroi tabela de lookup (join_keys + date_col) a partir de uma familia-fonte.
    Usada para filtrar temporalmente familias que compartilham a chave composta
    mas nao possuem coluna de data propria (ex: train_credit_bureau_a_2).
    """
    arquivos = sorted(grupos_tabelas[familia_fonte])
    dfs = []
    for arq in arquivos:
        schema_cols = pq.read_schema(arq).names
        for k in join_keys + [date_col]:
            if k not in schema_cols:
                raise ValueError(
                    f"Coluna '{k}' ausente em '{arq.name}' "
                    f"(familia fonte '{familia_fonte}')."
                )
        df = pq.read_table(arq, columns=join_keys + [date_col]).to_pandas()
        dfs.append(df)
    df_lookup = pd.concat(dfs, ignore_index=True)
    df_lookup[date_col] = pd.to_datetime(df_lookup[date_col], errors='coerce')
    return df_lookup


def auditar_anti_leakage(
    grupos_tabelas: Dict[str, List[Path]],
    date_col_map: Dict[str, str],
    date_lookup_map: Optional[Dict[str, Dict]] = None,
    col_to_familia: Optional[Dict[str, str]] = None,
    familias_alvo: Optional[set] = None,
    policy: str = 'strict',
) -> pd.DataFrame:
    """
    Auditoria de risco de leakage por familia Depth 1/2.

    Classifica cada familia em 3 categorias:
      - 'evento'   : filtro linha-a-linha por data propria       -> zero leakage
      - 'contrato' : data herdada via lookup do contrato pai     -> leakage residual
      - 'sem_filtro': nenhuma coluna de data mapeada             -> leakage total

    Com base na 'policy', emite decisao MANTER/REMOVER:
      - 'strict'  : apenas filtro event-level e aceito. 'contrato' = REMOVER.
      - 'relaxed' : filtro 'contrato' tambem e aceito.

    Parameters
    ----------
    grupos_tabelas : Dict[str, List[Path]]
    date_col_map : Dict[str, str]
        Familias COM data propria.
    date_lookup_map : Dict[str, Dict], optional
        Familias SEM data propria que herdam via lookup.
    col_to_familia : Dict[str, str], optional
        Mapa coluna->familia para listar features perdidas.
    familias_alvo : set, optional
        Se fornecido, audita apenas essas familias.
    policy : str, default 'strict'
        'strict' ou 'relaxed'.

    Returns
    -------
    pd.DataFrame
        Uma linha por familia, com decisao e justificativa.
    """
    date_lookup_map = date_lookup_map or {}
    col_to_familia = col_to_familia or {}
    rows = []

    for familia in sorted(grupos_tabelas.keys()):
        if familia in SKIP_FAMILIES:
            continue
        if familias_alvo is not None and familia not in familias_alvo:
            continue

        arquivos = sorted(grupos_tabelas[familia])
        n_part = len(arquivos)

        # Features que seriam perdidas se a familia sair do merge
        features = sorted([c for c, f in col_to_familia.items() if f == familia])
        features_str = ", ".join(features[:4]) + ("..." if len(features) > 4 else "")

        # ---------- Tipo 1: data propria (event-level) ----------
        date_col = None
        for sub, col in date_col_map.items():
            if sub in familia:
                date_col = col
                break

        if date_col is not None:
            faltantes = [
                a.name for a in arquivos
                if date_col not in pq.read_schema(a).names
            ]
            if not faltantes:
                rows.append({
                    'familia': familia,
                    'tipo_filtro': 'evento',
                    'coluna_data': date_col,
                    'particoes_ok': f"{n_part}/{n_part}",
                    'status': 'OK',
                    'decisao': 'MANTER',
                    'justificativa': (
                        f"Filtro linha-a-linha por '{date_col}'. Zero leakage residual."
                    ),
                    'features_em_risco': '',
                })
            else:
                rows.append({
                    'familia': familia,
                    'tipo_filtro': 'evento',
                    'coluna_data': date_col,
                    'particoes_ok': f"{n_part - len(faltantes)}/{n_part}",
                    'status': 'CRITICO',
                    'decisao': 'REMOVER',
                    'justificativa': (
                        f"Coluna '{date_col}' ausente em "
                        f"{len(faltantes)}/{n_part} particoes."
                    ),
                    'features_em_risco': features_str,
                })
            continue

        # ---------- Tipo 2: lookup (contract-level) ----------
        if familia in date_lookup_map:
            cfg = date_lookup_map[familia]
            # valida integridade da familia-fonte
            arqs_fonte = sorted(grupos_tabelas.get(cfg['source_family'], []))
            chaves_faltando = []
            for arq in arqs_fonte:
                cols = pq.read_schema(arq).names
                for k in cfg['join_keys'] + [cfg['date_col_source']]:
                    if k not in cols:
                        chaves_faltando.append(f"{arq.name}:{k}")
            integridade = "OK" if not chaves_faltando else "FALHA"

            if policy == 'strict':
                decisao, status = 'REMOVER', 'LEAKAGE_RESIDUAL'
                justif = (
                    f"Data herdada de {cfg['source_family']}.{cfg['date_col_source']} "
                    f"por {cfg['join_keys']}. Filtro a nivel de CONTRATO: "
                    f"parcelas individuais pos-decisao dentro de contratos iniciados "
                    f"antes NAO sao removidas. Policy 'strict' exige zero leakage."
                )
            else:
                decisao, status = 'MANTER', 'OK_LEAKAGE_ACEITO'
                justif = (
                    f"Data herdada de {cfg['source_family']}.{cfg['date_col_source']}. "
                    f"Filtro a nivel de contrato. Leakage residual aceito pela policy."
                )
            if integridade == 'FALHA':
                decisao, status = 'REMOVER', 'CRITICO'
                justif = f"Familia-fonte incompleta: {chaves_faltando[:3]}"

            rows.append({
                'familia': familia,
                'tipo_filtro': 'contrato',
                'coluna_data': cfg['date_col_source'],
                'particoes_ok': f"{n_part}/{n_part}",
                'status': status,
                'decisao': decisao,
                'justificativa': justif,
                'features_em_risco': features_str,
            })
            continue

        # ---------- Tipo 3: sem filtro ----------
        rows.append({
            'familia': familia,
            'tipo_filtro': 'sem_filtro',
            'coluna_data': '-',
            'particoes_ok': f"0/{n_part}",
            'status': 'LEAKAGE_TOTAL',
            'decisao': 'REMOVER',
            'justificativa': (
                "Nenhuma coluna de data mapeada. Eventos do futuro "
                "entrariam nas agregacoes."
            ),
            'features_em_risco': features_str,
        })

    df = pd.DataFrame(rows)

    # ---------- Relatorio ----------
    print("=" * 100)
    print(f" AUDITORIA DE ANTI-LEAKAGE (policy='{policy}')")
    print("=" * 100)
    for _, r in df.iterrows():
        icone = {'MANTER': '[OK]', 'REMOVER': '[X] '}[r['decisao']]
        print(f"{icone} {r['familia']:<32} | {r['tipo_filtro']:<10} | "
              f"{r['status']:<22} | {r['decisao']}")

    manter = df[df['decisao'] == 'MANTER']['familia'].tolist()
    remover = df[df['decisao'] == 'REMOVER']['familia'].tolist()

    print("-" * 100)
    print(f" Familias a MANTER : {manter}")
    print(f" Familias a REMOVER: {remover}")
    if remover:
        feats_perdidas = df[df['decisao'] == 'REMOVER']['features_em_risco']
        feats_perdidas = [f for f in feats_perdidas if f]
        if feats_perdidas:
            print(f" Features em risco: {feats_perdidas}")
    print("=" * 100)

    return df

def imprimir_mapa_familias(
    grupos_tabelas: Dict[str, List[Path]],
    detalhar_arquivos: bool = True,
    max_arquivos_por_familia: int = 6,
) -> None:
    """
    Imprime, em formato texto legivel, o mapa de familias de tabelas
    produzido por `get_table_groups`.

    Agrupa visualmente por depth (0, 1, 2), marca familias que serao
    puladas no merge (`SKIP_FAMILIES`) e lista as particoes de cada
    familia (com compressao caso haja muitas).

    Parameters
    ----------
    grupos_tabelas : Dict[str, List[Path]]
        Mapeamento {nome_familia: [lista_de_paths]} retornado por
        `get_table_groups`.
    detalhar_arquivos : bool, default True
        Se True, lista os nomes das particoes de cada familia.
        Se False, mostra apenas o resumo em uma linha por familia.
    max_arquivos_por_familia : int, default 6
        Limite de particoes listadas por familia antes de comprimir
        a saida com "... (+N arquivos)".

    Returns
    -------
    None
        A saida e feita via print (side effect).
    """

    # ---------- Helpers internos ----------
    def _depth(familia: str) -> int:
        if familia in ('train_base', 'train_static_0', 'train_static_cb_0'):
            return 0
        sufixo = familia.rsplit('_', 1)[-1]
        return int(sufixo) if sufixo.isdigit() else 99

    # ---------- Ordenacao ----------
    familias_ordenadas = sorted(
        grupos_tabelas.items(),
        key=lambda kv: (_depth(kv[0]), kv[0]),
    )

    if not familias_ordenadas:
        print("Nenhuma familia encontrada em 'grupos_tabelas'.")
        return

    largura_nome = max(len(f) for f, _ in familias_ordenadas)
    linha = "=" * 78

    # ---------- Cabecalho ----------
    print(linha)
    print(f" MAPA DE FAMILIAS DE TABELAS  ({len(grupos_tabelas)} familias)")
    print(linha)

    # ---------- Corpo ----------
    depth_atual = None
    for familia, arquivos in familias_ordenadas:
        d = _depth(familia)

        # Cabecalho de bloco por depth
        if d != depth_atual:
            depth_atual = d
            rotulo = {
                0: "DEPTH 0 (base + estaticos)",
                1: "DEPTH 1",
                2: "DEPTH 2",
            }.get(d, "OUTRAS")
            print(f"\n--- {rotulo} ---")

        tag = " [SKIP]" if familia in SKIP_FAMILIES else ""
        n = len(arquivos)

        print(f"  {familia:<{largura_nome}}  | {n:>2} particao(oes){tag}")

        if not detalhar_arquivos:
            continue

        nomes = sorted(a.name for a in arquivos)
        if n <= max_arquivos_por_familia:
            for nome in nomes:
                print(f"      - {nome}")
        else:
            metade = max_arquivos_por_familia // 2
            for nome in nomes[:metade]:
                print(f"      - {nome}")
            print(f"      ... (+{n - max_arquivos_por_familia + 1} arquivos)")
            for nome in nomes[-(metade - 1):]:
                print(f"      - {nome}")

    # ---------- Resumo ----------
    total_part = sum(len(v) for v in grupos_tabelas.values())
    n_depth_0  = sum(1 for f in grupos_tabelas if _depth(f) == 0)
    n_depth_1  = sum(1 for f in grupos_tabelas if _depth(f) == 1)
    n_depth_2  = sum(1 for f in grupos_tabelas if _depth(f) == 2)

    print("\n" + linha)
    print(" RESUMO")
    print(linha)
    print(f"  Total de familias .......... {len(grupos_tabelas)}")
    print(f"    - Depth 0 ................ {n_depth_0}")
    print(f"    - Depth 1 ................ {n_depth_1}")
    print(f"    - Depth 2 ................ {n_depth_2}")
    print(f"  Total de particoes (arquivos) {total_part}")
    print(linha)
"""
utils_2.py — Funções modulares para o projeto Home Credit Credit Risk Model Stability.

Contém: configuração de ambiente, download Kaggle, mapeamento de famílias parquet,
carregamento otimizado via PyArrow, análise de nulos, visualizações corporativas
e testes estatísticos de drift temporal.
"""
import os
import re
import gc
import locale
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import shap
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import kagglehub
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from dotenv import load_dotenv
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.stats import ks_2samp

# =============================================================================
# 1. CONFIGURAÇÃO DE AMBIENTE
# =============================================================================

def setup_environment(project_root: str = '/content/Projeto_06') -> Path:



    """
    Configura o diretório de trabalho, instala dependências se necessário
    e carrega credenciais Kaggle a partir do .env.

    Returns
    -------
    Path
        Caminho raiz do projeto.
    """
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


# =============================================================================
# 2. DOWNLOAD E MAPEAMENTO DE FAMÍLIAS
# =============================================================================

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

def analisar_valores_nulos(
    df: pd.DataFrame,
    plotar: bool = True,
    figsize: tuple = (10, 6)
) -> pd.DataFrame:
    """
    Analisa valores nulos do DataFrame, retornando tabela com quantidade
    e percentual por coluna. Opcionalmente plota gráfico de barras.

    Parameters
    ----------
    df : pd.DataFrame
    plotar : bool
    figsize : tuple

    Returns
    -------
    pd.DataFrame
        Colunas com nulos, ordenadas da maior para menor quantidade.
    """
    total_nulos = df.isnull().sum()
    percentual_nulos = (total_nulos / len(df)) * 100

    df_nulos = pd.DataFrame({
        'Total Nulos': total_nulos,
        'Percentual (%)': percentual_nulos,
        'Tipo de Dado': df.dtypes
    })
    df_nulos = df_nulos[df_nulos['Total Nulos'] > 0].sort_values(
        by='Total Nulos', ascending=False
    )

    if df_nulos.empty:
        print("Nenhum valor nulo encontrado no DataFrame.")
        return df_nulos

    if plotar:
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            x='Percentual (%)', y=df_nulos.index, data=df_nulos,
            color='#E66C37', edgecolor='black', alpha=0.8
        )
        for p in ax.patches:
            width = p.get_width()
            plt.text(
                width + 0.5, p.get_y() + p.get_height() / 2.0 + 0.1,
                f'{width:.1f}%', ha="left", va="center", fontsize=10, color='black'
            )
        plt.title('Percentual de Valores Nulos por Variavel', fontsize=14, fontweight='bold')
        plt.xlabel('Percentual (%)', fontsize=12)
        plt.ylabel('Variaveis', fontsize=12)
        plt.xlim(0, max(df_nulos['Percentual (%)']) + 10)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    return df_nulos


# =============================================================================
# 5. VISUALIZAÇÕES CORPORATIVAS
# =============================================================================

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.Series, pd.Series, float]:
    """
    Plota a distribuição da variável alvo com estilo corporativo.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    Tuple[pd.Series, pd.Series, float]
        (contagens, percentuais, ratio adimplente:default)
    """
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    ratio = target_counts[0] / target_counts[1]

    print(f"Distribuicao do target:\n")
    print(f"  target=0 (adimplente): {target_counts[0]:>10,}  ({target_pct[0]:.2f}%)")
    print(f"  target=1 (default):    {target_counts[1]:>10,}  ({target_pct[1]:.2f}%)")
    print(f"\n  Ratio (0:1):           {ratio:.1f}:1")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    cor_adimplente = "#AFAFAF"
    cor_default = '#e60000'

    bars = ax.bar(
        [0, 1], target_counts.values,
        color=[cor_adimplente, cor_default],
        edgecolor=['#cccccc', '#cc0000'],
        linewidth=1.5, width=0.55
    )

    max_y = target_counts.max()
    for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
        texto = f'{count:,.0f}\n({pct:.2f}%)'.replace(',', '.')
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_y * 0.03),
            texto, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333'
        )

    ax.set_title('Distribuicao de Clientes: Bom Pagador vs Devedor',
                 fontsize=16, fontweight='bold', color='#333333', pad=25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Bom Pagador\n(0)', 'Devedor\n(1)'],
                       fontsize=13, fontweight='bold', color='#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.get_yaxis().set_visible(False)
    ax.set_ylim(0, max_y * 1.15)
    plt.tight_layout()
    plt.show()

    return target_counts, target_pct, ratio


def plot_sazonalidade(
    df: pd.DataFrame,
    target_col: str = 'target',
    month_col: str = 'MONTH'
) -> pd.DataFrame:
    """
    Plota volume mensal (barras) vs taxa de default (linha) com eixo duplo.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    month_col : str

    Returns
    -------
    pd.DataFrame
        Tabela com Default Rate (%) e Volume por mês.
    """
    periodo_min = df['date_decision'].min()
    periodo_max = df['date_decision'].max()
    n_meses = df[month_col].nunique()

    print(f"Periodo: {periodo_min.strftime('%Y-%m-%d')} -> {periodo_max.strftime('%Y-%m-%d')}")
    print(f"Meses unicos: {n_meses}")
    print(f"Volume medio/mes: {len(df) / n_meses:,.0f}")

    taxa_mensal = df.groupby(month_col)[target_col].agg(
        Volume='count', Default_Rate='mean'
    ).copy()
    taxa_mensal['Default_Rate'] *= 100
    meses_str = taxa_mensal.index.astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')

    cor_volume = '#f0f0f0'
    cor_borda = '#cccccc'
    ax1.bar(meses_str, taxa_mensal['Volume'], color=cor_volume,
            edgecolor=cor_borda, width=0.6, label='Volume de Pedidos')
    ax1.set_ylabel('Volume de Pedidos', fontsize=12, fontweight='bold', color='#666666')
    ax1.tick_params(axis='y', colors='#666666')
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.'))
    )

    ax2 = ax1.twinx()
    cor_taxa = '#e60000'
    ax2.plot(meses_str, taxa_mensal['Default_Rate'], color=cor_taxa,
             marker='o', linewidth=3.5, markersize=8, label='Taxa de Default (%)')
    ax2.set_ylabel('Taxa de Default (%)', fontsize=12, fontweight='bold', color=cor_taxa)
    ax2.tick_params(axis='y', colors=cor_taxa)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
    ax2.set_ylim(0, taxa_mensal['Default_Rate'].max() * 1.3)

    for i, txt in enumerate(taxa_mensal['Default_Rate']):
        ax2.annotate(f'{txt:.2f}%', (meses_str[i], txt),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=10, fontweight='bold', color=cor_taxa)

    plt.title('Sazonalidade: Taxa de Default vs Volume de Pedidos',
              fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax1.set_xlabel('Mes de Decisao (YYYYMM)', fontsize=12, fontweight='bold', color='#555555')
    ax1.set_xticks(range(len(meses_str)))
    ax1.set_xticklabels(meses_str, rotation=45, ha='right', fontsize=11)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left',
               frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=11)

    plt.tight_layout()
    plt.show()

    taxa_display = taxa_mensal.copy()
    taxa_display.index.name = 'Mes'
    taxa_display = taxa_display.rename(columns={'Default_Rate': 'Default Rate (%)'})
    taxa_display = taxa_display[['Default Rate (%)', 'Volume']]

    return taxa_display


def teste_qui_quadrado_temporal(taxa_mensal_display: pd.DataFrame) -> Dict[str, float]:
    """
    Aplica teste qui-quadrado de independencia entre mes e target,
    e calcula V de Cramer como tamanho do efeito.

    Parameters
    ----------
    taxa_mensal_display : pd.DataFrame
        Tabela com colunas 'Default Rate (%)' e 'Volume' (index = mes).

    Returns
    -------
    Dict[str, float]
        Dicionario com chi2, p_valor, gl, freq_esperada_min, cramers_v.
    """
    contingencia = taxa_mensal_display.copy()
    contingencia['taxa_decimal'] = contingencia['Default Rate (%)'] / 100
    contingencia['eventos'] = (contingencia['taxa_decimal'] * contingencia['Volume']).round().astype(int)
    contingencia['nao_eventos'] = contingencia['Volume'] - contingencia['eventos']
    tabela = contingencia[['eventos', 'nao_eventos']]

    chi2, p, gl, expected = stats.chi2_contingency(tabela)

    n = tabela.values.sum()
    k = min(tabela.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))

    resultados = {
        'chi2': chi2,
        'p_valor': p,
        'gl': gl,
        'freq_esperada_min': expected.min(),
        'cramers_v': cramers_v
    }

    print(f"Qui-Quadrado:            {chi2:.2f}")
    print(f"Graus de Liberdade (gl): {gl}")
    print(f"P-valor:                 {p:.4e}")
    print(f"Frequencia esperada min: {expected.min():.2f}")
    print(f"V de Cramer:             {cramers_v:.4f}")

    return resultados


# =============================================================================
# 6. MERGE DEPTH 0
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

def calcular_psi(esperado, atual, bins=10):
    """Calcula o PSI para uma variável numérica contínua para deteção de drift."""
    esperado = esperado.dropna()
    atual = atual.dropna()
    
    limites = np.unique(np.percentile(esperado, np.linspace(0, 100, bins + 1)))
    if len(limites) < 2: return 0.0
    
    limites[0], limites[-1] = -np.inf, np.inf
    
    contagem_esperado, _ = np.histogram(esperado, bins=limites)
    contagem_atual, _ = np.histogram(atual, bins=limites)
    
    contagem_esperado = np.where(contagem_esperado == 0, 0.001, contagem_esperado)
    contagem_atual = np.where(contagem_atual == 0, 0.001, contagem_atual)
    
    perc_esperado = contagem_esperado / np.sum(contagem_esperado)
    perc_atual = contagem_atual / np.sum(contagem_atual)
    
    return np.sum((perc_atual - perc_esperado) * np.log(perc_atual / perc_esperado))

def relatorio_feature_drift_psi(X_hist, X_oot, bins=10):
    """Gera relatório de PSI (Population Stability Index) de todas as features numéricas."""
    from scipy.stats import ks_2samp
    cols_num = X_hist.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']).columns
    resultados = []

    for col in cols_num:
        psi_val = calcular_psi(X_hist[col], X_oot[col], bins=bins)
        ks_val, _ = ks_2samp(X_hist[col].dropna(), X_oot[col].dropna())

        if psi_val < 0.10: status = " Estável"
        elif psi_val < 0.25: status = " Alerta (Drift Moderado)"
        else: status = " Crítico (Drift Severo)"

        resultados.append({'Variável': col, 'PSI': psi_val, 'KS': ks_val, 'Status': status})

    df_psi = pd.DataFrame(resultados).sort_values(by='PSI', ascending=False).reset_index(drop=True)
    df_psi['PSI'] = df_psi['PSI'].apply(lambda x: f"{x:.4f}")
    df_psi['KS'] = df_psi['KS'].apply(lambda x: f"{x:.4f}")

    return df_psi


# =============================================================================

def plot_matriz_correlacao(df, titulo='Matriz de Correlação', triangulo=True,
                           cmap='coolwarm', figsize=(18, 14), annot_size=9):
    """
    Plota heatmap da matriz de correlação de variáveis numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas numéricas.
    titulo : str, default='Matriz de Correlação'
        Título do gráfico.
    triangulo : bool or str, default=True
        - True ou 'inferior': exibe apenas o triângulo inferior.
        - 'superior': exibe apenas o triângulo superior.
        - False ou 'completa': exibe a matriz completa.
    cmap : str, default='coolwarm'
        Paleta de cores do heatmap. Ex: 'RdBu_r', 'viridis', 'magma'.
    figsize : tuple, default=(18, 14)
        Dimensões da figura.
    annot_size : int, default=9
        Tamanho da fonte das anotações.

    Retorna
    -------
    pd.DataFrame
        Matriz de correlação calculada.
    """
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    mascara = None
    if triangulo in (True, 'inferior'):
        mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif triangulo == 'superior':
        mascara = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=.5, cbar_kws={'shrink': .75},
        mask=mascara, annot_kws={'size': annot_size}
    )
    plt.title(titulo, fontsize=16, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return corr

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

def plot_feature_importance(
    modelo,
    feature_names: list,
    top_n: int = 20,
    titulo: str = None,
    figsize: tuple = (14, 10),
    cor: str = 'steelblue',
    xlabel: str = None,
    annot: bool = True,
    fmt: str = '.3f',
    alpha: float = 0.85,
):
    """
    Plota a importância das features de um modelo treinado (coeficientes
    para modelos lineares ou feature_importances_ para árvores).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn já treinado.
    feature_names : list
        Lista com os nomes das features.
    top_n : int, default=20
        Número de features mais importantes a exibir.
    titulo : str, optional
        Título do gráfico.
    figsize : tuple, default=(14, 10)
        Dimensões da figura.
    cor : str, default='steelblue'
        Cor das barras.
    xlabel : str, optional
        Rótulo do eixo X. Se None, usa o tipo detectado.
    annot : bool, default=True
        Se True, exibe o valor numérico ao lado de cada barra.
    fmt : str, default='.3f'
        Formatação dos valores anotados.
    alpha : float, default=0.85
        Transparência das barras.
    """
    clf = modelo
    if hasattr(modelo, 'named_steps') and 'clf' in modelo.named_steps:
        clf = modelo.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        importancias = clf.feature_importances_
        tipo = 'Feature Importance'
    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importancias = np.abs(coefs)
        tipo = '|Coeficientes|'
    else:
        raise ValueError("O modelo não possui 'feature_importances_' nem 'coef_'.")

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importância': importancias
    }).sort_values('Importância', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        df_imp['Feature'], df_imp['Importância'],
        color=cor, edgecolor='black', alpha=alpha,
    )

    if annot:
        for bar, val in zip(bars, df_imp['Importância']):
            ax.text(
                bar.get_width() + df_imp['Importância'].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:{fmt}}',
                va='center', fontsize=9, fontweight='bold',
            )

    ax.set_xlabel(xlabel or tipo, fontsize=12)
    ax.set_title(
        titulo or f'Top {top_n} {tipo}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Organiza do maior para o menor e remove o índice desordenado
    df_retorno = df_imp.sort_values('Importância', ascending=False).reset_index(drop=True)
    
    # Exibe a tabela interativa/formatada no notebook
    display(df_retorno)

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
        Se True, renderiza a tabela diretamente no notebook com display().

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
        display(df_bottom)

    return df_bottom


# =============================================================================
# 13. MODELAGEM — THRESHOLD FINANCEIRO, PODA SHAP E ESTABILIDADE TEMPORAL
#     (funções migradas do notebook 03_Modeling.ipynb — lógica inalterada)
# =============================================================================

def otimizar_thresholds_por_lucro(dicionario_modelos, X_test, y_true, valores_credito, margem_lucro=0.15):
    """
    Encontra, para cada modelo, o threshold que maximiza o lucro esperado
    segundo a matriz de custos:

        Aprovado Bom (pred=0, real=0) -> + valor_credito * margem_lucro
        Aprovado Mau (pred=0, real=1) -> - valor_credito (perda integral)
        Rejeitados                    ->   0

    Parâmetros:
    ----------
    dicionario_modelos : dict[str, estimator]
        Modelos já treinados (precisam expor predict_proba).
    X_test : pd.DataFrame
        Base de avaliação.
    y_true : array-like
        Target real da base de avaliação.
    valores_credito : array-like
        Exposição de cada contrato (ex.: credamount_770A).
    margem_lucro : float, default=0.15
        Margem obtida sobre o valor do crédito para clientes bons.

    Retorno:
    -------
    dict[str, float]
        Threshold ótimo por modelo (grid np.linspace(0.01, 0.99, 100)).
    """
    thresholds_otimizados = {}
    thresholds_teste = np.linspace(0.01, 0.99, 100)
    y_true_arr = np.array(y_true)
    valores_arr = np.array(valores_credito)

    print('=' * 65)
    print(' OTIMIZACAO DE THRESHOLD POR MATRIZ DE CUSTOS (LUCRO MAXIMO) ')
    print('=' * 65)

    for nome, modelo in dicionario_modelos.items():
        y_probs = modelo.predict_proba(X_test)[:, 1]
        lucros = []
        for t in thresholds_teste:
            y_pred = (y_probs >= t).astype(int)
            aprovados_bons = (y_pred == 0) & (y_true_arr == 0)
            aprovados_maus = (y_pred == 0) & (y_true_arr == 1)
            lucro_bons = np.sum(valores_arr[aprovados_bons] * margem_lucro)
            prejuizo_maus = np.sum(valores_arr[aprovados_maus])
            lucros.append(lucro_bons - prejuizo_maus)

        best_idx = np.argmax(lucros)
        best_thresh = thresholds_teste[best_idx]
        lucro_maximo = lucros[best_idx]

        print(f'\nModelo: {nome}')
        print(f'  Threshold Otimo : {best_thresh:.4f}')
        print(f'  Lucro Esperado  : R$ {lucro_maximo:,.2f}')

        thresholds_otimizados[nome] = best_thresh

    return thresholds_otimizados


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


def analisar_estabilidade_gini(X_val, y_val, meses, dicionario_modelos):
    """
    GINI mês a mês (safra) do ensemble por MÉDIA SIMPLES dos 3 Boostings
    ('LightGBM_Opt', 'XGBoost_Opt', 'CatBoost_Opt') e plota a série temporal.

    Retorno:
    -------
    (pd.DataFrame, np.ndarray)
        DataFrame ['MONTH', 'GINI'] e o vetor de probabilidades do ensemble.
    """
    probs_ensemble = np.zeros(len(X_val))
    boostings = ['LightGBM_Opt', 'XGBoost_Opt', 'CatBoost_Opt']
    for nome in boostings:
        probs_ensemble += dicionario_modelos[nome].predict_proba(X_val)[:, 1]
    probs_ensemble /= len(boostings)

    df_eval = pd.DataFrame({'MONTH': meses.values, 'target': y_val.values, 'prob': probs_ensemble})
    resultados_mes = []

    for mes in sorted(df_eval['MONTH'].unique()):
        df_mes = df_eval[df_eval['MONTH'] == mes]
        if len(df_mes['target'].unique()) > 1:
            auc = roc_auc_score(df_mes['target'], df_mes['prob'])
            gini = (2 * auc) - 1
            resultados_mes.append({'MONTH': str(mes), 'GINI': gini})

    df_gini = pd.DataFrame(resultados_mes)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_gini, x='MONTH', y='GINI', marker='o', linewidth=2.5, color='#2980b9')
    media_gini = df_gini['GINI'].mean()
    plt.axhline(media_gini, color='#e74c3c', linestyle='--', label=f'Media OOT: {media_gini:.4f}')
    plt.title('Estabilidade Temporal do GINI (Out-of-Time)', fontsize=14, pad=15)
    plt.xlabel('Mes da Decisao (Safra)', fontsize=12)
    plt.ylabel('GINI Score', fontsize=12)
    plt.ylim(0.40, 0.70)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_gini, probs_ensemble


def calcular_psi_scores(probs_treino, probs_oot, bins=10):
    """
    PSI (Population Stability Index) entre a distribuição de SCORES do treino
    e do OOT, com faixas definidas pelos decis do treino (limites em [0, 1]).

    Nota: renomeada de `calcular_psi` (versão do notebook) para não sobrescrever
    a função `calcular_psi(esperado, atual, bins)` já existente neste módulo,
    usada por `relatorio_feature_drift_psi`. Lógica inalterada.

    Retorno:
    -------
    (pd.DataFrame, float)
        Tabela de PSI por faixa e o PSI total.
    """
    _, limites = pd.qcut(probs_treino, q=bins, retbins=True, duplicates='drop')
    limites[0] = 0.0
    limites[-1] = 1.0

    esperado = pd.cut(probs_treino, bins=limites).value_counts() / len(probs_treino)
    real = pd.cut(probs_oot, bins=limites).value_counts() / len(probs_oot)

    df_psi = pd.DataFrame({'Faixa': esperado.index, 'Treino_%': esperado.values, 'OOT_%': real.values})
    df_psi['Treino_%'] = np.where(df_psi['Treino_%'] == 0, 0.0001, df_psi['Treino_%'])
    df_psi['OOT_%'] = np.where(df_psi['OOT_%'] == 0, 0.0001, df_psi['OOT_%'])
    df_psi['PSI'] = (df_psi['OOT_%'] - df_psi['Treino_%']) * np.log(df_psi['OOT_%'] / df_psi['Treino_%'])

    psi_total = df_psi['PSI'].sum()

    print(f'PSI Total: {psi_total:.4f}')
    if psi_total < 0.10:
        print('[STATUS] EXCELENTE (nenhuma mudanca populacional).')
    elif psi_total < 0.25:
        print('[STATUS] ATENCAO (leve degradacao).')
    else:
        print('[STATUS] ALERTA (concept drift provavel).')

    return df_psi, psi_total


# =============================================================================
# 14. VALIDAÇÃO OOT E ANÁLISE DE NEGÓCIO (04_Model_Avaliation.ipynb)
# =============================================================================

def plot_curvas_roc_pr_sobrepostas(y_test, prob_test, y_oot, prob_oot, figsize=(14, 6)):
    """
    Plota as curvas ROC e Precision-Recall sobrepondo as bases de Teste (In-Time) e OOT.
    Apresenta paleta corporativa (Navy Blue e Crimson Red).
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    fpr_test, tpr_test, _ = roc_curve(y_test, prob_test)
    auc_test = roc_auc_score(y_test, prob_test)
    fpr_oot, tpr_oot, _ = roc_curve(y_oot, prob_oot)
    auc_oot = roc_auc_score(y_oot, prob_oot)

    prec_test, rec_test, _ = precision_recall_curve(y_test, prob_test)
    pr_auc_test = average_precision_score(y_test, prob_test)
    prec_oot, rec_oot, _ = precision_recall_curve(y_oot, prob_oot)
    pr_auc_oot = average_precision_score(y_oot, prob_oot)

    cor_teste = '#1B365D'  # Navy Blue
    cor_oot = '#8B0000'    # Deep Crimson / Vinho

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Curva ROC
    ax1.plot(fpr_test, tpr_test, color=cor_teste, linewidth=2.2,
             label=f'Teste (In-Time) | ROC-AUC = {auc_test:.4f}')
    ax1.plot(fpr_oot, tpr_oot, color=cor_oot, linewidth=2.2, linestyle='--',
             label=f'OOT (Out-of-Time) | ROC-AUC = {auc_oot:.4f}')
    ax1.plot([0, 1], [0, 1], color='#7f8c8d', linestyle=':', linewidth=1.2, label='Aleatório (Baseline = 0.50)')
    ax1.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Curva ROC — Teste vs. OOT', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.grid(True, linestyle='--', alpha=0.35)

    # Curva Precision-Recall
    ax2.plot(rec_test, prec_test, color=cor_teste, linewidth=2.2,
             label=f'Teste (In-Time) | PR-AUC = {pr_auc_test:.4f}')
    ax2.plot(rec_oot, prec_oot, color=cor_oot, linewidth=2.2, linestyle='--',
             label=f'OOT (Out-of-Time) | PR-AUC = {pr_auc_oot:.4f}')
    base_test = y_test.mean()
    base_oot = y_oot.mean()
    ax2.axhline(base_test, color=cor_teste, linestyle=':', alpha=0.6, label=f'Prevalência Teste ({base_test:.2%})')
    ax2.axhline(base_oot, color=cor_oot, linestyle=':', alpha=0.6, label=f'Prevalência OOT ({base_oot:.2%})')
    ax2.set_xlabel('Recall (Cobertura de Inadimplentes)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precisão (Taxa de Acerto no Risco)', fontsize=11, fontweight='bold')
    ax2.set_title('Curva Precision-Recall — Teste vs. OOT', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.grid(True, linestyle='--', alpha=0.35)

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def plot_distribuicao_kde_scores(y_true, prob, nome_base='OOT', figsize=(10, 5)):
    """
    Plota a distribuição de densidade (KDE) dos scores preditos separando
    a classe Adimplente (Bons) da Inadimplente (Maus).
    """
    df_plot = pd.DataFrame({'score': prob, 'target': y_true.values if hasattr(y_true, 'values') else y_true})
    
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df_plot[df_plot['target'] == 0], x='score', color='#1B365D',
                fill=True, alpha=0.3, linewidth=2.0, label='Bons Pagadores (Target = 0)')
    sns.kdeplot(data=df_plot[df_plot['target'] == 1], x='score', color='#8B0000',
                fill=True, alpha=0.35, linewidth=2.0, label='Inadimplentes (Target = 1)')

    plt.title(f'Distribuição de Densidade dos Scores (KDE) — Base {nome_base}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Probabilidade Prevista de Inadimplência (Score)', fontsize=11, fontweight='bold')
    plt.ylabel('Densidade', fontsize=11, fontweight='bold')
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def plot_target_drift_mensal(df_train, df_test, df_oot, col_mes='MONTH', col_target='target', figsize=(12, 5)):
    """
    Gera gráfico temporal corporativo da taxa de default safra a safra (Target Drift)
    cobrindo Treino, Teste e OOT.
    """
    dados = []
    for nome_base, base in [('Treino', df_train), ('Teste', df_test), ('OOT', df_oot)]:
        agrupado = base.groupby(col_mes)[col_target].agg(['count', 'mean']).reset_index()
        agrupado['Base'] = nome_base
        dados.append(agrupado)

    df_consolidado = pd.concat(dados, ignore_index=True)
    df_consolidado['Taxa_Default_Pct'] = df_consolidado['mean'] * 100
    df_consolidado = df_consolidado.sort_values(col_mes)

    plt.figure(figsize=figsize)
    paleta = {'Treino': '#1B365D', 'Teste': '#4682B4', 'OOT': '#8B0000'}

    for base, cor in paleta.items():
        subset = df_consolidado[df_consolidado['Base'] == base]
        if not subset.empty:
            plt.plot(subset[col_mes].astype(str), subset['Taxa_Default_Pct'],
                     marker='o', linewidth=2.2, color=cor, label=f'Taxa de Default — {base}')

    media_total = (df_train[col_target].sum() + df_test[col_target].sum() + df_oot[col_target].sum()) / \
                  (len(df_train) + len(df_test) + len(df_oot)) * 100
    plt.axhline(media_total, color='#7f8c8d', linestyle=':', linewidth=1.5,
                label=f'Média Global ({media_total:.2f}%)')

    plt.title('Target Drift — Evolução da Taxa de Default por Safra (Mês de Decisão)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Safra de Decisão (MONTH)', fontsize=11, fontweight='bold')
    plt.ylabel('Taxa de Default (%)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=10)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()

    return df_consolidado


def plot_comparativo_psi_features(X_test, X_oot, features_list, bins=10, figsize=(15, 4)):
    """
    Plota histogramas de densidade comparativos entre Teste e OOT para uma lista de features
    e exibe o valor de PSI de cada uma no título.
    """
    n_feats = len(features_list)
    fig, axes = plt.subplots(1, n_feats, figsize=(figsize[0], figsize[1]))
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_list):
        s_test = X_test[feat].dropna()
        s_oot = X_oot[feat].dropna()
        psi_val = calcular_psi(s_test, s_oot, bins=bins)

        # Truncamento de percentis para visualização limpa em dados assimétricos
        p_low = min(np.percentile(s_test, 1), np.percentile(s_oot, 1))
        p_high = max(np.percentile(s_test, 99), np.percentile(s_oot, 99))

        sns.kdeplot(s_test[(s_test >= p_low) & (s_test <= p_high)], ax=ax,
                    color='#1B365D', linewidth=2.0, label='Teste (In-Time)', fill=True, alpha=0.25)
        sns.kdeplot(s_oot[(s_oot >= p_low) & (s_oot <= p_high)], ax=ax,
                    color='#8B0000', linewidth=2.0, linestyle='--', label='OOT', fill=True, alpha=0.25)

        status = 'Estável' if psi_val < 0.10 else ('Alerta' if psi_val < 0.25 else 'Crítico')
        ax.set_title(f'{feat}\nPSI = {psi_val:.4f} ({status})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor da Feature', fontsize=10)
        ax.set_ylabel('Densidade', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc')

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def gerar_tabela_decis_ks(y_true, prob, n_decis=10):
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
    plt.show()

    return tabela


def simular_lucro_e_curva_tradeoff(y_true, prob, valores_credito, margem_lucro=0.15, threshold_otimizado=0.7425):
    """
    Simula o lucro financeiro em OOT no threshold ótimo e gera o gráfico
    de trade-off entre Taxa de Aprovação vs. Taxa de Risco da Carteira Aprovada.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    y_true_arr = np.array(y_true)
    valores_arr = np.array(valores_credito)

    taxas_aprovacao = []
    riscos_carteira = []
    lucros = []

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
            tx_aprov = 0.0
            tx_risco = 0.0
            lucro_total = 0.0

        taxas_aprovacao.append(tx_aprov)
        riscos_carteira.append(tx_risco)
        lucros.append(lucro_total)

    # Ponto de operação no threshold selecionado
    aprov_op = prob < threshold_otimizado
    tx_aprov_op = (np.sum(aprov_op) / len(prob)) * 100
    tx_risco_op = (np.sum(aprov_op & (y_true_arr == 1)) / max(np.sum(aprov_op), 1)) * 100
    lucro_op = np.sum(valores_arr[aprov_op & (y_true_arr == 0)] * margem_lucro) - np.sum(valores_arr[aprov_op & (y_true_arr == 1)])

    print('=' * 75)
    print(f' SIMULAÇÃO DE RESULTADO FINANCEIRO NO OOT (Threshold = {threshold_otimizado:.4f}) ')
    print('=' * 75)
    print(f' Taxa de Aprovação da Carteira : {tx_aprov_op:.2f}% ({np.sum(aprov_op):,} contratos)')
    print(f' Taxa de Default dos Aprovados : {tx_risco_op:.2f}%')
    print(f' Lucro Financeiro Estimado     : R$ {lucro_op:,.2f}')
    print('=' * 75)

    # Gráfico de Trade-off
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    linha1 = ax1.plot(thresholds, taxas_aprovacao, color='#1B365D', linewidth=2.3, label='Taxa de Aprovação (%)')
    linha2 = ax2.plot(thresholds, riscos_carteira, color='#8B0000', linewidth=2.3, linestyle='--', label='Default na Carteira Aprovada (%)')

    ax1.axvline(threshold_otimizado, color='#2c3e50', linestyle=':', linewidth=1.8,
                label=f'Threshold Adotado ({threshold_otimizado:.4f})')

    ax1.set_xlabel('Ponto de Corte de Risco (Threshold)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Taxa de Aprovação (%)', color='#1B365D', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Taxa de Inadimplência Aprovada (%)', color='#8B0000', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1B365D')
    ax2.tick_params(axis='y', labelcolor='#8B0000')

    linhas = linha1 + linha2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc='center left', frameon=True, facecolor='#ffffff', edgecolor='#dcdcdc', fontsize=9.5)

    plt.title('Curva de Trade-off — Taxa de Concessão vs. Risco da Carteira OOT', fontsize=13, fontweight='bold', pad=14)
    ax1.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
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

class HardcodedStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador de Stacking pré-ajustado (zero overhead computacional).
    Combina linearmente as probabilidades previstas por estimadores base pré-treinados
    utilizando pesos estáticos (coeficientes) e intercepto de uma Regressão Logística.

    Parâmetros
    ----------
    estimators : list of tuple
        Lista de tuplas (nome, estimator) com os modelos base já treinados.
    weights : list or np.ndarray
        Pesos (coeficientes) atribuídos a cada estimador base.
    intercept : float
        Termo de intercepto da regressão logística meta-learner.
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
        
        # Matriz (n_samples, n_estimators)
        P = np.column_stack(prob_cols)
        
        # Log-odds = intercept + sum(w_i * p_i)
        z = self.intercept + P.dot(self.weights)
        
        # Função sigmoide numericamente estável
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)


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

    dest = Path(diretorio_destino)
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

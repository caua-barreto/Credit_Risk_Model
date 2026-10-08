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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import kagglehub

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
    agg_map: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Para cada familia depth 1, carrega apenas as colunas necessarias,
    agrega por case_id e faz LEFT JOIN no df_master.
    Processa um arquivo por vez para controlar RAM.

    Returns
    -------
    pd.DataFrame
        Master table enriquecida com agregacoes depth 1/2.
    """
    agg_map = agg_map or AGG_MAP_DEPTH_1
    linhas_originais = df_master.shape[0]
    total_cols = 0

    for familia in sorted(familias):
        cols_familia = [c for c, f in col_to_familia.items() if f == familia]
        arquivos = sorted(grupos_tabelas[familia])
        print(f"\n-> {familia} ({len(cols_familia)} colunas, {len(arquivos)} arquivos)")

        agg_dict = {col: agg_map[col] for col in cols_familia}
        lista_agg = []

        for arq in arquivos:
            df_part = pq.read_table(arq, columns=['case_id'] + cols_familia).to_pandas()
            df_part['case_id'] = df_part['case_id'].astype('int32')
            for c in df_part.select_dtypes(include=['float64']).columns:
                df_part[c] = df_part[c].astype('float32')

            df_part_agg = df_part.groupby('case_id').agg(agg_dict)
            df_part_agg.columns = [f"{col}_{func}" for col, func in df_part_agg.columns]
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


def analisar_multicolinearidade_pares(df, limite=0.80):
    # Calcula a matriz de correlação (em valores absolutos)
    matriz_corr = df.select_dtypes(include=[np.number]).corr().abs()
    
    # Pega apenas o triângulo superior para evitar pares duplicados (ex: A-B e B-A)
    triangulo_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    
    # Filtra e lista os pares acima do limite
    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna]) 
        for coluna in triangulo_sup.columns 
        for linha in triangulo_sup.index 
        if triangulo_sup.loc[linha, coluna] > limite
    ]
    
    print(f"--- Pares com Correlação > {limite} ---")
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        print(f"{par[0]} & {par[1]}: {par[2]:.3f}")

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
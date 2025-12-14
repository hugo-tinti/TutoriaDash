from dash import Dash, html, dcc, Input, Output, State, callback_context, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta
import tempfile
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# ==========================================
# FLASK-LOGIN IMPORTS
# ==========================================
from flask import Flask, request, redirect, session, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import secrets
import hashlib
import json
from typing import Optional, List, Dict, Tuple

# ==========================================

# LOGGING (mejora: trazabilidad sin cambiar análisis)
# ==========================================
import logging

LOG_LEVEL = os.getenv("CAI_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cai-sport-science")
# FPDF IMPORT PARA EXPORTACIÓN PDF
# ==========================================
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    # Crear clase dummy para evitar errores de nombre
    class FPDF:
        pass
    logger.warning("FPDF no instalado. La exportación PDF no estará disponible.")
    logger.warning("Instala con: pip install fpdf")
    logger.warning("El dashboard funcionará normalmente sin exportación PDF.")


# ==========================================
"""
===================================================================================
CAI SPORT SCIENCE PLATFORM - VERSIÓN CON DASH BOOTSTRAP COMPONENTS
===================================================================================

MEJORAS IMPLEMENTADAS CON BOOTSTRAP:

1. DISEÑO RESPONSIVO COMPLETO:
   - Sistema de grillas con dbc.Row y dbc.Col
   - Breakpoints: xs (móvil), sm, md, lg, xl
   - Sidebar: 12 columnas en móvil, 3 en desktop
   - Main area: 12 columnas en móvil, 9 en desktop

2. COMPONENTES BOOTSTRAP:
   - dbc.Container: Contenedor principal fluido
   - dbc.Card: Tarjetas para KPIs y paneles
   - dbc.Row/Col: Sistema de grillas responsivo
   - Clases CSS de Bootstrap: mb-3, fw-bold, text-uppercase, etc.

3. TEMA Y ESTILOS:
   - Bootstrap 5 theme aplicado
   - Font Awesome icons incluidos
   - Mantiene gradientes y colores personalizados del diseño premium

4. COMPATIBILIDAD:
   - Mantiene todas las funcionalidades originales
   - Callbacks sin modificar
   - Mejora la visualización en dispositivos móviles

REQUISITOS:
   pip install dash-bootstrap-components

USO:
   python gps_dashboard_bootstrap.py

===================================================================================
"""



# ==========================================
# ESTILOS & CONSTANTES UI (Premium)
# ==========================================

COLORS = {
    'red': '#DC2626',
    'blue': '#2563EB',
    'blue_light': '#93C5FD', # Para barras bajo promedio
    'green': '#059669',
    'amber': '#D97706',
    'slate_900': '#0F172A',
    'slate_800': '#1E293B',
    'slate_700': '#334155',
    'slate_600': '#475569',
    'slate_500': '#64748B',
    'slate_400': '#94A3B8',
    'slate_200': '#E2E8F0',
    'bg': '#F8FAFC',
    'panel': '#FFFFFF',
}

GRAPH_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
}

# ==========================================
# AYUDA CONTEXTUAL (Sport Science) - MODALES
# ==========================================

HELP_TEXTS = {
    'monitoreo': """
### Monitoreo de carga (día a día)
- Objetivo: ver tendencia y detectar picos de carga externa.
- Recomendación: leer siempre con *contexto* (MD, minutos, tipo de tarea).
- Señal operativa: pico no planificado + fatiga reportada = ajustar dosis del próximo estímulo similar.
""",

    'acwr': """
### ACWR & Cuadrantes (banderas, no diagnóstico)
- ACWR usa agudo 7 / crónico 28 (etiquetas Bajo/Óptimo/Moderado/Alto).
- Cuadrantes usan Z-score (atípico vs período) y un índice combinado para priorizar seguimiento.
- Uso CT: priorizar casos (alto/critico) y pedir contexto (minutos, dolor, retorno, tarea).
""",

    'sesion': """
### Reporte de sesión
- Resume el día por fecha: promedios por posición + comparativa por variable.
- Uso CT: cierre del día en 5 líneas (qué se hizo, quién alto/bajo, banderas y decisión para mañana).
""",

    'control': """
### Control de partidos
- Es una vista rápida del archivo de partidos cargado.
- Uso: validar que el Excel correcto y completo entró antes de sacar conclusiones.
""",

    'zonas': """
### Zonas de carga
- Lectura tipo semáforo por variable para decidir rápido.
- Uso CT/PF: zona alta sostenida -> controlar exposición del próximo estímulo similar; zona baja crónica -> planear exposición.
""",
}


def build_help_modal(tab_key: str, title: str = 'Ayuda'):
    """Crea botón + modal de ayuda contextual por tab (no altera cálculos ni outputs)."""
    md = HELP_TEXTS.get(tab_key, 'Sin ayuda disponible para esta sección.')

    btn = dbc.Button(
        'Ayuda',
        id={'type': 'help-open', 'tab': tab_key},
        color='secondary',
        outline=True,
        size='sm',
        className='ms-2'
    )

    modal = dbc.Modal(
        id={'type': 'help-modal', 'tab': tab_key},
        is_open=False,
        size='lg',
        scrollable=True,
        children=[
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(dcc.Markdown(md, link_target='_blank')),
            dbc.ModalFooter(
                dbc.Button('Cerrar', id={'type': 'help-close', 'tab': tab_key}, color='primary')
            )
        ]
    )

    return html.Span([btn, modal])


def safe_pct_change(value, ref):
    # Retorna % cambio vs referencia, o NaN si no es computable.
    try:
        v = float(value)
        r = float(ref)
        if not np.isfinite(v) or not np.isfinite(r) or r == 0:
            return np.nan
        return (v - r) / r * 100.0
    except Exception:
        return np.nan

# ==========================================
# CONFIGURACIÓN ANÁLISIS DE RIESGO
# ==========================================

ROLL_WINDOWS = [3, 7, 14, 21, 28]
N_LAGS = 7
ID_COL = "Atleta"
DATE_COL = "Fecha"

vol_cols = [
    "Total_Tpo", "Distancia_Total", "Dis_16", "Dis_20", "Dis_25",
    "Tot_16", "Num_Sprint_25", "Mts_90pct_Vel_Max", "Mts_Acc_3",
    "Mts_Dcc_3", "Acc_3", "Dcc_3", "Pot_Met_20_Mts", "Pot_Met_55_Mts",
    "Total_Eff_20W", "Dist_Equivalente",
]

hybrid_cols = ["Tot_PL", "RHIE", "HIA_actions"]
intermit_cols = ["Avg_Power_Act", "Avg_Dur_Act", "Avg_Power_Pauses", "Avg_Dur_Pauses", "Pct_W_P"]
base_feature_cols = vol_cols + hybrid_cols + intermit_cols

# Mínimo de observaciones para Z-score confiable
MIN_OBSERVATIONS_ZSCORE = 10

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

# ==========================================
# DATA CONTRACT + AUDITORÍA (sin cambiar cálculos)
# ==========================================

# Nota: estas funciones NO cambian cálculos (rolling/EMA/lags/risk/ACWR).
# Solo agregan validación, warnings y trazabilidad (huella + metadata).

def _sha256_from_upload(contents: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    try:
        if not contents or (not isinstance(contents, str)) or (',' not in contents):
            return None, None
        _, content_string = contents.split(',', 1)
        decoded = base64.b64decode(content_string)
        return hashlib.sha256(decoded).hexdigest(), int(len(decoded))
    except Exception:
        return None, None


def build_audit_payload(
    contents: Optional[str],
    filename: Optional[str],
    df: pd.DataFrame,
    file_role: str,
    warnings: Optional[List[str]] = None,
) -> Dict:
    warnings = warnings or []
    sha, size_bytes = _sha256_from_upload(contents)

    date_min = None
    date_max = None
    athletes_unique = None

    try:
        if df is not None and (not df.empty) and 'Fecha' in df.columns:
            d = pd.to_datetime(df['Fecha'], errors='coerce')
            if d.notna().any():
                date_min = str(d.min().date())
                date_max = str(d.max().date())
    except Exception:
        pass

    try:
        if df is not None and (not df.empty) and 'Atleta' in df.columns:
            athletes_unique = int(df['Atleta'].nunique())
    except Exception:
        pass

    return {
        'role': file_role,
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sha256': sha,
        'size_bytes': size_bytes,
        'rows': int(len(df)) if df is not None else 0,
        'cols': sorted([str(c) for c in df.columns]) if df is not None else [],
        'date_min': date_min,
        'date_max': date_max,
        'athletes_unique': athletes_unique,
        'warnings': warnings,
    }


def validate_main_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], str]:
    # Return: (ok, warnings, block_message)
    if df is None or df.empty:
        return False, [], '❌ Archivo principal inválido: el archivo está vacío o no se pudo leer.'

    required = {'Fecha', 'Atleta'}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        return False, [], f"❌ Archivo principal inválido: faltan columnas obligatorias: {', '.join(missing)}."

    fechas = pd.to_datetime(df['Fecha'], errors='coerce')
    if int(fechas.notna().sum()) == 0:
        return False, [], '❌ Archivo principal inválido: la columna Fecha no pudo convertirse a fecha.'

    warnings: List[str] = []
    # Condicionales por tab (solo warnings, no bloquean)
    if 'MD' not in df.columns:
        warnings.append('⚠️ MD no disponible: se desactiva análisis histórico por MD (tab ACWR).')
    if 'Microciclo' not in df.columns:
        warnings.append('⚠️ Microciclo no disponible: agrupaciones por microciclo pueden estar limitadas.')
    if 'Tipo' not in df.columns:
        warnings.append('⚠️ Tipo no disponible: lectura contextual de sesiones puede estar limitada.')

    # Warnings institucionales (no bloquean): duplicados y valores negativos
    try:
        if {'Atleta', 'Fecha'}.issubset(df.columns):
            dup = int(df.duplicated(subset=['Atleta', 'Fecha']).sum())
            if dup > 0:
                warnings.append(f"⚠️ Duplicados Atleta-Fecha: {dup} filas. Club: consolidar sesiones o corregir carga.")
    except Exception:
        pass

    # Valores negativos en variables que deben ser >= 0 (sin inventar umbrales)
    nonneg_cols = [
        'Total_Tpo','Distancia_Total','Dis_16','Dis_20','Dis_25','Tot_16','Num_Sprint_25','Mts_90pct_Vel_Max',
        'Mts_Acc_3','Mts_Dcc_3','Acc_3','Dcc_3','Pot_Met_20_Mts','Pot_Met_55_Mts','Total_Eff_20W','Dist_Equivalente',
        'Tot_PL','RHIE','HIA_actions'
    ]
    try:
        cols_present = [c for c in nonneg_cols if c in df.columns]
        for c in cols_present:
            vals = pd.to_numeric(df[c], errors='coerce')
            nneg = int((vals < 0).sum())
            if nneg > 0:
                warnings.append(f"⚠️ {c}: {nneg} valores negativos. Club: revisar unidades/captura antes de decidir.")
    except Exception:
        pass

    return True, warnings, ''


def validate_positions_dataframe(dfpos: pd.DataFrame) -> Tuple[bool, List[str], str]:
    if dfpos is None or dfpos.empty:
        return False, [], '❌ Posiciones inválido: el archivo está vacío o no se pudo leer.'

    required = {'Atleta', 'Posición'}
    missing = sorted(list(required - set(dfpos.columns)))
    if missing:
        return False, [], f"❌ Posiciones inválido: faltan columnas obligatorias: {', '.join(missing)}."

    warnings: List[str] = []
    try:
        c = dfpos.groupby('Atleta')['Posición'].nunique(dropna=True)
        n_conf = int((c > 1).sum())
        if n_conf > 0:
            warnings.append(f'⚠️ Posiciones: {n_conf} atletas con múltiples posiciones; revisar consistencia (ver auditoría).')
    except Exception:
        pass

    return True, warnings, ''


def validate_matches_dataframe(dfm: pd.DataFrame) -> Tuple[bool, List[str], str]:
    if dfm is None or dfm.empty:
        return False, [], '❌ Control Partidos: archivo vacío o no se pudo leer.'
    return True, ['ℹ️ Control Partidos: vista previa para verificación de carga.'], ''


def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except:
        return str(x)

def fmt_num(x, nd=1):
    try:
        formatted = f"{x:,.{nd}f}"
        formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
        return formatted
    except:
        return str(x)

def load_and_prepare_data(contents, filename):
    """Carga Excel/CSV desde dcc.Upload y normaliza columnas básicas.

    Nota: evita cambios en los cálculos; esta función solo prepara el dataset.
    """
    if contents is None:
        return pd.DataFrame()
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
        else:
            return pd.DataFrame()
        
        df.columns = [str(col).strip() for col in df.columns]
        
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.dropna(subset=['Fecha'])
        
        return df
    except Exception as e:
        logger.exception(f"Error cargando archivo: {e}")
        return pd.DataFrame()

# ==========================================
# FUNCIONES ANÁLISIS CUADRANTES
# ==========================================

def calcular_zscore_seguro(df: pd.DataFrame, columna: str) -> pd.Series:
    """Calcula Z-score con manejo robusto de NaN y varianza cero."""
    if df is None or df.empty or columna is None or columna not in df.columns:
        return pd.Series(np.nan, index=getattr(df, 'index', None))

    valores = pd.to_numeric(df[columna], errors='coerce')
    mask = valores.notna()
    n_valid = int(mask.sum())

    out = pd.Series(np.nan, index=df.index, dtype='float64')
    if n_valid < MIN_OBSERVATIONS_ZSCORE:
        return out

    vals = valores[mask].astype('float64')
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1))

    if (not np.isfinite(sigma)) or sigma == 0.0:
        out.loc[mask] = 0.0
        return out

    out.loc[mask] = (vals - mu) / sigma
    return out

def calcular_zscore_historico_md(
    df_completo: pd.DataFrame,
    fecha_objetivo,
    valor_md,
    columna_x: str,
    columna_y: str,
    atleta: str | None = None,
):
    """Análisis histórico contextual por MD (Match Day) con validación y robustez."""
    if df_completo is None or df_completo.empty:
        return pd.DataFrame(), pd.DataFrame(), {'warning': 'Dataset vacío'}

    required = {'Fecha', 'MD', 'Atleta'}
    if not required.issubset(df_completo.columns):
        faltan = sorted(list(required - set(df_completo.columns)))
        return pd.DataFrame(), pd.DataFrame(), {'warning': f'Faltan columnas requeridas: {faltan}'}

    if columna_x not in df_completo.columns or columna_y not in df_completo.columns:
        return pd.DataFrame(), pd.DataFrame(), {'warning': 'Variables X/Y no disponibles en el dataset'}

    df = df_completo.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha'])

    df[columna_x] = pd.to_numeric(df[columna_x], errors='coerce')
    df[columna_y] = pd.to_numeric(df[columna_y], errors='coerce')

    if atleta and atleta != 'TODOS':
        df = df[df['Atleta'] == atleta]

    df_historico = df[df['MD'] == valor_md].copy()
    if df_historico.empty:
        return pd.DataFrame(), pd.DataFrame(), {'warning': f'Sin registros para MD={valor_md}'}

    fecha_dt = pd.to_datetime(fecha_objetivo)
    df_objetivo = df_historico[df_historico['Fecha'] == fecha_dt].copy()

    df_poblacion = df_historico[df_historico['Fecha'] != fecha_dt].copy()

    n_x = int(df_poblacion[columna_x].notna().sum())
    n_y = int(df_poblacion[columna_y].notna().sum())
    n_min = min(n_x, n_y)

    if n_min < MIN_OBSERVATIONS_ZSCORE:
        return pd.DataFrame(), df_objetivo, {
            'warning': f'Insuficientes registros históricos para MD={valor_md} (n={n_min} < {MIN_OBSERVATIONS_ZSCORE})'
        }

    pop_x = df_poblacion[columna_x].dropna().astype('float64')
    pop_y = df_poblacion[columna_y].dropna().astype('float64')

    stats_x = {
        'mean': float(pop_x.mean()),
        'std': float(pop_x.std(ddof=1)),
        'median': float(pop_x.median()),
        'n': int(pop_x.shape[0]),
    }
    stats_y = {
        'mean': float(pop_y.mean()),
        'std': float(pop_y.std(ddof=1)),
        'median': float(pop_y.median()),
        'n': int(pop_y.shape[0]),
    }

    sx = stats_x['std']
    sy = stats_y['std']

    df_historico['zscore_x'] = 0.0 if ((not np.isfinite(sx)) or sx == 0.0) else (df_historico[columna_x] - stats_x['mean']) / sx
    df_historico['zscore_y'] = 0.0 if ((not np.isfinite(sy)) or sy == 0.0) else (df_historico[columna_y] - stats_y['mean']) / sy

    df_historico['outlier_extremo'] = (
        (df_historico['zscore_x'].abs() > 4) |
        (df_historico['zscore_y'].abs() > 4)
    )

    df_historico['es_fecha_objetivo'] = df_historico['Fecha'] == fecha_dt

    stats_historicas = {
        'x': stats_x,
        'y': stats_y,
        'md_value': valor_md,
        'fecha_objetivo': str(fecha_dt.date()),
        'n_total': int(len(df_historico)),
    }

    return df_historico, df_objetivo, stats_historicas

def calcular_percentil(df, columna):
    """Calcula percentil de valores"""
    valores = df[columna].dropna()
    if len(valores) < 2:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[columna].apply(lambda x: stats.percentileofscore(valores, x, kind='rank') if pd.notna(x) else np.nan)

def asignar_color_cuadrante(row):
    """Asigna color según posición en cuadrante"""
    x_in_range = -1 <= row['zscore_x'] <= 1
    y_in_range = -1 <= row['zscore_y'] <= 1
    if x_in_range and y_in_range:
        return '#78909c'
    else:
        max_z = max(abs(row['zscore_x']), abs(row['zscore_y']))
        if max_z > 2:
            return '#8b0000'
        elif max_z > 1.5:
            return '#c62828'
        elif (row['zscore_x'] > 1) or (row['zscore_y'] > 1):
            return '#f57c00'
        else:
            return '#ffa726'

def asignar_cuadrante(row):
    """Determina cuadrante según Z-scores"""
    if row['zscore_x'] >= 0 and row['zscore_y'] >= 0:
        return 'Q1: Alto X, Alto Y'
    elif row['zscore_x'] < 0 and row['zscore_y'] >= 0:
        return 'Q2: Bajo X, Alto Y'
    elif row['zscore_x'] < 0 and row['zscore_y'] < 0:
        return 'Q3: Bajo X, Bajo Y'
    else:
        return 'Q4: Alto X, Bajo Y'

def calcular_indices_riesgo(df):
    """Calcula índice de riesgo combinado usando distancia euclidiana en espacio Z-score"""
    df['indice_riesgo_combinado'] = np.sqrt(df['zscore_x']**2 + df['zscore_y']**2)
    condiciones = [
        df['indice_riesgo_combinado'] > 2.5,
        df['indice_riesgo_combinado'] > 2.0,
        df['indice_riesgo_combinado'] > 1.5,
        df['indice_riesgo_combinado'] > 1.0
    ]
    categorias = ['CRÍTICO', 'MUY ALTO', 'ALTO', 'MODERADO']
    df['categoria_riesgo'] = np.select(condiciones, categorias, default='BAJO')
    return df

def graficar_cuadrantes_con_etiquetas(fig):
    """Añade cuadrantes, sombreados y etiquetas al gráfico"""
    # Sombreado por cuadrante
    fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=3, line=dict(width=0),
                  fillcolor="rgba(239, 83, 80, 0.12)", layer="below")
    fig.add_shape(type="rect", x0=-3, y0=0, x1=0, y1=3, line=dict(width=0),
                  fillcolor="rgba(200, 200, 200, 0.08)", layer="below")
    fig.add_shape(type="rect", x0=-3, y0=-3, x1=0, y1=0, line=dict(width=0),
                  fillcolor="rgba(102, 187, 106, 0.12)", layer="below")
    fig.add_shape(type="rect", x0=0, y0=-3, x1=3, y1=0, line=dict(width=0),
                  fillcolor="rgba(200, 200, 200, 0.08)", layer="below")
    
    # Zona central ±1σ
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="#78909c", width=2.0, dash="dash"),
                  fillcolor="rgba(224,224,224,0.15)", layer="below")

    # Ejes y guías
    fig.add_hline(y=0, line_dash="solid", line_color="#1a1a1a", line_width=2.2, opacity=0.7)
    fig.add_vline(x=0, line_dash="solid", line_color="#1a1a1a", line_width=2.2, opacity=0.7)
    for val in [1.5, 2.0]:
        for s in [val, -val]:
            fig.add_hline(y=s, line_dash="dot", line_color="#bdbdbd", line_width=1, opacity=0.35)
            fig.add_vline(x=s, line_dash="dot", line_color="#bdbdbd", line_width=1, opacity=0.35)

    # Etiquetas internas
    internas = [
        {'x': 1.8, 'y': 1.8, 'text': 'Q1<br>Alto-Alto', 'color': '#d32f2f'},
        {'x': -1.8, 'y': 1.8, 'text': 'Q2<br>Bajo-Alto', 'color': '#424242'},
        {'x': -1.8, 'y': -1.8, 'text': 'Q3<br>Bajo-Bajo', 'color': '#2e7d32'},
        {'x': 1.8, 'y': -1.8, 'text': 'Q4<br>Alto-Bajo', 'color': '#424242'},
    ]
    for e in internas:
        fig.add_annotation(x=e['x'], y=e['y'], text=e['text'], showarrow=False,
                           font=dict(size=11, color=e['color'], family='Inter'),
                           bgcolor='rgba(255,255,255,0.85)', bordercolor='#e0e0e0',
                           borderwidth=1.2, borderpad=6)

    return fig

# ==========================================
# FUNCIONES FEATURE ENGINEERING (ACWR)
# ==========================================

def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL])

    cols_present = [c for c in base_feature_cols if c in df.columns]
    if cols_present:
        df[cols_present] = df[cols_present].apply(pd.to_numeric, errors='coerce')

    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ID_COL, DATE_COL])
    for col in base_feature_cols:
        if col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            roll = (
                df.groupby(ID_COL)[col]
                .rolling(window=w, min_periods=1)
                .agg(["sum", "mean", "std"])
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll{w}_sum"] = roll["sum"]
            df[f"{col}_roll{w}_mean"] = roll["mean"]
            df[f"{col}_roll{w}_std"] = roll["std"]
            df[f"{col}_roll{w}_cv"] = (
                df[f"{col}_roll{w}_std"] /
                (df[f"{col}_roll{w}_mean"].replace(0, np.nan))
            )
    return df

def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ID_COL, DATE_COL])
    spans = [3, 7, 21]
    for col in base_feature_cols:
        if col not in df.columns:
            continue
        for span in spans:
            df[f"{col}_ema{span}"] = (
                df.groupby(ID_COL)[col]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ID_COL, DATE_COL])
    lag_cols = [
        "Distancia_Total", "Dis_20", "Dis_25", "Tot_16", "Num_Sprint_25",
        "Mts_Acc_3", "Mts_Dcc_3", "Tot_PL", "RHIE", "HIA_actions"
    ]
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in range(1, N_LAGS + 1):
            df[f"{col}_lag{lag}"] = df.groupby(ID_COL)[col].shift(lag)
    return df

def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    risk_feats = [
        "Distancia_Total_roll7_mean", "Dis_20_roll7_mean", "Dis_25_roll7_mean",
        "Mts_Acc_3_roll7_sum", "Mts_Dcc_3_roll7_sum", "Tot_PL_roll7_sum",
        "RHIE_roll7_sum", "HIA_actions_roll7_sum"
    ]
    risk_feats = [f for f in risk_feats if f in df.columns]
    
    for col in risk_feats:
        mean_col = df.groupby(ID_COL)[col].transform("mean")
        std_col = df.groupby(ID_COL)[col].transform("std")
        z_col = (df[col] - mean_col) / std_col.replace(0, np.nan)
        df[f"{col}_z"] = z_col
    
    z_cols = [f"{c}_z" for c in risk_feats]
    z_matrix = df[z_cols].fillna(0)
    z_positive = z_matrix.clip(lower=0)
    df["z_pos_mean"] = z_positive.mean(axis=1)
    
    if df["z_pos_mean"].max() > 0:
        scaler = MinMaxScaler(feature_range=(0, 100))
        df["risk_score"] = scaler.fit_transform(df[["z_pos_mean"]])
    else:
        df["risk_score"] = 0
    
    return df

def calculate_acwr(df: pd.DataFrame, metric: str = "Distancia_Total") -> pd.DataFrame:
    """Calcula ACWR (Acute:Chronic Workload Ratio) de forma robusta."""
    df = df.copy()
    df = df.sort_values([ID_COL, DATE_COL])

    if metric not in df.columns:
        df['ACWR'] = np.nan
        df['ACWR_Risk'] = None
        df['Acute_Load'] = np.nan
        df['Chronic_Load'] = np.nan
        return df

    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    df['Acute_Load'] = (
        df.groupby(ID_COL)[metric]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['Chronic_Load'] = (
        df.groupby(ID_COL)[metric]
        .rolling(window=28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    denom = df['Chronic_Load'].where((df['Chronic_Load'].notna()) & (df['Chronic_Load'] != 0), np.nan)
    df['ACWR'] = df['Acute_Load'] / denom

    # Sanitiza inf y -inf
    df['ACWR'] = pd.to_numeric(df['ACWR'], errors='coerce')
    df.loc[~np.isfinite(df['ACWR']), 'ACWR'] = np.nan

    df['ACWR_Risk'] = pd.cut(
        df['ACWR'],
        bins=[0, 0.8, 1.3, 1.5, np.inf],
        labels=['🔵 Bajo', '🟢 Óptimo', '🟡 Moderado', '🔴 Alto']
    )

    return df

def build_feature_dataset(df: pd.DataFrame, acwr_metric: str = "Distancia_Total") -> pd.DataFrame:
    """Construye el dataset con features y calcula ACWR"""
    df = prepare_base_dataframe(df)
    df = add_rolling_features(df)
    df = add_ema_features(df)
    df = add_lag_features(df)
    df = add_risk_score(df)
    df = calculate_acwr(df, acwr_metric)
    return df

# ==========================================
# INICIALIZAR APP
# ==========================================


# ==========================================
# FUNCIONES DE EXPORTACIÓN PDF
# ==========================================


def save_plotly_figure_to_temp(fig):
    """Guarda una figura de Plotly como imagen temporal y retorna la ruta.
    Retorna None si falla."""
    try:
        # Crear archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_path = temp_file.name
        temp_file.close()

        # Intentar guardar la figura
        try:
            pio.write_image(fig, temp_path, width=1200, height=600, scale=2)
            return temp_path
        except Exception as e:
            logger.warning(f"No se pudo guardar gráfica (requiere kaleido): {e}")
            # Limpiar archivo temporal si falló
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    except Exception as e:
        logger.warning(f"Error creando archivo temporal: {e}")
        return None


if PDF_AVAILABLE:
    # Funciones PDF solo disponibles si FPDF está instalado
    class PDFReport(FPDF):
        """Clase personalizada para generar reportes PDF"""

        def header(self):
            # Logo y título
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'CAI SPORT SCIENCE - REPORTE GPS', 0, 1, 'C')
            self.set_font('Arial', '', 10)
            self.cell(0, 5, f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 14)
            self.set_fill_color(220, 38, 38)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, title, 0, 1, 'L', True)
            self.set_text_color(0, 0, 0)
            self.ln(4)

        def add_data_table(self, df, title="Datos"):
            """Agrega una tabla de datos al PDF"""
            self.chapter_title(title)
            self.set_font('Arial', '', 9)

            # Limitar columnas para que quepan
            max_cols = 6
            cols = df.columns[:max_cols].tolist()

            # Header
            self.set_fill_color(15, 23, 42)
            self.set_text_color(255, 255, 255)
            col_width = 190 / len(cols)
            for col in cols:
                self.cell(col_width, 8, str(col)[:15], 1, 0, 'C', True)
            self.ln()

            # Data
            self.set_text_color(0, 0, 0)
            self.set_fill_color(248, 250, 252)

            for i, row in df.head(20).iterrows():
                for col in cols:
                    val = str(row[col])[:15]
                    fill = i % 2 == 0
                    self.cell(col_width, 7, val, 1, 0, 'L', fill)
                self.ln()

    def generar_pdf_monitoreo(df, atleta="TODOS", metrica="Distancia_Total"):
        """Genera PDF del tab de monitoreo con gráficas"""
        if not PDF_AVAILABLE:
            return None

        pdf = PDFReport()
        pdf.add_page()

        # Título del reporte
        pdf.chapter_title("REPORTE DE MONITOREO DE CARGA")

        # Información general
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f'Atleta: {atleta}', 0, 1)
        pdf.cell(0, 8, f'Métrica analizada: {metrica}', 0, 1)
        pdf.cell(0, 8, f'Registros totales: {len(df)}', 0, 1)
        pdf.ln(5)

        # Filtrar datos
        df_filt = df.copy()
        if atleta != "TODOS" and 'Atleta' in df.columns:
            df_filt = df_filt[df_filt['Atleta'] == atleta]

        # GRÁFICA: Serie temporal de la métrica
        if metrica in df_filt.columns and 'Fecha' in df_filt.columns and len(df_filt) > 0:
            try:
                # Crear gráfica
                df_plot = df_filt.sort_values('Fecha')
                fig = go.Figure()

                if atleta == "TODOS" and 'Atleta' in df_filt.columns:
                    # Gráfica por atleta
                    for atleta_name in df_plot['Atleta'].unique()[:5]:  # Max 5 atletas
                        df_atleta = df_plot[df_plot['Atleta'] == atleta_name]
                        fig.add_trace(go.Scatter(
                            x=df_atleta['Fecha'],
                            y=df_atleta[metrica],
                            mode='lines+markers',
                            name=atleta_name,
                            line=dict(width=2)
                        ))
                else:
                    # Gráfica individual
                    fig.add_trace(go.Scatter(
                        x=df_plot['Fecha'],
                        y=df_plot[metrica],
                        mode='lines+markers',
                        name=metrica,
                        line=dict(width=3, color='#DC2626'),
                        marker=dict(size=8)
                    ))

                fig.update_layout(
                    title=f'Evolución de {metrica}',
                    xaxis_title='Fecha',
                    yaxis_title=metrica,
                    template='plotly_white',
                    height=400,
                    showlegend=True,
                    font=dict(family='Arial', size=12)
                )

                # Guardar gráfica como imagen
                img_path = save_plotly_figure_to_temp(fig)
                if img_path:
                    pdf.chapter_title("GRÁFICA DE EVOLUCIÓN")
                    pdf.image(img_path, x=10, y=None, w=190)
                    pdf.ln(5)
                    # Limpiar archivo temporal
                    os.remove(img_path)
            except Exception as e:
                logger.warning(f"Error generando gráfica en PDF: {e}")

        # Estadísticas
        if metrica in df_filt.columns:
            pdf.chapter_title("ESTADÍSTICAS")
            stats_data = {
                'Media': df_filt[metrica].mean(),
                'Mediana': df_filt[metrica].median(),
                'Desv. Est.': df_filt[metrica].std(),
                'Mínimo': df_filt[metrica].min(),
                'Máximo': df_filt[metrica].max()
            }

            for key, val in stats_data.items():
                pdf.cell(0, 8, f'{key}: {val:.2f}', 0, 1)
            pdf.ln(5)

        # Tabla de datos
        if not df_filt.empty:
            cols_to_show = ['Fecha', 'Atleta', metrica] if all(c in df_filt.columns for c in ['Fecha', 'Atleta', metrica]) else df_filt.columns[:6]
            pdf.add_data_table(df_filt[cols_to_show], "ÚLTIMAS 20 SESIONES")

        # Guardar en memoria
        pdf_output = io.BytesIO()
        pdf_data = pdf.output(dest='S').encode('latin-1')
        pdf_output.write(pdf_data)
        pdf_output.seek(0)

        return pdf_output


    def generar_pdf_sesion(df, fecha_seleccionada, variables):
        """Genera PDF del reporte de sesión con gráficas"""
        if not PDF_AVAILABLE:
            return None

        pdf = PDFReport()
        pdf.add_page()

        pdf.chapter_title("REPORTE DE SESIÓN DIARIA")

        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f'Fecha: {fecha_seleccionada}', 0, 1)
        pdf.cell(0, 8, f'Variables analizadas: {", ".join(variables)}', 0, 1)
        pdf.ln(5)

        # Filtrar datos del día
        df_day = df[df['Fecha'].dt.date == pd.to_datetime(fecha_seleccionada).date()].copy()

        if not df_day.empty:
            # GRÁFICA: Comparativa de atletas
            if variables and len(variables) > 0 and 'Atleta' in df_day.columns:
                try:
                    # Usar la primera variable para la gráfica
                    var_principal = variables[0]
                    if var_principal in df_day.columns:
                        # Crear gráfica de barras
                        df_plot = df_day.sort_values(var_principal, ascending=False).head(10)

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=df_plot['Atleta'],
                            y=df_plot[var_principal],
                            marker=dict(
                                color=df_plot[var_principal],
                                colorscale='Reds',
                                showscale=False
                            ),
                            text=df_plot[var_principal].round(1),
                            textposition='outside'
                        ))

                        fig.update_layout(
                            title=f'Comparativa: {var_principal}',
                            xaxis_title='Atleta',
                            yaxis_title=var_principal,
                            template='plotly_white',
                            height=400,
                            font=dict(family='Arial', size=12),
                            showlegend=False
                        )

                        # Guardar gráfica como imagen
                        img_path = save_plotly_figure_to_temp(fig)
                        if img_path:
                            pdf.chapter_title("GRÁFICA COMPARATIVA DE ATLETAS")
                            pdf.image(img_path, x=10, y=None, w=190)
                            pdf.ln(5)
                            # Limpiar archivo temporal
                            os.remove(img_path)
                except Exception as e:
                    logger.warning(f"Error generando gráfica en PDF: {e}")

            pdf.chapter_title(f"DATOS DEL DÍA ({len(df_day)} atletas)")

            # Promedios
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, 'PROMEDIOS DEL EQUIPO:', 0, 1)
            pdf.set_font('Arial', '', 10)

            for var in variables:
                if var in df_day.columns:
                    mean_val = df_day[var].mean()
                    pdf.cell(0, 7, f'{var}: {mean_val:.2f}', 0, 1)
            pdf.ln(5)

            # Tabla de atletas
            cols_to_show = ['Atleta'] + [v for v in variables if v in df_day.columns]
            pdf.add_data_table(df_day[cols_to_show], "DATOS POR ATLETA")
        else:
            pdf.cell(0, 8, 'Sin datos para esta fecha', 0, 1)

        pdf_output = io.BytesIO()
        pdf_data = pdf.output(dest='S').encode('latin-1')
        pdf_output.write(pdf_data)
        pdf_output.seek(0)

        return pdf_output


    # ==========================================

else:
    # Funciones dummy cuando FPDF no está disponible
    def generar_pdf_monitoreo(df, atleta='TODOS', metrica='Distancia_Total'):
        return None
    
    def generar_pdf_sesion(df, fecha_seleccionada, variables):
        return None

# RUTAS FLASK PARA DESCARGA PDF
# ==========================================


server = Flask(__name__)
server.config['SECRET_KEY'] = secrets.token_hex(16)

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username == 'admin':
        return User(username)
    return None

# ==========================================
# FLASK-LOGIN: RUTAS DE AUTENTICACIÓN
# ==========================================

@server.route('/login', methods=['GET', 'POST'])
def login_route():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == 'admin':
            user = User('admin')
            login_user(user)
            return redirect('/')
        else:
            return '''
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Login - CAI Sport Science</title>
                    <style>
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body {
                            font-family: \'Inter\', -apple-system, BlinkMacSystemFont, sans-serif;
                            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .login-container {
                            background: white;
                            padding: 60px 50px;
                            border-radius: 24px;
                            box-shadow: 0 20px 80px rgba(0, 0, 0, 0.4);
                            width: 100%;
                            max-width: 450px;
                            border-top: 6px solid #DC2626;
                        }
                        .logo-container {
                            text-align: center;
                            margin-bottom: 40px;
                        }
                        .logo-emoji {
                            font-size: 64px;
                            margin-bottom: 20px;
                        }
                        h1 {
                            font-size: 28px;
                            font-weight: 900;
                            color: #0F172A;
                            text-align: center;
                            margin-bottom: 12px;
                            letter-spacing: -0.5px;
                        }
                        .subtitle {
                            text-align: center;
                            color: #64748B;
                            font-size: 14px;
                            font-weight: 600;
                            margin-bottom: 40px;
                        }
                        .error-message {
                            background: #FEE2E2;
                            border: 2px solid #DC2626;
                            color: #7C2D12;
                            padding: 16px;
                            border-radius: 12px;
                            margin-bottom: 24px;
                            font-size: 14px;
                            font-weight: 700;
                            text-align: center;
                        }
                        .form-group {
                            margin-bottom: 24px;
                        }
                        label {
                            display: block;
                            font-size: 12px;
                            font-weight: 800;
                            color: #475569;
                            margin-bottom: 10px;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        }
                        input[type="text"], input[type="password"] {
                            width: 100%;
                            padding: 16px 18px;
                            border: 2px solid #E2E8F0;
                            border-radius: 12px;
                            font-size: 15px;
                            font-weight: 600;
                            color: #0F172A;
                            transition: all 0.3s;
                            font-family: \'Inter\', sans-serif;
                        }
                        input[type="text"]:focus, input[type="password"]:focus {
                            outline: none;
                            border-color: #DC2626;
                            box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.1);
                        }
                        button {
                            width: 100%;
                            padding: 18px;
                            background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
                            color: white;
                            border: none;
                            border-radius: 12px;
                            font-size: 15px;
                            font-weight: 900;
                            cursor: pointer;
                            transition: all 0.3s;
                            text-transform: uppercase;
                            letter-spacing: 1px;
                            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.3);
                        }
                        button:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 24px rgba(220, 38, 38, 0.4);
                        }
                        button:active {
                            transform: translateY(0);
                        }
                    </style>
                </head>
                <body>
                    <div class="login-container">
                        <div class="logo-container">
                            <div class="logo-emoji">⚽</div>
                            <h1>CAI SPORT SCIENCE</h1>
                            <div class="subtitle">Sistema de Análisis GPS Premium</div>
                        </div>
                        <div class="error-message">
                            ❌ Usuario o contraseña incorrectos
                        </div>
                        <form method="POST" action="/login">
                            <div class="form-group">
                                <label>👤 Usuario</label>
                                <input type="text" name="username" required autofocus placeholder="Ingresa tu usuario">
                            </div>
                            <div class="form-group">
                                <label>🔒 Contraseña</label>
                                <input type="password" name="password" required placeholder="Ingresa tu contraseña">
                            </div>
                            <button type="submit">🚀 Iniciar Sesión</button>
                        </form>
                    </div>
                </body>
                </html>
            '''

    return '''
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Login - CAI Sport Science</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: \'Inter\', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .login-container {
                    background: white;
                    padding: 60px 50px;
                    border-radius: 24px;
                    box-shadow: 0 20px 80px rgba(0, 0, 0, 0.4);
                    width: 100%;
                    max-width: 450px;
                    border-top: 6px solid #DC2626;
                }
                .logo-container {
                    text-align: center;
                    margin-bottom: 40px;
                }
                .logo-emoji {
                    font-size: 64px;
                    margin-bottom: 20px;
                }
                h1 {
                    font-size: 28px;
                    font-weight: 900;
                    color: #0F172A;
                    text-align: center;
                    margin-bottom: 12px;
                    letter-spacing: -0.5px;
                }
                .subtitle {
                    text-align: center;
                    color: #64748B;
                    font-size: 14px;
                    font-weight: 600;
                    margin-bottom: 40px;
                }
                .form-group {
                    margin-bottom: 24px;
                }
                label {
                    display: block;
                    font-size: 12px;
                    font-weight: 800;
                    color: #475569;
                    margin-bottom: 10px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                input[type="text"], input[type="password"] {
                    width: 100%;
                    padding: 16px 18px;
                    border: 2px solid #E2E8F0;
                    border-radius: 12px;
                    font-size: 15px;
                    font-weight: 600;
                    color: #0F172A;
                    transition: all 0.3s;
                    font-family: \'Inter\', sans-serif;
                }
                input[type="text"]:focus, input[type="password"]:focus {
                    outline: none;
                    border-color: #DC2626;
                    box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.1);
                }
                button {
                    width: 100%;
                    padding: 18px;
                    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 15px;
                    font-weight: 900;
                    cursor: pointer;
                    transition: all 0.3s;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    box-shadow: 0 4px 16px rgba(220, 38, 38, 0.3);
                }
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 24px rgba(220, 38, 38, 0.4);
                }
                button:active {
                    transform: translateY(0);
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <div class="logo-container">
                    <div class="logo-emoji">⚽</div>
                    <h1>CAI SPORT SCIENCE</h1>
                    <div class="subtitle">Sistema de Análisis GPS Premium</div>
                </div>
                <form method="POST" action="/login">
                    <div class="form-group">
                        <label>👤 Usuario</label>
                        <input type="text" name="username" required autofocus placeholder="Ingresa tu usuario">
                    </div>
                    <div class="form-group">
                        <label>🔒 Contraseña</label>
                        <input type="password" name="password" required placeholder="Ingresa tu contraseña">
                    </div>
                    <button type="submit">🚀 Iniciar Sesión</button>
                </form>
            </div>
        </body>
        </html>
    '''

@server.route('/logout')
@login_required
def logout_route():
    logout_user()
    return redirect('/login')

# ==========================================
# INICIALIZAR APP DASH CON FLASK SERVER
# ==========================================


@server.route('/download-pdf/<report_type>')
@login_required
def download_pdf(report_type):
    """Ruta para descargar PDFs generados"""
    # Verificar si FPDF está disponible
    if not PDF_AVAILABLE:
        return '''
            <html>
            <head>
                <title>PDF No Disponible</title>
                <style>
                    body { font-family: Arial, sans-serif; display: flex; justify-content: center;
                           align-items: center; min-height: 100vh;
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 0; }
                    .message { background: white; padding: 40px; border-radius: 20px;
                               box-shadow: 0 10px 40px rgba(0,0,0,0.3); text-align: center; max-width: 500px; }
                    h1 { color: #DC2626; margin-bottom: 20px; }
                    p { color: #475569; line-height: 1.6; margin: 10px 0; }
                    code { background: #f1f5f9; padding: 4px 8px; border-radius: 4px;
                           font-family: monospace; color: #DC2626; }
                    .icon { font-size: 64px; margin-bottom: 20px; }
                    a { display: inline-block; margin-top: 20px; padding: 12px 24px;
                        background: #DC2626; color: white; text-decoration: none;
                        border-radius: 8px; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class='message'>
                    <div class='icon'>⚠️</div>
                    <h1>Exportación PDF No Disponible</h1>
                    <p>La librería FPDF no está instalada.</p>
                    <p>Para habilitar la exportación a PDF, ejecuta:</p>
                    <p><code>pip install fpdf</code></p>
                    <p>Luego reinicia la aplicación.</p>
                    <a href='/'>← Volver al Dashboard</a>
                </div>
            </body>
            </html>
        ''', 503

    try:
        # Crear DataFrame de ejemplo con estructura correcta para evitar errores
        df_ejemplo = pd.DataFrame({
            'Fecha': [datetime.now() - timedelta(days=i) for i in range(10)],
            'Atleta': ['Atleta_' + str(i % 5) for i in range(10)],
            'Distancia_Total': [5000 + i*100 for i in range(10)],
            'Dis_20': [500 + i*10 for i in range(10)],
            'Dis_25': [200 + i*5 for i in range(10)]
        })
        
        pdf_file = None
        
        if report_type == 'monitoreo':
            pdf_file = generar_pdf_monitoreo(df_ejemplo, 'TODOS', 'Distancia_Total')
        elif report_type == 'acwr':
            pdf_file = generar_pdf_monitoreo(df_ejemplo, 'TODOS', 'Dis_20')
        elif report_type == 'sesion':
            fecha_ejemplo = datetime.now().date()
            variables_ejemplo = ['Distancia_Total', 'Dis_20', 'Dis_25']
            pdf_file = generar_pdf_sesion(df_ejemplo, fecha_ejemplo, variables_ejemplo)
        else:
            return f'<h1>Error: Tipo de reporte no válido: {report_type}</h1><a href="/">Volver</a>', 400

        if pdf_file is None:
            return '<h1>Error: No se pudo generar el PDF</h1><a href="/">Volver</a>', 500

        return send_file(
            pdf_file,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'reporte_{report_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        return f'<h1>Error generando PDF</h1><p>{str(e)}</p><a href="/">Volver</a>', 500


# ==========================================
# FLASK-LOGIN: CONFIGURACIÓN Y USUARIO
# ==========================================

app = Dash(__name__, server=server, suppress_callback_exceptions=True, url_base_pathname='/', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


# ==========================================
# LOG OPERATIVO (toma evidencia, sin cambiar cálculos)
# ==========================================

@app.callback(
    Output('stored-ops-log', 'data'),
    Output('ops-log-preview', 'children'),
    Input('ops-add', 'n_clicks'),
    State('ops-note', 'value'),
    State('ops-who', 'value'),
    State('stored-audit', 'data'),
    State('stored-ops-log', 'data'),
    State('main-tabs', 'value'),
    prevent_initial_call=True,
)
def update_ops_log(n_clicks, note, who, audit, log_data, active_tab):
    log_data = log_data or []
    note = (note or '').strip()
    who = (who or '').strip()
    if note == '':
        preview = dbc.Alert('Escribí una nota antes de registrar.', color='warning', style={'fontWeight': 800})
        return log_data, preview

    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tab': active_tab,
        'who': who,
        'note': note,
        'audit_sha256': (audit or {}).get('sha256'),
        'audit_filename': (audit or {}).get('filename'),
        'audit_rows': (audit or {}).get('rows'),
        'audit_date_min': (audit or {}).get('date_min'),
        'audit_date_max': (audit or {}).get('date_max'),
    }

    log_data = [entry] + log_data
    log_data = log_data[:200]  # límite local para performance UI

    preview = _render_ops_preview(log_data)
    return log_data, preview


def _render_ops_preview(log_data):
    if not log_data:
        return html.Div('—', style={'color': COLORS['slate_500'], 'fontWeight': 700})

    items = []
    for e in log_data[:5]:
        items.append(
            dbc.Card(
                dbc.CardBody([
                    html.Div(e.get('timestamp'), style={'color': COLORS['slate_600'], 'fontWeight': 800, 'fontSize': '12px'}),
                    html.Div(e.get('tab'), style={'color': COLORS['slate_700'], 'fontWeight': 900, 'fontSize': '12px'}),
                    html.Div(e.get('note'), style={'fontWeight': 650, 'fontSize': '12px'}),
                ]),
                style={'borderRadius': '12px', 'border': '1px solid ' + COLORS['slate_200'], 'marginBottom': '8px'}
            )
        )

    return html.Div(items)


@app.callback(
    Output('download-ops-log', 'data'),
    Input('ops-export', 'n_clicks'),
    State('stored-ops-log', 'data'),
    prevent_initial_call=True,
)
def export_ops_log(n_clicks, log_data):
    log_data = log_data or []
    if not log_data:
        return None

    # Export Excel institucional (incluye observación y profesional)
    df = pd.DataFrame(log_data)
    if df.empty:
        return None

    colmap = {
        'timestamp': 'Timestamp',
        'tab': 'Tab',
        'who': 'Profesional',
        'note': 'Observacion',
        'audit_filename': 'Archivo',
        'audit_sha256': 'SHA256',
        'audit_rows': 'Filas',
        'audit_date_min': 'Fecha_min',
        'audit_date_max': 'Fecha_max',
    }
    df = df.rename(columns=colmap)
    preferred = ['Timestamp','Tab','Profesional','Observacion','Archivo','SHA256','Filas','Fecha_min','Fecha_max']
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]
    filename = f"ops_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return dcc.send_bytes(lambda b: df.to_excel(b, index=False), filename)


# ==========================================
# UI: Auditoría de carga (sin cambiar cálculos)
# ==========================================

@app.callback(
    Output('audit-panel', 'children'),
    Input('stored-audit', 'data'),
)
def render_audit_panel(audit):
    if not audit:
        return html.Div('—', style={'color': COLORS['slate_500'], 'fontWeight': 700})

    lines = [
        html.Div([html.Strong('Archivo: '), str(audit.get('filename'))]),
        html.Div([html.Strong('Huella (sha256): '), str(audit.get('sha256'))]),
        html.Div([html.Strong('Cargado: '), str(audit.get('timestamp'))]),
        html.Div([html.Strong('Filas: '), str(audit.get('rows'))]),
    ]

    if audit.get('athletes_unique') is not None:
        lines.append(html.Div([html.Strong('Atletas únicos: '), str(audit.get('athletes_unique'))]))

    if audit.get('date_min') and audit.get('date_max'):
        lines.append(html.Div([html.Strong('Rango fechas: '), f"{audit.get('date_min')} → {audit.get('date_max')}" ]))

    warn = audit.get('warnings') or []
    if len(warn) > 0:
        return html.Div([
            dbc.Alert('⚠️ Warnings de datos', color='warning', style={'fontWeight': 900, 'marginBottom': '10px'}),
            html.Div(lines),
            html.Ul([html.Li(w) for w in warn], style={'marginTop': '10px'}),
        ])

    return html.Div([
        dbc.Alert('✅ Auditoría OK (sin warnings)', color='success', style={'fontWeight': 900, 'marginBottom': '10px'}),
        html.Div(lines),
    ])
app.title = "CAI Sport Science Platform | Premium Edition"

# ==========================================
# CSS GLOBAL (equivalente a html.Style, compatible con Dash)
# ==========================================
GLOBAL_CSS = r"""
/* Forzar uso de todo el ancho disponible (especialmente al ocultar sidebar) */
.container-fluid {
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Si usas dbc.Row con g-3, evita márgenes horizontales */
.row.g-3 {
    --bs-gutter-x: 0rem !important;
}
#main-col {
    max-width: 100% !important;
}
#main-area {
    width: 100% !important;
    max-width: 100% !important;
}
/* Asegurar que Plotly se adapte al ancho del contenedor */
.dash-graph, .js-plotly-plot, .plot-container {
    width: 100% !important;
}

/* Impresión / Exportar a PDF (usar el diálogo del navegador: Guardar como PDF) */
@media print {
    body {
        background: #FFFFFF !important;
    }
    .no-print, .modebar {
        display: none !important;
    }
    /* Evitar cortes incómodos en gráficos y tarjetas */
    .dash-graph, .js-plotly-plot, .plot-container, .card, .dbc {
        break-inside: avoid;
        page-break-inside: avoid;
    }
}
"""

# Inserta el CSS en el <head> sin cambiar el resto del HTML base de Dash
app.index_string = app.index_string.replace(
    "</head>",
    f"<style>{GLOBAL_CSS}</style></head>"
)


# ==========================================
# PROTECCIÓN DE RUTAS DASH
# ==========================================

@server.before_request
def check_login():
    if request.path.startswith('/_dash') or request.path.startswith('/assets'):
        if not current_user.is_authenticated:
            return redirect('/login')
    elif request.path == '/' and not current_user.is_authenticated:
        return redirect('/login')


# ==========================================
# LAYOUT PRINCIPAL
# ==========================================

app.layout = dbc.Container(
    fluid=True,
    className='p-0 m-0',
    style={
        'fontFamily': '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
        'backgroundColor': '#F8FAFC',
        'minHeight': '100vh',
        'margin': '0',
        'padding': '0'
    },
    children=[
        # Dummy output para callbacks clientside (no se muestra)
        html.Div(id='print-dummy', style={'display': 'none'}),

        html.Div(id='resize-dummy', style={'display': 'none'}),

        # STORES GLOBALES
        dcc.Store(id='stored-data'),
        dcc.Store(id='stored-positions'),
        dcc.Store(id='stored-risk-data'),
            dcc.Store(id='stored-audit'),
            dcc.Store(id='stored-ops-log', data=[]),
            dcc.Download(id='download-ops-log'),
        dcc.Store(id='stored-matches-data'),
        dcc.Store(id='sidebar-collapsed', data=False),
        
        # STORES INDEPENDIENTES POR GRÁFICO (Pattern Matching)
        dcc.Store(id={'type': 'graph-filters', 'index': 'monitoreo-timeseries'}),
        dcc.Store(id={'type': 'graph-filters', 'index': 'monitoreo-boxplot'}),
        dcc.Store(id={'type': 'graph-filters', 'index': 'cuadrante-general'}),
        dcc.Store(id={'type': 'graph-filters', 'index': 'cuadrante-historico'}),
        dcc.Store(id={'type': 'graph-filters', 'index': 'sesion-diaria'}), # Store para el nuevo tab
        
        # BARRA SUPERIOR
        html.Div(
            style={
                'background': 'linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)',
                'borderBottom': '4px solid #DC2626',
                'padding': '18px 40px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
                'position': 'sticky',
                'top': '0',
                'zIndex': '1000'
            },
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '28px'},
                    children=[
                        html.Button('☰', id='toggle-sidebar', n_clicks=0,
                                    style={'backgroundColor': '#DC2626', 'color': '#FFFFFF', 'border': 'none', 'borderRadius': '10px', 'padding': '12px 18px', 'fontSize': '22px', 'fontWeight': '700', 'cursor': 'pointer'}),
                        html.Img(src='/assets/escudo_cai.png', style={'height': '64px', 'width': 'auto'}),
                        html.Div([
                            html.H1('CLUB ATLÉTICO INDEPENDIENTE', style={'margin': '0', 'fontSize': '24px', 'fontWeight': '900', 'color': '#FFFFFF', 'letterSpacing': '0.08em', 'textTransform': 'uppercase'}),
                            html.P('⚡ Sport Science & GPS Performance Analytics • Premium Edition', style={'margin': '6px 0 0 0', 'fontSize': '13px', 'color': '#CBD5E1', 'fontWeight': '700'})
                        ])
                    ]
                ),
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '24px'},
                    children=[
                        html.Div('🏆 JEFE DEPTO. CIENCIAS', style={'background': 'linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)', 'color': '#FFFFFF', 'padding': '10px 20px', 'borderRadius': '10px', 'fontSize': '12px', 'fontWeight': '800'}),
                        html.Div(f"🕐 {datetime.now().strftime('%d/%m/%Y • %H:%M')}", style={'fontSize': '14px', 'color': '#94A3B8', 'fontWeight': '700'}),
                        html.A('🚪 Salir', href='/logout', style={'backgroundColor': '#475569', 'color': '#FFFFFF', 'padding': '10px 20px', 'borderRadius': '10px', 'fontSize': '12px', 'fontWeight': '800', 'textDecoration': 'none', 'cursor': 'pointer'})
                    ]
                )
            ]
        ),
        
        # CONTENEDOR PRINCIPAL CON BOOTSTRAP
        dbc.Row([
            # SIDEBAR
            dbc.Col([
                html.Div(
                    id='sidebar',
                    style={'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'},
                    children=[
                        html.Div(
                            style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #E2E8F0', 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.12)'},
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '28px', 'paddingBottom': '20px', 'borderBottom': '3px solid #DC2626'},
                                    children=[
                                        html.Div('⚙️', style={'fontSize': '28px'}),
                                        html.H3('CONFIGURACIÓN', style={'margin': '0', 'fontSize': '16px', 'fontWeight': '900', 'color': '#0F172A', 'letterSpacing': '0.1em', 'textTransform': 'uppercase'})
                                    ]
                                ),
                                html.Label('Archivo Principal de Datos', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '800', 'color': '#475569', 'marginBottom': '12px', 'textTransform': 'uppercase'}),
                                dcc.Upload(
                                    id='main-file',
                                    children=html.Div(
                                        style={'border': '3px dashed #94A3B8', 'borderRadius': '16px', 'padding': '32px 24px', 'textAlign': 'center', 'cursor': 'pointer', 'background': 'linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%)'},
                                        children=[
                                            html.Div('📊', style={'fontSize': '48px', 'marginBottom': '16px'}),
                                            html.Div('Arrastra archivo aquí', style={'fontSize': '15px', 'fontWeight': '800', 'color': '#1E293B', 'marginBottom': '6px'}),
                                            html.Div('Excel (.xlsx) o CSV (.csv)', style={'fontSize': '12px', 'color': '#64748B', 'fontWeight': '600'})
                                        ]
                                    ),
                                    multiple=False
                                ),
                                html.Div(id='main-file-status', style={'marginTop': '12px', 'fontSize': '13px', 'fontWeight': '700'}),
                                html.Div(style={'height': '16px'}),

                                html.Label('📌 Control Partidos (Excel)', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '800', 'color': '#475569', 'marginBottom': '12px', 'textTransform': 'uppercase'}),
                                dcc.Upload(
                                    id='matches-file',
                                    children=html.Div(
                                        style={'border': '2px dashed #94A3B8', 'borderRadius': '14px', 'padding': '18px 16px', 'textAlign': 'center', 'cursor': 'pointer', 'background': 'linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%)'},
                                        children=[
                                            html.Div('🏟️', style={'fontSize': '34px', 'marginBottom': '10px'}),
                                            html.Div('Carga Excel Control Partidos', style={'fontSize': '13px', 'fontWeight': '800', 'color': '#1E293B', 'marginBottom': '4px'}),
                                            html.Div('Excel (.xlsx)', style={'fontSize': '11px', 'color': '#64748B', 'fontWeight': '600'})
                                        ]
                                    ),
                                    multiple=False
                                ),
                                html.Div(id='matches-file-status', style={'marginTop': '10px', 'fontSize': '12px', 'fontWeight': '700'}),

html.Div(style={'height': '18px'}),
dbc.Alert(
    [
        html.Div('DISCLAIMER OFICIAL', style={'fontWeight': 900, 'letterSpacing': '0.08em'}),
        html.Div('Este sistema emite banderas de riesgo/seguimiento; no diagnostica lesiones.'),
        html.Div('Antes de decidir: revisar minutos, dolor/DOMS, retorno a juego y contexto de tarea.'),
    ],
    color='secondary',
    style={'borderRadius': '14px', 'fontWeight': 650},
),

dbc.Card(
    [
        dbc.CardHeader('LOG OPERATIVO', style={'fontWeight': 900, 'letterSpacing': '0.08em'}),
        dbc.CardBody(
            [
                html.Div('Registro interno de decisiones (sin BD).', style={'color': COLORS['slate_600'], 'fontWeight': 700, 'fontSize': '12px'}),
                html.Div(style={'height': '8px'}),
                dcc.Input(
                    id='ops-who',
                    type='text',
                    placeholder='Profesional (quien registra)...',
                    style={'width': '100%', 'borderRadius': '12px', 'padding': '10px', 'fontWeight': 650},
                ),
                html.Div(style={'height': '8px'}),
dcc.Textarea(
                    id='ops-note',
                    placeholder='Nota (qué se miró / qué se decide / qué contexto pedir)...',
                    style={'width': '100%', 'height': '80px', 'borderRadius': '12px', 'padding': '10px', 'fontWeight': 650},
                ),
                html.Div(style={'height': '10px'}),
                dbc.Row([
                    dbc.Col(dbc.Button('Registrar', id='ops-add', color='primary', className='w-100'), width=6),
                    dbc.Col(dbc.Button('Exportar', id='ops-export', color='secondary', outline=True, className='w-100'), width=6),
                ], className='g-2'),
                html.Div(style={'height': '10px'}),
                html.Div(id='ops-log-preview'),
            ]
        ),
    ],
    style={
        'border': '2px solid ' + COLORS['slate_200'],
        'borderRadius': '16px',
        'boxShadow': '0 6px 18px rgba(0,0,0,0.06)',
    },
),


html.Div(style={'height': '18px'}),
dbc.Card(
    [
        dbc.CardHeader('AUDITORÍA DE CARGA', style={'fontWeight': '900', 'letterSpacing': '0.08em'}),
        dbc.CardBody(html.Div(id='audit-panel')),
    ],
    style={
        'border': '2px solid ' + COLORS['slate_200'],
        'borderRadius': '16px',
        'boxShadow': '0 6px 18px rgba(0,0,0,0.06)',
    },
),


                                html.Div(style={'height': '24px'}),
                                html.Div(
                                    style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '16px'},
                                    children=[
                                        html.Div([
                                            html.Label('⚽ Posiciones', style={'fontSize': '10px', 'fontWeight': '800', 'color': '#64748B', 'display': 'block', 'marginBottom': '10px', 'textTransform': 'uppercase'}),
                                            dcc.Upload(id='pos-file', children=html.Div('Opcional', style={'border': '2px solid #CBD5E1', 'borderRadius': '10px', 'padding': '16px 12px', 'fontSize': '12px', 'textAlign': 'center', 'cursor': 'pointer', 'backgroundColor': '#F8FAFC', 'fontWeight': '700', 'color': '#475569'}), multiple=False)
                                        ])
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ], id='sidebar-col', className='mb-3', width={'xs': 12, 'lg': 3}),
                
                # ÁREA PRINCIPAL
            dbc.Col([
                html.Div(
                    id='main-area',
                    style={'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'},
                    children=[
                        # KPIs
                        html.Div(
                            style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '24px', 'marginBottom': '24px'},
                            children=[
                                html.Div(
                                    style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #FEE2E2 100%)', 'borderRadius': '20px', 'padding': '28px', 'border': '2px solid #FECACA', 'borderLeft': '6px solid #DC2626', 'boxShadow': '0 10px 40px rgba(220, 38, 38, 0.18)', 'position': 'relative'},
                                    children=[
                                        html.Div('📊', style={'position': 'absolute', 'top': '24px', 'right': '24px', 'fontSize': '56px', 'opacity': '0.12'}),
                                        html.Div('REGISTROS TOTALES', style={'fontSize': '11px', 'fontWeight': '900', 'color': '#7C2D12', 'textTransform': 'uppercase', 'marginBottom': '12px'}),
                                        html.Div('-', id='metric-rows', style={'fontSize': '42px', 'fontWeight': '900', 'color': '#DC2626'})
                                    ]
                                ),
                                html.Div(
                                    style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #DBEAFE 100%)', 'borderRadius': '20px', 'padding': '28px', 'border': '2px solid #BFDBFE', 'borderLeft': '6px solid #2563EB', 'boxShadow': '0 10px 40px rgba(37, 99, 235, 0.18)', 'position': 'relative'},
                                    children=[
                                        html.Div('📋', style={'position': 'absolute', 'top': '24px', 'right': '24px', 'fontSize': '56px', 'opacity': '0.12'}),
                                        html.Div('VARIABLES GPS', style={'fontSize': '11px', 'fontWeight': '900', 'color': '#1E3A8A', 'textTransform': 'uppercase', 'marginBottom': '12px'}),
                                        html.Div('-', id='metric-cols', style={'fontSize': '42px', 'fontWeight': '900', 'color': '#2563EB'})
                                    ]
                                ),
                                html.Div(
                                    style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #D1FAE5 100%)', 'borderRadius': '20px', 'padding': '28px', 'border': '2px solid #A7F3D0', 'borderLeft': '6px solid #059669', 'boxShadow': '0 10px 40px rgba(5, 150, 105, 0.18)', 'position': 'relative'},
                                    children=[
                                        html.Div('👥', style={'position': 'absolute', 'top': '24px', 'right': '24px', 'fontSize': '56px', 'opacity': '0.12'}),
                                        html.Div('ATLETAS MONITOREADOS', style={'fontSize': '11px', 'fontWeight': '900', 'color': '#064E3B', 'textTransform': 'uppercase', 'marginBottom': '12px'}),
                                        html.Div('-', id='metric-athletes', style={'fontSize': '42px', 'fontWeight': '900', 'color': '#059669'})
                                    ]
                                )
                            ]
                        ),
                        
                        # TABS
                        dcc.Tabs(
                            id='main-tabs',
                            value='tab-monitoreo',
                            style={'borderBottom': '3px solid #E2E8F0', 'marginBottom': '24px', 'backgroundColor': '#FFFFFF', 'borderRadius': '16px 16px 0 0', 'padding': '0 12px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'},
                            children=[
                                dcc.Tab(
                                    label='📈 MONITOREO DE CARGA',
                                    value='tab-monitoreo',
                                    style={'padding': '16px 32px', 'fontWeight': '700', 'fontSize': '13px', 'border': 'none', 'color': '#64748B', 'textTransform': 'uppercase'},
                                    selected_style={'padding': '16px 32px', 'fontWeight': '900', 'fontSize': '13px', 'border': 'none', 'borderBottom': '4px solid #DC2626', 'color': '#DC2626', 'backgroundColor': '#FEF2F2'}
                                ,
                                    children=[html.Div(id='tab-monitoreo-content')]
                                ),
                                dcc.Tab(
                                    label='⚠️ ACWR & CUADRANTES',
                                    value='tab-acwr',
                                    style={'padding': '16px 32px', 'fontWeight': '700', 'fontSize': '13px', 'border': 'none', 'color': '#64748B', 'textTransform': 'uppercase'},
                                    selected_style={'padding': '16px 32px', 'fontWeight': '900', 'fontSize': '13px', 'border': 'none', 'borderBottom': '4px solid #DC2626', 'color': '#DC2626', 'backgroundColor': '#FEF2F2'}
                                ,
                                    children=[html.Div(id='tab-acwr-content')]
                                ),
                                # NUEVO TAB: REPORTE SESION (TAB 3)
                                dcc.Tab(
                                    label='📝 REPORTE SESIÓN',
                                    value='tab-sesion',
                                    style={'padding': '16px 32px', 'fontWeight': '700', 'fontSize': '13px', 'border': 'none', 'color': '#64748B', 'textTransform': 'uppercase'},
                                    selected_style={'padding': '16px 32px', 'fontWeight': '900', 'fontSize': '13px', 'border': 'none', 'borderBottom': '4px solid #DC2626', 'color': '#DC2626', 'backgroundColor': '#FEF2F2'},
                                    children=[html.Div(id='tab-sesion-content')]
                                ),
                                dcc.Tab(
                                    label='🏟️ CONTROL PARTIDOS',
                                    value='tab-control-partidos',
                                    style={'padding': '16px 32px', 'fontWeight': '700', 'fontSize': '13px', 'border': 'none', 'color': '#64748B', 'textTransform': 'uppercase'},
                                    selected_style={'padding': '16px 32px', 'fontWeight': '900', 'fontSize': '13px', 'border': 'none', 'borderBottom': '4px solid #DC2626', 'color': '#DC2626', 'backgroundColor': '#FEF2F2'},
                                    children=[html.Div(id='tab-control-partidos-content')]
                                )
                                ,
                                dcc.Tab(
                                    label='📊 ZONAS DE CARGA',
                                    value='tab-zonas-carga',
                                    style={'padding': '16px 32px', 'fontWeight': '700', 'fontSize': '13px', 'border': 'none', 'color': '#64748B', 'textTransform': 'uppercase'},
                                    selected_style={'padding': '16px 32px', 'fontWeight': '900', 'fontSize': '13px', 'border': 'none', 'borderBottom': '4px solid #DC2626', 'color': '#DC2626', 'backgroundColor': '#FEF2F2'},
                                    children=[html.Div(id='tab-zonas-carga-content')]
                                )
                            ]
                        ),
                    ]
                )
            ], id='main-col', width={'xs': 12, 'lg': 9})
        ], className='g-3')
    ]
)

# ==========================================
# CALLBACKS
# ==========================================
# ==========================================
# EXPORTACIÓN PDF (PANTALLA): usa impresión del navegador
# ==========================================

app.clientside_callback(
    """
    function(clicks) {
        // clicks es un array de n_clicks de todos los botones type=btn-print
        if (!clicks) {
            return '';
        }
        const anyClicked = clicks.some(function(c) { return c && c > 0; });
        if (anyClicked) {
            // Pequeño delay para asegurar reflow antes de imprimir
            setTimeout(function() {
                window.print();
            }, 600);
        }
        return '';
    }
    """,
    Output('print-dummy', 'children'),
    Input({'type': 'btn-print', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(collapsed) {
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 250);
        return '';
    }
    """,
    Output('resize-dummy', 'children'),
    Input('sidebar-collapsed', 'data'),
    prevent_initial_call=True
)



# ==========================================
# CALLBACKS - AYUDA (MODALES)
# ==========================================

@app.callback(
    Output({'type': 'help-modal', 'tab': MATCH}, 'is_open'),
    Input({'type': 'help-open', 'tab': MATCH}, 'n_clicks'),
    Input({'type': 'help-close', 'tab': MATCH}, 'n_clicks'),
    State({'type': 'help-modal', 'tab': MATCH}, 'is_open'),
    prevent_initial_call=True
)
def toggle_help_modal(n_open, n_close, is_open):
    # Toggle simple: cualquier click abre/cierra.
    if callback_context.triggered:
        return not is_open
    return is_open

@app.callback(
    Output('sidebar-col', 'width'),
    Output('sidebar-col', 'style'),
    Output('main-col', 'width'),
    Output('main-col', 'style'),
    Output('sidebar-collapsed', 'data'),
    Input('toggle-sidebar', 'n_clicks'),
    State('sidebar-collapsed', 'data')
)
def toggle_sidebar(n_clicks, is_collapsed):
    """Toggle sidebar con comportamiento responsivo mejorado - Las gráficas ocupan todo el ancho al colapsar"""
    if n_clicks is None or n_clicks == 0:
        # Estado inicial: sidebar visible
        return {'xs': 12, 'lg': 3}, {'display': 'block', 'transition': 'all 0.4s'}, {'xs': 12, 'lg': 9}, {}, False

    new_collapsed = not is_collapsed

    if new_collapsed:
        # Sidebar escondido: main area ocupa todo el ancho
        return {'xs': 0, 'lg': 0}, {'display': 'none', 'transition': 'all 0.4s'}, {'xs': 12, 'lg': 12}, {'paddingLeft': '0px', 'paddingRight': '0px'}, True
    else:
        # Sidebar visible: distribución normal
        return {'xs': 12, 'lg': 3}, {'display': 'block', 'transition': 'all 0.4s'}, {'xs': 12, 'lg': 9}, {}, False

@app.callback(
    Output('stored-data', 'data'),
    Output('stored-positions', 'data'),
    Output('stored-risk-data', 'data'),
    Output('stored-matches-data', 'data'),
    Output('stored-audit', 'data'),
    Output('main-file-status', 'children'),
    Output('matches-file-status', 'children'),
    Input('main-file', 'contents'),
    State('main-file', 'filename'),
    Input('pos-file', 'contents'),
    State('pos-file', 'filename'),
    Input('matches-file', 'contents'),
    State('matches-file', 'filename'),
)
def store_uploaded_data(main_contents, main_filename, pos_contents, pos_filename, matches_contents, matches_filename):
    # Outputs (7): stored-data, stored-positions, stored-risk-data, stored-matches-data, stored-audit, main-file-status, matches-file-status
    if main_contents is None:
        return None, None, None, None, None, '', ''

    # 1) Carga principal
    df = load_and_prepare_data(main_contents, main_filename)

    ok_main, warn_main, block_main = validate_main_dataframe(df)
    if not ok_main:
        audit = build_audit_payload(
            main_contents,
            main_filename,
            df,
            file_role='principal',
            warnings=[block_main],
        )
        statusmsg = html.Div(
            block_main,
            style={'color': COLORS['red'], 'fontWeight': 800, 'fontSize': '12px'}
        )
        return None, None, None, None, audit, statusmsg, ''

    if df.empty:
        # Importante: mantener cantidad de outputs. No ejecuta análisis.
        block = f"❌ Error: {main_filename}"
        audit = build_audit_payload(
            main_contents,
            main_filename,
            df,
            file_role='principal',
            warnings=[block],
        )
        statusmsg = html.Div(
            block,
            style={'color': COLORS['red'], 'fontWeight': '800', 'fontSize': '12px'}
        )
        return None, None, None, None, audit, statusmsg, ''

    # 2) Posiciones (opcional) — solo validación + carga
    warn_pos = []
    df_pos = None
    if pos_contents is not None and pos_filename is not None:
        df_pos_temp = load_and_prepare_data(pos_contents, pos_filename)
        ok_pos, warn_pos, block_pos = validate_positions_dataframe(df_pos_temp)
        if ok_pos and (not df_pos_temp.empty):
            df_pos = df_pos_temp.to_dict('records')
        else:
            # No bloquea el análisis principal; solo warning y no mergea.
            if block_pos:
                warn_pos = (warn_pos or []) + [block_pos]
            df_pos = None

    # 3) Control Partidos (opcional) — solo validación + carga
    warn_matches = []
    matches_df = None
    matches_status = ''
    if matches_contents is not None and matches_filename is not None:
        df_matches_temp = load_and_prepare_data(matches_contents, matches_filename)
        ok_m, warn_matches, block_m = validate_matches_dataframe(df_matches_temp)
        if (not ok_m) or df_matches_temp.empty:
            matches_df = None
            if block_m:
                warn_matches = (warn_matches or []) + [block_m]
            matches_status = html.Div(
                f"❌ Error: {matches_filename}",
                style={'color': COLORS['red'], 'fontWeight': '800', 'fontSize': '12px'}
            )
        else:
            matches_df = df_matches_temp.to_dict('records')
            matches_status = html.Div(
                f"✅ {matches_filename} • {fmt_int(len(df_matches_temp))} registros",
                style={'color': COLORS['green'], 'fontWeight': '800', 'fontSize': '12px'}
            )

    # 4) Risk dataset (análisis existente) — intacto
    risk_df = None
    try:
        risk_df_calc = build_feature_dataset(df, 'Distancia_Total')
        risk_df = risk_df_calc.to_dict('records')
    except Exception as e:
        logger.exception(f"Error calculando risk_score y ACWR: {e}")

    # 5) Status principal
    status_msg = html.Div(
        f"✅ {main_filename} • {fmt_int(len(df))} registros",
        style={'color': COLORS['green'], 'fontWeight': '800', 'fontSize': '12px'}
    )

    # 6) Auditoría: huella + metadata + warnings (sin tocar cálculos)
    warnings_all = []
    warnings_all.extend(warn_main if isinstance(warn_main, list) else [])
    warnings_all.extend(warn_pos if isinstance(warn_pos, list) else [])
    warnings_all.extend(warn_matches if isinstance(warn_matches, list) else [])

    audit = build_audit_payload(
        main_contents,
        main_filename,
        df,
        file_role='principal',
        warnings=warnings_all,
    )

    return df.to_dict('records'), df_pos, risk_df, matches_df, audit, status_msg, matches_status

@app.callback(
    Output("metric-rows", "children"),
    Output("metric-cols", "children"),
    Output("metric-athletes", "children"),
    Output("tab-monitoreo-content", "children"),
    Output("tab-acwr-content", "children"),
    Output("tab-sesion-content", "children"),
    Output("tab-control-partidos-content", "children"),
    Output("tab-zonas-carga-content", "children"),
    Input("stored-data", "data"),
    Input("stored-positions", "data"),
    Input("stored-risk-data", "data"),
    Input("stored-matches-data", "data"),
)
def update_dashboard(stored_data, stored_positions, stored_risk_data, stored_matches_data):
    if stored_data is None:
        welcome = html.Div(
            style={'textAlign': 'center', 'padding': '120px 40px'},
            children=[
                html.Div('🎯', style={'fontSize': '120px', 'marginBottom': '40px'}),
                html.H2('SISTEMA SPORT SCIENCE CAI', style={'fontSize': '38px', 'fontWeight': '900', 'color': '#0F172A'}),
                html.P('Carga un archivo (Excel/CSV) para iniciar el análisis', style={'fontSize': '18px', 'color': '#64748B'})
            ]
        )
        return "-", "-", "-", welcome, welcome, welcome, welcome, welcome

    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])

    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
        if 'Posición_Pos' in df.columns:
            df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
            df = df.drop(columns=['Posición_Pos'])

    n_rows = fmt_int(len(df))
    n_cols = fmt_int(df.shape[1])
    n_athletes = fmt_int(df['Atleta'].nunique()) if 'Atleta' in df.columns else "-"

    content_monitoreo = render_tab_monitoreo(df)
    content_acwr = render_tab_acwr(stored_data, stored_positions)
    content_sesion = render_tab_sesion(stored_data, stored_positions)
    content_control_partidos = render_tab_control_partidos(stored_matches_data)
    content_zonas_carga = render_tab_zonas_carga(stored_data, stored_positions)

    return n_rows, n_cols, n_athletes, content_monitoreo, content_acwr, content_sesion, content_control_partidos, content_zonas_carga

# ==========================================
# RENDER TABS
# ==========================================

def render_tab_monitoreo(df):
    """TAB 1: MONITOREO DE CARGA CON BOX PLOT"""
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    categorical_vars = ['Atleta', 'Microciclo', 'Tipo', 'MD', 'Posición']
    available_categorical = [c for c in categorical_vars if c in df.columns]
    
    if not numeric_vars or 'Fecha' not in df.columns:
        return html.Div('⚠️ Datos insuficientes', style={'padding': '80px', 'textAlign': 'center'})
    
    min_date = df['Fecha'].min().date()
    max_date = df['Fecha'].max().date()
    atletas_unicos = sorted(df['Atleta'].unique()) if 'Atleta' in df.columns else []
    atletas_options = [{"label": "🏆 Todos los atletas", "value": "TODOS"}] + [{"label": f"👤 {a}", "value": a} for a in atletas_unicos]
    
    help_ctl = build_help_modal('monitoreo', title='Ayuda - Monitoreo')
    return html.Div([
        html.Div(style={'display':'flex','justifyContent':'flex-end','marginBottom':'12px'}, children=[help_ctl]),
        # GRÁFICO 1: EVOLUCIÓN TEMPORAL
        html.Div(
            style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #E2E8F0', 'marginBottom': '24px', 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.08)'},
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '28px'},
                    children=[
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'gap': '14px'},
                            children=[
                                html.Div('📈', style={'fontSize': '28px'}),
                                html.H3('EVOLUCIÓN TEMPORAL', style={'fontSize': '18px', 'fontWeight': '900', 'color': '#0F172A', 'margin': '0'})
                            ]
                        ),
                        html.Button(
                            '📥 EXPORTAR PDF (PANTALLA)',
                            id={'type': 'btn-print', 'index': 'monitoreo'},
                            n_clicks=0,
                            className='no-print',
                            style={
                                'backgroundColor': '#DC2626',
                                'color': 'white',
                                'padding': '10px 20px',
                                'borderRadius': '10px',
                                'textDecoration': 'none',
                                'fontSize': '12px',
                                'fontWeight': '800',
                                'cursor': 'pointer',
                                'border': 'none',
                                'boxShadow': '0 2px 8px rgba(220, 38, 38, 0.3)'
                            }
                        )
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '2fr 2fr 2fr 2fr', 'gap': '28px', 'alignItems': 'end', 'marginBottom': '24px'},
                    children=[
                        html.Div([
                            html.Label('👤 ATLETA', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-atleta', 'index': 'timeseries'}, options=atletas_options, value="TODOS", clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📅 PERÍODO', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.DatePickerRange(id={'type': 'filter-fechas', 'index': 'timeseries'}, min_date_allowed=min_date, max_date_allowed=max_date, start_date=min_date, end_date=max_date, display_format="DD/MM/YY")
                        ]),
                        html.Div([
                            html.Label('📊 MÉTRICA', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-metrica', 'index': 'timeseries'}, options=[{"label": c, "value": c} for c in numeric_vars], value=numeric_vars[0], clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Button('🔄 ACTUALIZAR', id={'type': 'btn-update', 'index': 'timeseries'}, n_clicks=0, style={'backgroundColor': '#DC2626', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'padding': '12px 24px', 'fontSize': '13px', 'fontWeight': '800', 'cursor': 'pointer', 'marginTop': '26px'})
                        ])
                    ]
                ),
                html.Div(id={'type': 'graph-output', 'index': 'timeseries'})
            ]
        ),
        
        # GRÁFICO 2: BOX PLOT
        html.Div(
            style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #E2E8F0', 'marginBottom': '24px', 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.08)'},
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '28px'},
                    children=[
                        html.Div('📦', style={'fontSize': '28px'}),
                        html.H3('DISTRIBUCIÓN POR CATEGORÍA', style={'fontSize': '18px', 'fontWeight': '900', 'color': '#0F172A', 'margin': '0'}),
                        html.Div('Box Plot con análisis de cuartiles y outliers', style={'fontSize': '12px', 'color': '#64748B', 'fontWeight': '600', 'marginLeft': '16px', 'fontStyle': 'italic'})
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '2fr 2fr 2fr 2fr', 'gap': '28px', 'alignItems': 'end', 'marginBottom': '24px'},
                    children=[
                        html.Div([
                            html.Label('👤 ATLETA', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-atleta', 'index': 'boxplot'}, options=atletas_options, value="TODOS", clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📊 MÉTRICA A ANALIZAR', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-metrica-box', 'index': 'boxplot'}, options=[{"label": c, "value": c} for c in numeric_vars], value=numeric_vars[0], clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('🏷️ AGRUPAR POR', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-categoria-box', 'index': 'boxplot'}, options=[{"label": c, "value": c} for c in available_categorical], value='MD' if 'MD' in available_categorical else (available_categorical[0] if available_categorical else None), clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Button('🔄 ACTUALIZAR', id={'type': 'btn-update', 'index': 'boxplot'}, n_clicks=0, style={'backgroundColor': '#2563EB', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'padding': '12px 24px', 'fontSize': '13px', 'fontWeight': '800', 'cursor': 'pointer', 'marginTop': '26px'})
                        ])
                    ]
                ),
                html.Div(id={'type': 'graph-output', 'index': 'boxplot'})
            ]
        )
    ])

def render_tab_acwr(stored_data, stored_positions):
    """TAB 2: ACWR & CUADRANTES CON ANÁLISIS HISTÓRICO POR MD"""
    if stored_data is None:
        return html.Div('⚠️ Sin datos')
    
    df_base = pd.DataFrame(stored_data)
    df_base['Fecha'] = pd.to_datetime(df_base['Fecha'])
    
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        df_base = df_base.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
        if 'Posición_Pos' in df_base.columns:
            df_base['Posición'] = df_base['Posición_Pos'].fillna(df_base.get('Posición', ''))
            df_base = df_base.drop(columns=['Posición_Pos'])
    
    numeric_columns = df_base.select_dtypes(include=np.number).columns.tolist()
    min_date = df_base['Fecha'].min().date()
    max_date = df_base['Fecha'].max().date()
    atletas = sorted(df_base['Atleta'].dropna().unique().tolist())
    atletas_options = [{"label": "🏆 Todos los atletas", "value": "TODOS"}] + [{"label": f"👤 {a}", "value": a} for a in atletas]
    
    fechas_disponibles = sorted(df_base['Fecha'].dt.date.unique(), reverse=True)
    fechas_options = [{"label": f"📅 {f.strftime('%d/%m/%Y')}", "value": f.strftime('%Y-%m-%d')} for f in fechas_disponibles[:100]]
    
    help_ctl = build_help_modal('acwr', title='Ayuda - ACWR & Cuadrantes')
    return html.Div([
        html.Div(style={'display':'flex','justifyContent':'flex-end','marginBottom':'12px'}, children=[help_ctl]),
        # HEADER EXPLICATIVO
        html.Div(
            style={'background': 'linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%)', 'borderRadius': '20px', 'padding': '40px', 'border': '3px solid #FCA5A5', 'marginBottom': '28px', 'boxShadow': '0 10px 40px rgba(220, 38, 38, 0.18)'},
            children=[
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'},
                    children=[
                        html.H2('⚠️ ACWR & ANÁLISIS DE CUADRANTES', style={'fontSize': '28px', 'fontWeight': '900', 'color': '#DC2626', 'margin': '0'}),
                        html.Button(
                            '📥 EXPORTAR PDF (PANTALLA)',
                            id={'type': 'btn-print', 'index': 'acwr'},
                            n_clicks=0,
                            className='no-print',
                            style={
                                'backgroundColor': '#DC2626',
                                'color': 'white',
                                'padding': '10px 20px',
                                'borderRadius': '10px',
                                'textDecoration': 'none',
                                'fontSize': '12px',
                                'fontWeight': '800',
                                'cursor': 'pointer',
                                'boxShadow': '0 2px 8px rgba(220, 38, 38, 0.3)'
                            }
                        )
                    ]
                ),
                html.P('Sistema integrado de análisis de riesgo lesivo mediante ACWR y posicionamiento en cuadrantes Z-score. Incluye análisis histórico contextual por tipo de sesión (MD).', style={'fontSize': '15px', 'color': '#475569', 'lineHeight': '1.8', 'fontWeight': '600'})
            ]
        ),
        
        # GRÁFICO 1: CUADRANTE GENERAL
        html.Div(
            style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #E2E8F0', 'marginBottom': '28px', 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.08)'},
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '28px'},
                    children=[
                        html.Div('📊', style={'fontSize': '28px'}),
                        html.H3('CUADRANTES GENERALES', style={'fontSize': '18px', 'fontWeight': '900', 'color': '#0F172A', 'margin': '0'})
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr 1fr 2fr 2fr', 'gap': '24px', 'marginBottom': '24px'},
                    children=[
                        html.Div([
                            html.Label('👤 ATLETA', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-atleta', 'index': 'cuadrante-gral'}, options=atletas_options, value="TODOS", clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📅 INICIO', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.DatePickerSingle(id={'type': 'filter-start', 'index': 'cuadrante-gral'}, date=min_date, display_format="DD/MM/YY", style={'fontSize': '13px'})
                        ]),
                        html.Div([
                            html.Label('📅 FIN', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.DatePickerSingle(id={'type': 'filter-end', 'index': 'cuadrante-gral'}, date=max_date, display_format="DD/MM/YY", style={'fontSize': '13px'})
                        ]),
                        html.Div([
                            html.Label('📊 VARIABLE X', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-var-x', 'index': 'cuadrante-gral'}, options=[{"label": c, "value": c} for c in numeric_columns], value=numeric_columns[0] if numeric_columns else None, clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📊 VARIABLE Y', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-var-y', 'index': 'cuadrante-gral'}, options=[{"label": c, "value": c} for c in numeric_columns], value=numeric_columns[1] if len(numeric_columns) > 1 else None, clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ])
                    ]
                ),
                html.Div([
                    html.Button('🔄 ACTUALIZAR', id={'type': 'btn-update', 'index': 'cuadrante-gral'}, n_clicks=0, style={'backgroundColor': '#059669', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'padding': '12px 24px', 'fontSize': '13px', 'fontWeight': '800', 'cursor': 'pointer'})
                ]),
                html.Div(id={'type': 'graph-output', 'index': 'cuadrante-gral'}, style={'marginTop': '24px'})
            ]
        ),
        
        # GRÁFICO 2: ANÁLISIS HISTÓRICO POR MD
        html.Div(
            style={'background': 'linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '3px solid #FDE68A', 'marginBottom': '28px', 'boxShadow': '0 10px 40px rgba(245, 158, 11, 0.18)'},
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '28px'},
                    children=[
                        html.Div('🎯', style={'fontSize': '28px'}),
                        html.H3('ANÁLISIS HISTÓRICO CONTEXTUAL POR MD', style={'fontSize': '18px', 'fontWeight': '900', 'color': '#92400E', 'margin': '0'}),
                        html.Div('Compara sesión específica vs historial del mismo tipo de entrenamiento', style={'fontSize': '12px', 'color': '#78350F', 'fontWeight': '600', 'marginLeft': '16px', 'fontStyle': 'italic'})
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '2fr 2fr 2fr 2fr', 'gap': '24px', 'marginBottom': '24px'},
                    children=[
                        html.Div([
                            html.Label('👤 ATLETA', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#78350F', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-atleta', 'index': 'historico-md'}, options=atletas_options, value="TODOS", clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📅 FECHA A ANALIZAR', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#78350F', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-fecha-objetivo', 'index': 'historico-md'}, options=fechas_options, value=fechas_options[0]['value'] if fechas_options else None, clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📊 VARIABLE X', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#78350F', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-var-x', 'index': 'historico-md'}, options=[{"label": c, "value": c} for c in numeric_columns], value=numeric_columns[0] if numeric_columns else None, clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ]),
                        html.Div([
                            html.Label('📊 VARIABLE Y', style={'display': 'block', 'fontSize': '11px', 'fontWeight': '900', 'color': '#78350F', 'marginBottom': '12px'}),
                            dcc.Dropdown(id={'type': 'filter-var-y', 'index': 'historico-md'}, options=[{"label": c, "value": c} for c in numeric_columns], value=numeric_columns[1] if len(numeric_columns) > 1 else None, clearable=False, style={'fontSize': '14px', 'fontWeight': '700'})
                        ])
                    ]
                ),
                html.Div([
                    html.Button('🎯 ANALIZAR HISTORIAL', id={'type': 'btn-update', 'index': 'historico-md'}, n_clicks=0, style={'backgroundColor': '#D97706', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'padding': '12px 32px', 'fontSize': '14px', 'fontWeight': '900', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(217, 119, 6, 0.4)'})
                ]),
                html.Div(id={'type': 'graph-output', 'index': 'historico-md'}, style={'marginTop': '24px'})
            ]
        )
    ])

def render_tab_sesion(stored_data, stored_positions):
    """TAB 3: REPORTE SESION (NUEVO)"""
    if stored_data is None:
        return html.Div('⚠️ Sin datos cargados')

    df_base = pd.DataFrame(stored_data)
    df_base['Fecha'] = pd.to_datetime(df_base['Fecha'])

    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        df_base = df_base.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
        if 'Posición_Pos' in df_base.columns:
            df_base['Posición'] = df_base['Posición_Pos'].fillna(df_base.get('Posición', ''))
            df_base = df_base.drop(columns=['Posición_Pos'])

    numeric_columns = df_base.select_dtypes(include=np.number).columns.tolist()
    # Ordenar fechas descendente para facilitar selección
    fechas_disponibles = sorted(df_base['Fecha'].dt.date.unique(), reverse=True)
    max_date = fechas_disponibles[0] if fechas_disponibles else datetime.now().date()
    
    # Valores por defecto para dropdowns
    def_var1 = numeric_columns[0] if len(numeric_columns) > 0 else None
    def_var2 = numeric_columns[1] if len(numeric_columns) > 1 else def_var1
    def_var3 = numeric_columns[2] if len(numeric_columns) > 2 else def_var1
    def_var4 = numeric_columns[3] if len(numeric_columns) > 3 else def_var1

    help_ctl = build_help_modal('sesion', title='Ayuda - Reporte de sesión')
    return html.Div([
        html.Div(style={'display':'flex','justifyContent':'flex-end','marginBottom':'12px'}, children=[help_ctl]),
        # HEADER Y SELECTORES
        html.Div(
            style={'background': '#E2E8F0', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #CBD5E1', 'marginBottom': '28px'},
            children=[
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '24px'},
                    children=[
                        html.Div([
                            html.H2('REPORTE SESIÓN', style={'fontSize': '28px', 'fontWeight': '900', 'color': '#0F172A', 'margin': '0'}),
                            html.P('Análisis detallado de sesión con comparativa vs promedio', style={'margin': '4px 0 0 0', 'color': '#64748B'})
                        ]),
                        html.Div(
                            style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-end'},
                            children=[
                                html.Div(
                                    style={'backgroundColor': 'white', 'padding': '14px 20px', 'borderRadius': '12px', 'border': '2px solid #94A3B8', 'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'},
                                    children=[
                                        html.Label('📅 FECHA DE SESIÓN', style={'fontSize': '11px', 'fontWeight': '900', 'color': '#475569', 'display': 'block', 'marginBottom': '8px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                                        dcc.DatePickerSingle(
                                            id={'type': 'sesion-date', 'index': 'sesion'},
                                            date=max_date,
                                            display_format='DD/MM/YYYY',
                                            style={'border': 'none', 'fontSize': '15px', 'fontWeight': '700', 'color': '#0F172A'}
                                        )
                                    ]
                                ),
                                html.Button(
                                    '📥 EXPORTAR PDF (PANTALLA)',
                                    id={'type': 'btn-print', 'index': 'sesion'},
                                    n_clicks=0,
                                    className='no-print',
                                    style={
                                        'backgroundColor': '#2563EB',
                                        'color': 'white',
                                        'padding': '12px 24px',
                                        'borderRadius': '10px',
                                        'textDecoration': 'none',
                                        'fontSize': '12px',
                                        'fontWeight': '800',
                                        'cursor': 'pointer',
                                        'boxShadow': '0 2px 8px rgba(37, 99, 235, 0.3)',
                                        'display': 'inline-block'
                                    }
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px'},
                    children=[
                        dcc.Dropdown(id={'type': 'var-1', 'index': 'sesion'}, options=[{'label': c, 'value': c} for c in numeric_columns], value=def_var1, clearable=False, style={'fontSize': '13px'}),
                        dcc.Dropdown(id={'type': 'var-2', 'index': 'sesion'}, options=[{'label': c, 'value': c} for c in numeric_columns], value=def_var2, clearable=False, style={'fontSize': '13px'}),
                        dcc.Dropdown(id={'type': 'var-3', 'index': 'sesion'}, options=[{'label': c, 'value': c} for c in numeric_columns], value=def_var3, clearable=False, style={'fontSize': '13px'}),
                        dcc.Dropdown(id={'type': 'var-4', 'index': 'sesion'}, options=[{'label': c, 'value': c} for c in numeric_columns], value=def_var4, clearable=False, style={'fontSize': '13px'}),
                    ]
                ),

            ]
        ),
        # CONTENEDOR DE GRÁFICOS Y TABLA
        html.Div(id={'type': 'graph-output', 'index': 'sesion'})
    ])


def render_tab_control_partidos(stored_matches_data):
    """TAB 4: CONTROL PARTIDOS"""
    if stored_matches_data is None:
        return html.Div('⚠️ Carga el Excel de Control Partidos en el sidebar.', style={'padding': '40px', 'textAlign': 'center'})

    dfm = pd.DataFrame(stored_matches_data)
    n_rows = fmt_int(len(dfm))
    n_cols = fmt_int(dfm.shape[1])
    preview_cols = dfm.columns[:8].tolist()

    table = html.Div(
        style={'overflowX': 'auto', 'marginTop': '18px'},
        children=[
            html.Table(
                style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'},
                children=[
                    html.Thead(html.Tr([
                        html.Th(c, style={'padding': '10px', 'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': '800', 'textAlign': 'left'})
                        for c in preview_cols
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(str(dfm.iloc[i][c]), style={'padding': '8px', 'borderBottom': '1px solid #E2E8F0', 'fontWeight': '600'})
                            for c in preview_cols
                        ], style={'backgroundColor': '#F8FAFC' if i % 2 == 0 else '#FFFFFF'})
                        for i in range(min(20, len(dfm)))
                    ])
                ]
            )
        ]
    )

    help_ctl = build_help_modal('control', title='Ayuda - Control partidos')
    return html.Div(
        style={'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)', 'borderRadius': '20px', 'padding': '32px', 'border': '2px solid #E2E8F0'},
        children=[
        html.Div(style={'display':'flex','justifyContent':'flex-end','marginBottom':'12px'}, children=[help_ctl]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '10px'}, children=[
                html.Div('🏟️', style={'fontSize': '28px'}),
                html.H3('CONTROL PARTIDOS', style={'fontSize': '18px', 'fontWeight': '900', 'color': '#0F172A', 'margin': '0'})
            ]),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px', 'marginTop': '18px'}, children=[
                html.Div(style={'background': '#EEF2FF', 'borderRadius': '14px', 'padding': '16px', 'border': '1px solid #C7D2FE'}, children=[
                    html.Div('REGISTROS', style={'fontSize': '10px', 'fontWeight': '900', 'color': '#3730A3'}),
                    html.Div(n_rows, style={'fontSize': '26px', 'fontWeight': '900', 'color': '#1D4ED8'})
                ]),
                html.Div(style={'background': '#ECFDF5', 'borderRadius': '14px', 'padding': '16px', 'border': '1px solid #A7F3D0'}, children=[
                    html.Div('COLUMNAS', style={'fontSize': '10px', 'fontWeight': '900', 'color': '#064E3B'}),
                    html.Div(n_cols, style={'fontSize': '26px', 'fontWeight': '900', 'color': '#059669'})
                ]),
                html.Div(style={'background': '#FEF3C7', 'borderRadius': '14px', 'padding': '16px', 'border': '1px solid #FDE68A'}, children=[
                    html.Div('VISTA PREVIA', style={'fontSize': '10px', 'fontWeight': '900', 'color': '#78350F'}),
                    html.Div('Primeras 20 filas', style={'fontSize': '14px', 'fontWeight': '800', 'color': '#92400E'})
                ])
            ]),
            table
        ]
    )

# ==========================================
# GRAPH CALLBACKS (EXISTENTES Y NUEVO)
# ==========================================



def render_tab_zonas_carga(stored_data, stored_positions):
    """TAB 5: ZONAS DE CARGA CON GAUGES - DISEÑO PREMIUM"""
    if stored_data is None:
        return html.Div(
            style={'padding': '80px', 'textAlign': 'center'},
            children=[
                html.Div('📊', style={'fontSize': '64px', 'marginBottom': '20px', 'opacity': '0.5'}),
                html.Div('Sin datos cargados', 
                        style={'fontSize': '20px', 'fontWeight': '700', 'color': '#64748B'})
            ]
        )

    df = pd.DataFrame(stored_data)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
        if 'Posición_Pos' in df.columns:
            df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
            df = df.drop(columns=['Posición_Pos'])

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    fechas_disponibles = sorted(df['Fecha'].dt.date.unique(), reverse=True)
    max_date = fechas_disponibles[0] if fechas_disponibles else datetime.now().date()

    help_ctl = build_help_modal('zonas', title='Ayuda - Zonas de carga')
    return html.Div([
        html.Div(style={'display':'flex','justifyContent':'flex-end','marginBottom':'12px'}, children=[help_ctl]),
        # HEADER PREMIUM CON GRADIENTE
        html.Div(
            style={
                'background': 'linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%)',
                'borderRadius': '24px',
                'padding': '40px',
                'marginBottom': '32px',
                'boxShadow': '0 20px 60px rgba(0, 0, 0, 0.3)',
                'border': '1px solid rgba(255, 255, 255, 0.1)',
                'position': 'relative',
                'overflow': 'hidden'
            },
            children=[
                # Decoración de fondo
                html.Div(
                    style={
                        'position': 'absolute',
                        'top': '-50%',
                        'right': '-10%',
                        'width': '600px',
                        'height': '600px',
                        'background': 'radial-gradient(circle, rgba(220, 38, 38, 0.15) 0%, transparent 70%)',
                        'borderRadius': '50%',
                        'pointerEvents': 'none'
                    }
                ),
                # Contenido principal
                html.Div(
                    style={'position': 'relative', 'zIndex': '1'},
                    children=[
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'space-between', 'flexWrap': 'wrap', 'gap': '32px'},
                            children=[
                                # Título y descripción
                                html.Div(
                                    style={'flex': '1', 'minWidth': '300px'},
                                    children=[
                                        html.Div(
                                            style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'marginBottom': '16px'},
                                            children=[
                                                html.Div(
                                                    '📊',
                                                    style={
                                                        'fontSize': '48px',
                                                        'background': 'linear-gradient(135deg, #DC2626, #F59E0B)',
                                                        'borderRadius': '20px',
                                                        'padding': '12px',
                                                        'lineHeight': '1'
                                                    }
                                                ),
                                                html.Div([
                                                    html.H2(
                                                        'ZONAS DE CARGA',
                                                        style={
                                                            'fontSize': '32px',
                                                            'fontWeight': '900',
                                                            'color': 'white',
                                                            'margin': '0',
                                                            'letterSpacing': '-0.5px',
                                                            'lineHeight': '1.2'
                                                        }
                                                    ),
                                                    html.Div(
                                                        'Análisis de carga por equipo con referencias de competición',
                                                        style={
                                                            'fontSize': '14px',
                                                            'color': 'rgba(255, 255, 255, 0.7)',
                                                            'marginTop': '4px',
                                                            'fontWeight': '500'
                                                        }
                                                    )
                                                ])
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),
                        # FILTROS EN GRID 50/50
                        html.Div(
                            style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px', 'marginTop': '24px'},
                            children=[
                                # Selector de fecha (50%)
                                html.Div([
                                    html.Label(
                                        style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px'},
                                        children=[
                                            html.Div('📅', style={'fontSize': '20px'}),
                                            html.Div(
                                                'FECHA DE SESIÓN',
                                                style={
                                                    'fontSize': '12px',
                                                    'fontWeight': '900',
                                                    'color': 'rgba(255, 255, 255, 0.9)',
                                                    'letterSpacing': '1px'
                                                }
                                            )
                                        ]
                                    ),
                                    dcc.DatePickerSingle(
                                        id={'type': 'zonas-date', 'index': 'zonas'},
                                        date=max_date,
                                        display_format='DD/MM/YYYY',
                                        with_portal=True,
                                        show_outside_days=True,
                                        number_of_months_shown=2,
                                        first_day_of_week=1,
                                        style={'width': '100%'}
                                    )
                                ]),
                                # Selector de variables (50%)
                                html.Div([
                                    html.Label(
                                        style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px'},
                                        children=[
                                            html.Div('📊', style={'fontSize': '20px'}),
                                            html.Div(
                                                'VARIABLES DE RENDIMIENTO',
                                                style={
                                                    'fontSize': '12px',
                                                    'fontWeight': '900',
                                                    'color': 'rgba(255, 255, 255, 0.9)',
                                                    'letterSpacing': '1px'
                                                }
                                            )
                                        ]
                                    ),
                                    dcc.Dropdown(
                                        id={'type': 'zonas-vars', 'index': 'zonas'},
                                        options=[{'label': v, 'value': v} for v in numeric_columns],
                                        value=numeric_columns[:6] if len(numeric_columns) >= 6 else numeric_columns,
                                        multi=True,
                                        placeholder='Selecciona todas las variables que desees...',
                                        optionHeight=45,
                                        maxHeight=350,
                                        searchable=True,
                                        clearable=True,
                                        style={
                                            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                                            'borderRadius': '12px',
                                            'fontSize': '14px',
                                            'fontWeight': '600'
                                        }
                                    )
                                ])
                            ]
                        )
                    ]
                )
            ]
        ),
        # LEYENDA DE ZONAS COMPACTA
        html.Div(
            style={
                'background': 'linear-gradient(135deg, #FEF3C7 0%, #FEF9C3 100%)',
                'padding': '16px',
                'borderRadius': '12px',
                'border': '2px solid #F59E0B',
                'marginBottom': '24px',
                'boxShadow': '0 4px 12px rgba(245, 158, 11, 0.15)'
            },
            children=[
                html.Div(
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '12px'},
                    children=[
                        html.Div('🎯', style={'fontSize': '18px'}),
                        html.Div(
                            'CLASIFICACIÓN DE ZONAS DE CARGA',
                            style={
                                'fontSize': '11px',
                                'fontWeight': '900',
                                'color': '#78350F',
                                'letterSpacing': '0.5px'
                            }
                        )
                    ]
                ),
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))',
                        'gap': '10px'
                    },
                    children=[
                        # DESARROLLO
                        html.Div(
                            style={
                                'background': 'linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%)',
                                'padding': '10px',
                                'borderRadius': '10px',
                                'border': '2px solid #DC2626',
                                'boxShadow': '0 2px 6px rgba(220, 38, 38, 0.2)'
                            },
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '6px'},
                                    children=[
                                        html.Div('🔴', style={'fontSize': '14px'}),
                                        html.Div(
                                            'DESARROLLO',
                                            style={
                                                'fontSize': '10px',
                                                'fontWeight': '900',
                                                'color': '#7C2D12',
                                                'letterSpacing': '0.3px'
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    '> 60%',
                                    style={
                                        'fontSize': '11px',
                                        'color': '#991B1B',
                                        'fontWeight': '700',
                                        'marginBottom': '4px'
                                    }
                                ),
                                html.Div(
                                    'Carga alta vs competición',
                                    style={
                                        'fontSize': '9px',
                                        'color': '#7C2D12',
                                        'lineHeight': '1.3',
                                        'fontWeight': '600'
                                    }
                                )
                            ]
                        ),
                        # MANTENIMIENTO
                        html.Div(
                            style={
                                'background': 'linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)',
                                'padding': '10px',
                                'borderRadius': '10px',
                                'border': '2px solid #D97706',
                                'boxShadow': '0 2px 6px rgba(217, 119, 6, 0.2)'
                            },
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '6px'},
                                    children=[
                                        html.Div('🟡', style={'fontSize': '14px'}),
                                        html.Div(
                                            'MANTENIMIENTO',
                                            style={
                                                'fontSize': '10px',
                                                'fontWeight': '900',
                                                'color': '#78350F',
                                                'letterSpacing': '0.3px'
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    '40-60%',
                                    style={
                                        'fontSize': '11px',
                                        'color': '#92400E',
                                        'fontWeight': '700',
                                        'marginBottom': '4px'
                                    }
                                ),
                                html.Div(
                                    'Zona óptima habitual',
                                    style={
                                        'fontSize': '9px',
                                        'color': '#78350F',
                                        'lineHeight': '1.3',
                                        'fontWeight': '600'
                                    }
                                )
                            ]
                        ),
                        # ACTIVACIÓN
                        html.Div(
                            style={
                                'background': 'linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%)',
                                'padding': '10px',
                                'borderRadius': '10px',
                                'border': '2px solid #059669',
                                'boxShadow': '0 2px 6px rgba(5, 150, 105, 0.2)'
                            },
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '6px'},
                                    children=[
                                        html.Div('🟢', style={'fontSize': '14px'}),
                                        html.Div(
                                            'ACTIVACIÓN',
                                            style={
                                                'fontSize': '10px',
                                                'fontWeight': '900',
                                                'color': '#064E3B',
                                                'letterSpacing': '0.3px'
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    '30-40%',
                                    style={
                                        'fontSize': '11px',
                                        'color': '#065F46',
                                        'fontWeight': '700',
                                        'marginBottom': '4px'
                                    }
                                ),
                                html.Div(
                                    'Ideal MD-1',
                                    style={
                                        'fontSize': '9px',
                                        'color': '#064E3B',
                                        'lineHeight': '1.3',
                                        'fontWeight': '600'
                                    }
                                )
                            ]
                        ),
                        # RECUPERACIÓN
                        html.Div(
                            style={
                                'background': 'linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%)',
                                'padding': '10px',
                                'borderRadius': '10px',
                                'border': '2px solid #93C5FD',
                                'boxShadow': '0 2px 6px rgba(147, 197, 253, 0.2)'
                            },
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '6px'},
                                    children=[
                                        html.Div('🔵', style={'fontSize': '14px'}),
                                        html.Div(
                                            'RECUPERACIÓN',
                                            style={
                                                'fontSize': '10px',
                                                'fontWeight': '900',
                                                'color': '#1E3A8A',
                                                'letterSpacing': '0.3px'
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    '< 30%',
                                    style={
                                        'fontSize': '11px',
                                        'color': '#1E40AF',
                                        'fontWeight': '700',
                                        'marginBottom': '4px'
                                    }
                                ),
                                html.Div(
                                    'Regenerativa',
                                    style={
                                        'fontSize': '9px',
                                        'color': '#1E3A8A',
                                        'lineHeight': '1.3',
                                        'fontWeight': '600'
                                    }
                                )
                            ]
                        )
                    ]
                )
            ]
        ),

        # Contenedor de gráficos
        html.Div(id={'type': 'graph-output', 'index': 'zonas'})
    ])

@app.callback(
    Output({'type': 'graph-output', 'index': 'timeseries'}, 'children'),
    Input({'type': 'btn-update', 'index': 'timeseries'}, 'n_clicks'),
    State({'type': 'filter-atleta', 'index': 'timeseries'}, 'value'),
    State({'type': 'filter-fechas', 'index': 'timeseries'}, 'start_date'),
    State({'type': 'filter-fechas', 'index': 'timeseries'}, 'end_date'),
    State({'type': 'filter-metrica', 'index': 'timeseries'}, 'value'),
    State('stored-data', 'data'),
    State('stored-positions', 'data'),
    prevent_initial_call=True
)
def update_timeseries_graph(n_clicks, atleta, fechas_start, fechas_end, metrica, stored_data, stored_positions):
    if not n_clicks or stored_data is None:
        return html.Div()
    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        if (not df_pos.empty) and ('Atleta' in df_pos.columns) and ('Posición' in df_pos.columns):
            df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
            if 'Posición_Pos' in df.columns:
                df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
                df = df.drop(columns=['Posición_Pos'])
    return render_timeseries_graph(df, atleta, fechas_start, fechas_end, metrica)

@app.callback(
    Output({'type': 'graph-output', 'index': 'boxplot'}, 'children'),
    Input({'type': 'btn-update', 'index': 'boxplot'}, 'n_clicks'),
    State({'type': 'filter-atleta', 'index': 'boxplot'}, 'value'),
    State({'type': 'filter-metrica-box', 'index': 'boxplot'}, 'value'),
    State({'type': 'filter-categoria-box', 'index': 'boxplot'}, 'value'),
    State('stored-data', 'data'),
    State('stored-positions', 'data'),
    prevent_initial_call=True
)
def update_boxplot_graph(n_clicks, atleta, metrica_box, categoria_box, stored_data, stored_positions):
    if not n_clicks or stored_data is None:
        return html.Div()
    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        if (not df_pos.empty) and ('Atleta' in df_pos.columns) and ('Posición' in df_pos.columns):
            df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
            if 'Posición_Pos' in df.columns:
                df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
                df = df.drop(columns=['Posición_Pos'])
    return render_boxplot_graph(df, atleta, metrica_box, categoria_box)

@app.callback(
    Output({'type': 'graph-output', 'index': 'cuadrante-gral'}, 'children'),
    Input({'type': 'btn-update', 'index': 'cuadrante-gral'}, 'n_clicks'),
    State({'type': 'filter-atleta', 'index': 'cuadrante-gral'}, 'value'),
    State({'type': 'filter-start', 'index': 'cuadrante-gral'}, 'date'),
    State({'type': 'filter-end', 'index': 'cuadrante-gral'}, 'date'),
    State({'type': 'filter-var-x', 'index': 'cuadrante-gral'}, 'value'),
    State({'type': 'filter-var-y', 'index': 'cuadrante-gral'}, 'value'),
    State('stored-data', 'data'),
    State('stored-positions', 'data'),
    prevent_initial_call=True
)
def update_cuadrante_gral_graph(n_clicks, atleta, cuad_start, cuad_end, var_x, var_y, stored_data, stored_positions):
    if not n_clicks or stored_data is None:
        return html.Div()
    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        if (not df_pos.empty) and ('Atleta' in df_pos.columns) and ('Posición' in df_pos.columns):
            df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
            if 'Posición_Pos' in df.columns:
                df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
                df = df.drop(columns=['Posición_Pos'])
    return render_cuadrante_general(df, atleta, cuad_start, cuad_end, var_x, var_y)

@app.callback(
    Output({'type': 'graph-output', 'index': 'historico-md'}, 'children'),
    Input({'type': 'btn-update', 'index': 'historico-md'}, 'n_clicks'),
    State({'type': 'filter-atleta', 'index': 'historico-md'}, 'value'),
    State({'type': 'filter-fecha-objetivo', 'index': 'historico-md'}, 'value'),
    State({'type': 'filter-var-x', 'index': 'historico-md'}, 'value'),
    State({'type': 'filter-var-y', 'index': 'historico-md'}, 'value'),
    State('stored-data', 'data'),
    State('stored-positions', 'data'),
    prevent_initial_call=True
)
def update_historico_md_graph(n_clicks, atleta, fecha_objetivo, var_x, var_y, stored_data, stored_positions):
    if not n_clicks or stored_data is None:
        return html.Div()
    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        if (not df_pos.empty) and ('Atleta' in df_pos.columns) and ('Posición' in df_pos.columns):
            df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
            if 'Posición_Pos' in df.columns:
                df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
                df = df.drop(columns=['Posición_Pos'])
    return render_historico_md(df, atleta, fecha_objetivo, var_x, var_y)

# CALLBACK NUEVO PARA SESIÓN DIARIA
@app.callback(
    Output({'type': 'graph-output', 'index': 'sesion'}, 'children'),
    Input({'type': 'sesion-date', 'index': 'sesion'}, 'date'),
    Input({'type': 'var-1', 'index': 'sesion'}, 'value'),
    Input({'type': 'var-2', 'index': 'sesion'}, 'value'),
    Input({'type': 'var-3', 'index': 'sesion'}, 'value'),
    Input({'type': 'var-4', 'index': 'sesion'}, 'value'),
    State('stored-data', 'data'),
    State('stored-positions', 'data'),
    prevent_initial_call=True
)
def update_sesion_graphs(selected_date, v1, v2, v3, v4, stored_data, stored_positions):
    if stored_data is None or not selected_date:
        return html.Div()
    
    df = pd.DataFrame(stored_data)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    if stored_positions is not None:
        df_pos = pd.DataFrame(stored_positions)
        if (not df_pos.empty) and ('Atleta' in df_pos.columns) and ('Posición' in df_pos.columns):
            df = df.merge(df_pos[['Atleta', 'Posición']], on='Atleta', how='left', suffixes=('', '_pos'))
            if 'Posición_Pos' in df.columns:
                df['Posición'] = df['Posición_Pos'].fillna(df.get('Posición', ''))
                df = df.drop(columns=['Posición_Pos'])

    # Filtrar por fecha
    target_date = pd.to_datetime(selected_date).date()
    df_day = df[df['Fecha'].dt.date == target_date].copy()

    # Consolidar duplicados Atleta-Fecha para visualización (evita doble barra por atleta)
    if 'Atleta' in df_day.columns and not df_day.empty:
        df_day['Atleta'] = df_day['Atleta'].astype(str).str.strip()
        num_cols = df_day.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df_day = df_day.groupby('Atleta', as_index=False)[num_cols].sum()
    
    if df_day.empty:
        return html.Div(f'⚠️ Sin datos para la fecha {target_date}', style={'padding': '20px', 'color': COLORS['red'], 'textAlign': 'center'})

    vars_list = [v for v in [v1, v2, v3, v4] if v]
    
    # 1. TABLA DE PROMEDIOS POR POSICIÓN
    stats_pos = []
    if 'Posición' in df_day.columns:
        # Calcular medias por posición para las variables seleccionadas
        grouped = df_day.groupby('Posición')[vars_list].mean().reset_index()
        # Formatear para display
        for _, row in grouped.iterrows():
            pos_name = row['Posición']
            vals = []
            for v in vars_list:
                vals.append(html.Div([
                    html.Span(f"{v[:10]}..: ", style={'color': '#64748B', 'fontSize': '11px'}),
                    html.Span(fmt_num(row[v], 1), style={'fontWeight': 'bold', 'color': '#0F172A'})
                ]))
            stats_pos.append(html.Div(
                style={'padding': '12px', 'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #E2E8F0'},
                children=[html.Div(pos_name, style={'fontWeight': '900', 'color': COLORS['blue'], 'marginBottom': '8px', 'borderBottom': '2px solid #E2E8F0'})] + vals
            ))
    
    header_stats = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '12px', 'marginBottom': '24px'},
        children=stats_pos
    )

    # 2. GRÁFICOS DE BARRAS
    graphs = []
    
    for var in vars_list:
        if var not in df_day.columns:
            continue
            
        # Calcular promedio del día (referencia)
        day_mean = df_day[var].mean()
        
        # Ordenar atletas por nombre para consistencia visual
        df_plot = df_day.sort_values('Atleta', ascending=False) # Ascending False para que salga A arriba en gráfico horizontal
        
        # Colores condicionales
        colors = [COLORS['blue'] if x > day_mean else '#94A3B8' for x in df_plot[var]]
        
        fig = go.Figure()
        
        # Barras
        fig.add_trace(go.Bar(
            x=df_plot[var],
            y=df_plot['Atleta'],
            orientation='h',
            marker_color=colors,
            text=[fmt_num(x, 0) for x in df_plot[var]],
            textposition='inside',
            insidetextanchor='end',
            textfont=dict(color='white', size=11),
            name=var,
            showlegend=False
        ))
        
        # Línea de promedio
        fig.add_vline(x=day_mean, line_width=2, line_dash="solid", line_color="#1e293b")
        fig.add_annotation(x=day_mean, y=1, yref='paper', text=f"Avg: {fmt_num(day_mean,0)}", showarrow=False, 
                           font=dict(size=10, color="#1e293b"), bgcolor="rgba(255,255,255,0.8)", yanchor='bottom')

        fig.update_layout(
            title=dict(text=var, font=dict(size=14, color=COLORS['slate_800'])),
            margin=dict(l=10, r=10, t=40, b=10),
            height=600, # Altura fija suficiente para lista de jugadores
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            yaxis=dict(tickfont=dict(size=11, color=COLORS['slate_600'])),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        graphs.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))

    graphs_container = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px'},
        children=graphs
    )

    return html.Div([
        html.H4("Resumen por Posición (Promedios)", style={'color': COLORS['slate_700'], 'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '12px'}),
        header_stats,
        html.Div(style={'borderTop': '1px solid #E2E8F0', 'margin': '20px 0'}),
        graphs_container
    ])

# ==========================================
# FUNCIONES DE RENDERIZADO GRÁFICO (REUTILIZADAS)
# ==========================================

def render_timeseries_graph(df, atleta, start_date, end_date, metrica):
    """Renderiza gráfico de serie temporal con Media Móvil y bloque pedagógico."""
    if metrica is None:
        return html.Div('⚠️ Selecciona una métrica', style={'color': COLORS['red'], 'padding': '40px', 'textAlign': 'center'})

    if 'Fecha' not in df.columns:
        return html.Div('⚠️ Falta columna Fecha', style={'color': COLORS['red'], 'padding': '40px', 'textAlign': 'center'})

    df_filt = df.copy()
    df_filt['Fecha'] = pd.to_datetime(df_filt['Fecha'], errors='coerce')
    df_filt = df_filt.dropna(subset=['Fecha'])

    if atleta and atleta != 'TODOS' and 'Atleta' in df_filt.columns:
        df_filt = df_filt[df_filt['Atleta'] == atleta]

    if start_date and end_date:
        s = pd.to_datetime(start_date).date()
        e = pd.to_datetime(end_date).date()
        df_filt = df_filt[(df_filt['Fecha'].dt.date >= s) & (df_filt['Fecha'].dt.date <= e)]

    if df_filt.empty:
        return html.Div('⚠️ Sin datos', style={'color': COLORS['red'], 'padding': '40px', 'textAlign': 'center'})

    if metrica not in df_filt.columns:
        return html.Div('⚠️ Métrica no disponible', style={'color': COLORS['red'], 'padding': '40px', 'textAlign': 'center'})

    df_filt[metrica] = pd.to_numeric(df_filt[metrica], errors='coerce')

    ts_df = (
        df_filt.groupby('Fecha', as_index=False)[metrica]
        .mean()
        .sort_values('Fecha')
        .reset_index(drop=True)
    )

    ts_df['MA_7'] = ts_df[metrica].rolling(window=7, min_periods=1).mean()

    overall_mean = float(ts_df[metrica].mean()) if ts_df[metrica].notna().any() else np.nan
    overall_std = float(ts_df[metrica].std(ddof=1)) if ts_df[metrica].notna().sum() > 1 else np.nan

    ts_df['pct_vs_mean'] = ts_df[metrica].apply(lambda v: safe_pct_change(v, overall_mean))
    ts_df['z_vs_mean'] = (ts_df[metrica] - overall_mean) / (overall_std if (np.isfinite(overall_std) and overall_std != 0) else np.nan)

    customdata = np.column_stack([
        ts_df['MA_7'].to_numpy(dtype='float64'),
        np.full(len(ts_df), overall_mean, dtype='float64'),
        ts_df['pct_vs_mean'].to_numpy(dtype='float64'),
        ts_df['z_vs_mean'].to_numpy(dtype='float64'),
    ])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts_df['Fecha'],
        y=ts_df[metrica],
        mode='lines+markers',
        name=metrica,
        line=dict(color=COLORS['red'], width=4),
        marker=dict(size=9, color=COLORS['red']),
        fill='tozeroy',
        fillcolor='rgba(220, 38, 38, 0.10)',
        customdata=customdata,
        hovertemplate=(
            '<b>%{x|%d/%m/%Y}</b><br>'
            f'<b>{metrica}</b>: %{{y:.2f}}<br>'
            'MM 7 días: %{customdata[0]:.2f}<br>'
            'Media período: %{customdata[1]:.2f}<br>'
            'Δ vs media: %{customdata[2]:.1f}%<br>'
            'Z vs media: %{customdata[3]:.2f}σ'
            '<extra></extra>'
        )
    ))

    fig.add_trace(go.Scatter(
        x=ts_df['Fecha'],
        y=ts_df['MA_7'],
        mode='lines',
        name='MM 7 días',
        line=dict(color=COLORS['blue'], width=5, dash='dash'),
        hovertemplate='<b>%{x|%d/%m/%Y}</b><br><b>MM 7 días</b>: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        template='plotly_white',
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='rgba(0,0,0,0)',
        height=550,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title='<b>FECHA</b>', gridcolor=COLORS['slate_200']),
        yaxis=dict(title=f'<b>{str(metrica).upper()}</b>', gridcolor=COLORS['slate_200']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        autosize=True,
    )

    moving_average_note = html.Div(
        style={
            'marginTop': '16px',
            'borderRadius': '14px',
            'padding': '16px 18px',
            'border': f"2px solid {COLORS['slate_200']}",
            'background': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)',
            'boxShadow': '0 6px 18px rgba(0,0,0,0.06)',
        },
        children=[
            dcc.Markdown(
                "**Definición (Media Móvil):** La Media Móvil (Moving Average) es un indicador estadístico que suaviza las fluctuaciones diarias de los datos para identificar tendencias subyacentes. En este contexto, calcula el promedio de la carga de los últimos 7 días, permitiendo visualizar el estado de forma 'agudo' del atleta sin el ruido de la variabilidad diaria.",
                style={'fontSize': '13px', 'color': COLORS['slate_700'], 'fontWeight': '600', 'lineHeight': '1.6'},
            )
        ]
    )

    return html.Div([
        dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={'width': '100%'}),
        moving_average_note
    ])

def render_boxplot_graph(df, atleta, metrica, categoria):
    """Renderiza box plot con bigotes"""
    if metrica is None or categoria is None:
        return html.Div('⚠️ Selecciona métrica y categoría', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    df_filt = df.copy()
    if atleta and atleta != "TODOS":
        df_filt = df_filt[df_filt['Atleta'] == atleta]
    
    if df_filt.empty or categoria not in df_filt.columns:
        return html.Div('⚠️ Sin datos válidos', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    fig = go.Figure()
    
    categorias_unicas = sorted(df_filt[categoria].dropna().unique())
    
    for cat in categorias_unicas:
        df_cat = df_filt[df_filt[categoria] == cat]
        valores = df_cat[metrica].dropna()
        
        if len(valores) > 0:
            fig.add_trace(go.Box(
                y=valores,
                name=str(cat),
                boxmean='sd',
                marker=dict(
                    color='#2563EB',
                    outliercolor='#DC2626',
                    line=dict(outliercolor='#DC2626', outlierwidth=2)
                ),
                line=dict(color='#1E293B', width=2),
                fillcolor='rgba(37, 99, 235, 0.3)',
                hovertemplate='<b>%{x}</b><br>Valor: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>Distribución de {metrica} por {categoria}</b>",
            font=dict(size=18, color='#0F172A', family='Inter'),
            x=0.5, xanchor='center'
        ),
        xaxis_title=f"<b>{categoria.upper()}</b>",
        yaxis_title=f"<b>{metrica.upper()}</b>",
        template='plotly_white',
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=False,
        xaxis=dict(gridcolor='#E2E8F0', tickfont=dict(size=12)),
        yaxis=dict(gridcolor='#E2E8F0', tickfont=dict(size=12)),
        font=dict(size=13, color='#0F172A', family='Inter')
    )
    
    stats_por_categoria = []
    for cat in categorias_unicas:
        valores = df_filt[df_filt[categoria] == cat][metrica].dropna()
        if len(valores) > 0:
            stats_por_categoria.append({
                categoria: str(cat),
                'N': len(valores),
                'Media': fmt_num(valores.mean(), 2),
                'Mediana': fmt_num(valores.median(), 2),
                'Desv.Std': fmt_num(valores.std(ddof=1), 2),
                'Min': fmt_num(valores.min(), 2),
                'Q1': fmt_num(valores.quantile(0.25), 2),
                'Q3': fmt_num(valores.quantile(0.75), 2),
                'Max': fmt_num(valores.max(), 2)
            })
    
    df_stats = pd.DataFrame(stats_por_categoria)
    
    tabla_stats = html.Div(
        style={'marginTop': '32px'},
        children=[
            html.H4('📊 ESTADÍSTICAS DESCRIPTIVAS POR CATEGORÍA', style={'fontSize': '16px', 'fontWeight': '900', 'color': '#0F172A', 'marginBottom': '16px'}),
            html.Div(
                style={'overflowX': 'auto'},
                children=[
                    html.Table(
                        style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'},
                        children=[
                            html.Thead(
                                html.Tr([html.Th(col, style={'padding': '10px', 'backgroundColor': '#2563EB', 'color': 'white', 'fontWeight': '800', 'textAlign': 'left'}) for col in df_stats.columns])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(str(df_stats.iloc[i][col]), style={'padding': '8px', 'borderBottom': '1px solid #E2E8F0', 'fontWeight': '600'})
                                    for col in df_stats.columns
                                ], style={'backgroundColor': '#EFF6FF' if i % 2 == 0 else '#FFFFFF'})
                                for i in range(len(df_stats))
                            ])
                        ]
                    )
                ]
            )
        ]
    )
    
    return html.Div([
        dcc.Graph(figure=fig, config=GRAPH_CONFIG),
        tabla_stats
    ])

def render_cuadrante_general(df, atleta, start_date, end_date, var_x, var_y):
    """Renderiza cuadrante general"""
    if var_x is None or var_y is None:
        return html.Div('⚠️ Selecciona variables X e Y', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    df_filt = df.copy()
    if atleta and atleta != "TODOS":
        df_filt = df_filt[df_filt['Atleta'] == atleta]
    
    if start_date and end_date:
        df_filt = df_filt[(df_filt['Fecha'].dt.date >= pd.to_datetime(start_date).date()) & 
                          (df_filt['Fecha'].dt.date <= pd.to_datetime(end_date).date())]
    
    if df_filt.empty:
        return html.Div('⚠️ Sin datos', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    df_filt['zscore_x'] = calcular_zscore_seguro(df_filt, var_x)
    df_filt['zscore_y'] = calcular_zscore_seguro(df_filt, var_y)
    
    df_clean = df_filt.dropna(subset=['zscore_x', 'zscore_y']).copy()
    df_clean['color'] = df_clean.apply(asignar_color_cuadrante, axis=1)
    df_clean['cuadrante'] = df_clean.apply(asignar_cuadrante, axis=1)
    df_clean = calcular_indices_riesgo(df_clean)
    
    if df_clean.empty:
        return html.Div('⚠️ Insuficientes datos válidos', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    fig = go.Figure()
    
    df_normal = df_clean[df_clean['color'] == '#78909c']
    df_precaucion = df_clean[df_clean['color'] == '#ffa726']
    df_atencion = df_clean[df_clean['color'] == '#f57c00']
    df_riesgo_alto = df_clean[df_clean['color'] == '#c62828']
    df_riesgo_critico = df_clean[df_clean['color'] == '#8b0000']
    
    grupos = [
        (df_normal, 'NORMAL', 'circle', 11, '#78909c'),
        (df_precaucion, 'PRECAUCIÓN', 'circle', 12, '#ffa726'),
        (df_atencion, 'ATENCIÓN', 'square', 14, '#f57c00'),
        (df_riesgo_alto, 'RIESGO ALTO', 'diamond', 16, '#c62828'),
        (df_riesgo_critico, 'CRÍTICO', 'x', 19, '#8b0000')
    ]
    
    for df_group, nombre, simbolo, tam, color_marker in grupos:
        if len(df_group) > 0:
            if "Posición" in df_group.columns:
                customdata = df_group[[var_x, var_y, 'indice_riesgo_combinado', 'Posición']].values
                hovertemplate = (
                    '<b>%{text}</b><br>' +
                    f'{var_x}: %{{customdata[0]:.2f}}<br>' +
                    f'{var_y}: %{{customdata[1]:.2f}}<br>' +
                    'Z-X: %{x:.2f}σ | Z-Y: %{y:.2f}σ<br>' +
                    'Índice: %{customdata[2]:.2f}<br>' +
                    'Posición: %{customdata[3]}<extra></extra>'
                )
            else:
                customdata = df_group[[var_x, var_y, 'indice_riesgo_combinado']].values
                hovertemplate = (
                    '<b>%{text}</b><br>' +
                    f'{var_x}: %{{customdata[0]:.2f}}<br>' +
                    f'{var_y}: %{{customdata[1]:.2f}}<br>' +
                    'Z-X: %{x:.2f}σ | Z-Y: %{y:.2f}σ<br>' +
                    'Índice: %{customdata[2]:.2f}<extra></extra>'
                )
            
            fig.add_trace(go.Scatter(
                x=df_group['zscore_x'], y=df_group['zscore_y'], mode='markers+text', name=nombre,
                text=df_group['Atleta'], textposition='top center',
                textfont=dict(size=10, color='#1a1a1a', family='Inter'),
                marker=dict(size=tam, color=color_marker, line=dict(width=2, color='white'), symbol=simbolo),
                hovertemplate=hovertemplate, customdata=customdata
            ))
    
    fig = graficar_cuadrantes_con_etiquetas(fig)
    
    fig.update_layout(
        title=dict(text=f"<b>Cuadrantes: {var_x} vs {var_y}</b>", font=dict(size=18, color='#0a1929', family='Inter'), x=0.5, xanchor='center'),
        xaxis_title=f"<b>{var_x} (Z-Score)</b>", yaxis_title=f"<b>{var_y} (Z-Score)</b>",
        height=750, showlegend=True, hovermode='closest',
        plot_bgcolor='#ffffff', paper_bgcolor='#f5f7fa',
        xaxis=dict(gridcolor='#e0e0e0', gridwidth=1, range=[-3,3]),
        yaxis=dict(gridcolor='#e0e0e0', gridwidth=1, range=[-3,3])
    )
    
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG)

def render_historico_md(df, atleta, fecha_objetivo, var_x, var_y):
    """Renderiza análisis histórico contextual por MD"""
    if fecha_objetivo is None or var_x is None or var_y is None or 'MD' not in df.columns:
        return html.Div('⚠️ Selecciona fecha y variables. Requiere columna "MD"', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center', 'fontWeight': '800'})
    
    fecha_dt = pd.to_datetime(fecha_objetivo)
    registros_fecha = df[df['Fecha'] == fecha_dt]
    
    if registros_fecha.empty:
        return html.Div('⚠️ Sin datos para fecha seleccionada', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    md_values = registros_fecha['MD'].dropna().unique()
    
    if len(md_values) == 0:
        return html.Div('⚠️ Fecha sin valor de MD', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    valor_md = md_values[0]
    
    df_historico, df_objetivo, stats_hist = calcular_zscore_historico_md(
        df_completo=df,
        fecha_objetivo=fecha_dt,
        valor_md=valor_md,
        columna_x=var_x,
        columna_y=var_y,
        atleta=atleta
    )
    
    if 'warning' in stats_hist:
        return html.Div(
            style={'padding': '40px', 'textAlign': 'center', 'backgroundColor': '#FEF3C7', 'borderRadius': '12px', 'border': '2px solid #FDE68A'},
            children=[
                html.Div('⚠️', style={'fontSize': '64px', 'marginBottom': '16px'}),
                html.Div(stats_hist['warning'], style={'fontSize': '16px', 'fontWeight': '700', 'color': '#78350F'})
            ]
        )
    
    if df_historico.empty:
        return html.Div('⚠️ Sin datos históricos', style={'color': '#DC2626', 'padding': '40px', 'textAlign': 'center'})
    
    fig = go.Figure()
    
    df_hist_otros = df_historico[~df_historico['es_fecha_objetivo']]
    if not df_hist_otros.empty:
        fig.add_trace(go.Scatter(
            x=df_hist_otros['zscore_x'],
            y=df_hist_otros['zscore_y'],
            mode='markers',
            name=f'Histórico {valor_md}',
            text=df_hist_otros['Atleta'],
            marker=dict(size=10, color='#94A3B8', opacity=0.4, line=dict(width=1, color='white')),
            hovertemplate='<b>%{text}</b><br>Z-X: %{x:.2f}σ | Z-Y: %{y:.2f}σ<extra></extra>'
        ))
    
    df_fecha_obj = df_historico[df_historico['es_fecha_objetivo']]
    if not df_fecha_obj.empty:
        fig.add_trace(go.Scatter(
            x=df_fecha_obj['zscore_x'],
            y=df_fecha_obj['zscore_y'],
            mode='markers+text',
            name=f'🎯 {fecha_dt.strftime("%d/%m/%Y")}',
            text=df_fecha_obj['Atleta'],
            textposition='top center',
            textfont=dict(size=12, color='#DC2626', family='Inter'),
            marker=dict(size=20, color='#DC2626', symbol='star', line=dict(width=3, color='#FFFFFF')),
            hovertemplate='<b>🎯 %{text}</b><br>Z-X: %{x:.2f}σ | Z-Y: %{y:.2f}σ<br><b>FECHA OBJETIVO</b><extra></extra>'
        ))
    
    df_outliers = df_hist_otros[df_hist_otros['outlier_extremo']]
    if not df_outliers.empty:
        fig.add_trace(go.Scatter(
            x=df_outliers['zscore_x'],
            y=df_outliers['zscore_y'],
            mode='markers',
            name='Outliers Extremos (>4σ)',
            text=df_outliers['Atleta'],
            marker=dict(size=14, color='#8B0000', symbol='x', line=dict(width=2, color='white')),
            hovertemplate='<b>⚠️ %{text}</b><br>Z-X: %{x:.2f}σ | Z-Y: %{y:.2f}σ<br><b>OUTLIER EXTREMO</b><extra></extra>'
        ))
    
    fig = graficar_cuadrantes_con_etiquetas(fig)
    
    fig.update_layout(
        title=dict(
            text=f"<b>Análisis Histórico MD: {valor_md}</b><br><sub>Fecha Objetivo: {fecha_dt.strftime('%d/%m/%Y')} vs {stats_hist['n_total']} registros históricos</sub>",
            font=dict(size=18, color='#0a1929', family='Inter'),
            x=0.5, xanchor='center'
        ),
        xaxis_title=f"<b>{var_x} (Z-Score vs Histórico {valor_md})</b>",
        yaxis_title=f"<b>{var_y} (Z-Score vs Histórico {valor_md})</b>",
        height=750, showlegend=True, hovermode='closest',
        plot_bgcolor='#ffffff', paper_bgcolor='#FFFBEB',
        xaxis=dict(gridcolor='#FDE68A', gridwidth=1, range=[-4,4]),
        yaxis=dict(gridcolor='#FDE68A', gridwidth=1, range=[-4,4]),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, bgcolor='rgba(255,255,255,0.95)', bordercolor='#FDE68A', borderwidth=2)
    )
    
    stats_x = stats_hist['x']
    stats_y = stats_hist['y']
    
    panel_stats = html.Div(
        style={'marginTop': '32px', 'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'},
        children=[
            html.Div(
                style={'background': 'rgba(255,255,255,0.9)', 'borderRadius': '16px', 'padding': '24px', 'border': '2px solid #FDE68A'},
                children=[
                    html.H4(f'📊 ESTADÍSTICAS POBLACIONALES • {var_x}', style={'fontSize': '15px', 'fontWeight': '900', 'color': '#92400E', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div([html.Strong('Media (μ): '), fmt_num(stats_x['mean'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('Desv.Std (σ): '), fmt_num(stats_x['std'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('Mediana: '), fmt_num(stats_x['median'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('n histórico: '), str(stats_x['n'])], style={'fontSize': '13px', 'color': '#92400E', 'fontWeight': '700'})
                    ])
                ]
            ),
            html.Div(
                style={'background': 'rgba(255,255,255,0.9)', 'borderRadius': '16px', 'padding': '24px', 'border': '2px solid #FDE68A'},
                children=[
                    html.H4(f'📊 ESTADÍSTICAS POBLACIONALES • {var_y}', style={'fontSize': '15px', 'fontWeight': '900', 'color': '#92400E', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div([html.Strong('Media (μ): '), fmt_num(stats_y['mean'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('Desv.Std (σ): '), fmt_num(stats_y['std'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('Mediana: '), fmt_num(stats_y['median'], 2)], style={'fontSize': '13px', 'color': '#78350F', 'marginBottom': '8px', 'fontWeight': '600'}),
                        html.Div([html.Strong('n histórico: '), str(stats_y['n'])], style={'fontSize': '13px', 'color': '#92400E', 'fontWeight': '700'})
                    ])
                ]
            )
        ]
    )

    if not df_fecha_obj.empty:
        primer_registro = df_fecha_obj.iloc[0]
        panel_objetivo = html.Div(
            style={'marginTop': '24px', 'background': 'linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%)', 'borderRadius': '16px', 'padding': '24px', 'border': '3px solid #DC2626'},
            children=[
                html.H4(f'🎯 VALORES FECHA OBJETIVO: {fecha_dt.strftime("%d/%m/%Y")}', style={'fontSize': '16px', 'fontWeight': '900', 'color': '#7C2D12', 'marginBottom': '16px'}),
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px'},
                    children=[
                        html.Div([
                            html.Div(var_x, style={'fontSize': '11px', 'fontWeight': '800', 'color': '#7C2D12', 'marginBottom': '8px'}),
                            html.Div(fmt_num(primer_registro[var_x], 2), style={'fontSize': '24px', 'fontWeight': '900', 'color': '#DC2626'})
                        ]),
                        html.Div([
                            html.Div(f'Z-score {var_x}', style={'fontSize': '11px', 'fontWeight': '800', 'color': '#7C2D12', 'marginBottom': '8px'}),
                            html.Div(fmt_num(primer_registro['zscore_x'], 2), style={'fontSize': '24px', 'fontWeight': '900', 'color': '#DC2626'})
                        ]),
                        html.Div([
                            html.Div(var_y, style={'fontSize': '11px', 'fontWeight': '800', 'color': '#7C2D12', 'marginBottom': '8px'}),
                            html.Div(fmt_num(primer_registro[var_y], 2), style={'fontSize': '24px', 'fontWeight': '900', 'color': '#DC2626'})
                        ]),
                        html.Div([
                            html.Div(f'Z-score {var_y}', style={'fontSize': '11px', 'fontWeight': '800', 'color': '#7C2D12', 'marginBottom': '8px'}),
                            html.Div(fmt_num(primer_registro['zscore_y'], 2), style={'fontSize': '24px', 'fontWeight': '900', 'color': '#DC2626'})
                        ])
                    ]
                )
            ]
        )
    else:
        panel_objetivo = html.Div()

    return html.Div([
        dcc.Graph(figure=fig, config=GRAPH_CONFIG),
        panel_stats,
        panel_objetivo
    ])



@app.callback(
    Output({'type': 'graph-output', 'index': 'equipo'}, 'children'),
    Input({'type': 'btn-update', 'index': 'equipo'}, 'n_clicks'),
    Input({'type': 'filter-date', 'index': 'equipo'}, 'date'),
    State({'type': 'filter-vars', 'index': 'equipo'}, 'value'),
    State('stored-data', 'data'),
    State('stored-matches-data', 'data'),
    prevent_initial_call=True
)
def update_equipo_graphs(nclicks, datestr, selectedvars, storeddata, storedmatches):
    if not datestr or storeddata is None:
        return html.Div()

    df = pd.DataFrame(storeddata)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    selecteddate = pd.to_datetime(datestr).date()
    dfday = df[df['Fecha'].dt.date == selecteddate]

    if dfday.empty:
        return html.Div(f"Sin datos para {selecteddate}", style={'color': COLORS['red']})

    dfmatches = pd.DataFrame(storedmatches) if storedmatches else pd.DataFrame()

    gauges = []
    if not selectedvars:
        return html.Div("Selecciona variables.")

    for var in selectedvars:
        if var not in dfday.columns:
            continue

        currentval = dfday[var].mean()

        # Calcular referencia de partido (promedio + máximo) / 2
        matchref = 0
        if not dfmatches.empty and var in dfmatches.columns:
            matchmean = pd.to_numeric(dfmatches[var], errors='coerce').mean()
            matchmax = pd.to_numeric(dfmatches[var], errors='coerce').max()
            matchref = (matchmean + matchmax) / 2

        if matchref == 0 or np.isnan(matchref):
            matchref = currentval * 1.5 if currentval > 0 else 100

        # Calcular porcentaje de carga
        pct_load = (currentval / matchref * 100) if matchref > 0 else 0

        # Asignar colores y etiquetas según el porcentaje
        if pct_load >= 60:
            gaugecolor = '#DC2626'
            loadlabel = "DESARROLLO"
        elif 40 <= pct_load < 60:
            gaugecolor = '#D97706'
            loadlabel = "MANTENIMIENTO"
        elif 30 <= pct_load < 40:
            gaugecolor = '#059669'
            loadlabel = "ACTIVACIÓN"
        else:
            gaugecolor = '#93C5FD'
            loadlabel = "RECUPERACIÓN"

        # Crear gráfico gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=currentval,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{var}</b>", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [None, matchref * 1.2], 'tickwidth': 1},
                'bar': {'color': gaugecolor},
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': matchref
                }
            }
        ))

        # Añadir anotación del porcentaje
        fig.add_annotation(
            x=0.5, y=-0.25,
            text=f"{pct_load:.1f}%",
            showarrow=False,
            font=dict(size=30, color=gaugecolor)
        )

        # Añadir etiqueta de categoría de carga
        fig.add_annotation(
            x=0.5, y=-0.10,
            text=loadlabel,
            showarrow=False,
            font=dict(size=12, color=gaugecolor)
        )

        fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=50, b=20)
        )

        gauges.append(
            html.Div(
                style={
                    'backgroundColor': 'white',
                    'borderRadius': '16px',
                    'padding': '16px',
                    'border': f'2px solid {gaugecolor}'
                },
                children=dcc.Graph(figure=fig, config={'displayModeBar': False})
            )
        )

    return html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))',
            'gap': '24px'
        },
        children=gauges
    )




@app.callback(
    Output({'type': 'graph-output', 'index': 'zonas'}, 'children'),
    Input({'type': 'zonas-date', 'index': 'zonas'}, 'date'),
    Input({'type': 'zonas-vars', 'index': 'zonas'}, 'value'),
    State('stored-data', 'data'),
    State('stored-matches-data', 'data'),
    prevent_initial_call=True
)
def update_zonas_carga_graphs(selected_date, selected_vars, stored_data, stored_matches):
    if not selected_date or stored_data is None or not selected_vars:
        return html.Div(
            style={'padding': '60px', 'textAlign': 'center', 'backgroundColor': '#FEF2F2', 'borderRadius': '20px', 'border': '2px dashed #DC2626'},
            children=[
                html.Div('⚠️', style={'fontSize': '48px', 'marginBottom': '16px'}),
                html.Div('Selecciona fecha y variables para visualizar los gráficos', 
                        style={'fontSize': '16px', 'fontWeight': '600', 'color': '#DC2626'})
            ]
        )

    df = pd.DataFrame(stored_data)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    selected_date_dt = pd.to_datetime(selected_date).date()
    df_day = df[df['Fecha'].dt.date == selected_date_dt].copy()

    if df_day.empty:
        return html.Div(
            style={'padding': '60px', 'textAlign': 'center', 'backgroundColor': '#FEF2F2', 'borderRadius': '20px', 'border': '2px dashed #DC2626'},
            children=[
                html.Div('📅', style={'fontSize': '48px', 'marginBottom': '16px'}),
                html.Div(f'Sin datos para la fecha {selected_date_dt.strftime("%d/%m/%Y")}', 
                        style={'fontSize': '16px', 'fontWeight': '600', 'color': '#DC2626'})
            ]
        )

    # Obtener MD del día
    md_value = df_day['MD'].mode()[0] if 'MD' in df_day.columns and df_day['MD'].notna().any() else 'N/A'

    # Obtener datos del día anterior
    df_sorted = df.sort_values('Fecha')
    df_previous = df[df['Fecha'].dt.date < selected_date_dt]

    # Cargar datos de partidos para referencias
    df_matches = pd.DataFrame(stored_matches) if stored_matches else pd.DataFrame()

    gauges = []

    for var in selected_vars:
        if var not in df_day.columns:
            continue

        # Valor actual (promedio del equipo)
        current_val = df_day[var].mean()

        # Calcular referencia de partido: (promedio + máximo) / 2
        match_ref = 0
        if not df_matches.empty and var in df_matches.columns:
            match_vals = pd.to_numeric(df_matches[var], errors='coerce').dropna()
            if len(match_vals) > 0:
                match_mean = match_vals.mean()
                match_max = match_vals.max()
                match_ref = (match_mean + match_max) / 2

        # Fallback si no hay referencia
        if match_ref == 0 or np.isnan(match_ref):
            match_ref = current_val * 1.5 if current_val > 0 else 100

        # Calcular porcentaje de carga
        pct_load = (current_val / match_ref * 100) if match_ref > 0 else 0

        # Asignar zona de carga y color CON GRADIENTES
        if pct_load >= 60:
            gauge_color = '#DC2626'
            gauge_bg = 'linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)'
            load_label = 'DESARROLLO'
            load_emoji = '🔴'
            zone_letter = 'D'
        elif 40 <= pct_load < 60:
            gauge_color = '#F59E0B'
            gauge_bg = 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)'
            load_label = 'MANTENIMIENTO'
            load_emoji = '🟡'
            zone_letter = 'M'
        elif 30 <= pct_load < 40:
            gauge_color = '#059669'
            gauge_bg = 'linear-gradient(135deg, #059669 0%, #047857 100%)'
            load_label = 'ACTIVACIÓN'
            load_emoji = '🟢'
            zone_letter = 'A'
        else:
            gauge_color = '#3B82F6'
            gauge_bg = 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)'
            load_label = 'RECUPERACIÓN'
            load_emoji = '🔵'
            zone_letter = 'R'

        # Calcular diferencia vs día anterior
        pct_diff = 0
        diff_text = 'Sin datos'
        diff_arrow = '━'
        if not df_previous.empty and var in df_previous.columns:
            last_dates = df_previous.groupby(df_previous['Fecha'].dt.date)[var].mean().sort_index()
            if len(last_dates) > 0:
                prev_val = last_dates.iloc[-1]
                if prev_val > 0:
                    pct_diff = ((current_val - prev_val) / prev_val) * 100
                    diff_arrow = '▲' if pct_diff > 0 else '▼' if pct_diff < 0 else '━'
                    diff_text = f"{diff_arrow} {abs(pct_diff):.1f}%"

        diff_color = '#059669' if pct_diff > 0 else '#DC2626' if pct_diff < 0 else '#64748B'

        # Crear gráfico gauge MEJORADO VISUALMENTE
        fig = go.Figure()

        # Gauge principal
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=current_val,
            domain={'x': [0, 1], 'y': [0.2, 1]},
            title={
                'text': f"<b>{var}</b>",
                'font': {'size': 18, 'color': '#0F172A', 'family': 'Inter, sans-serif'}
            },
            number={
                'font': {'size': 28, 'color': gauge_color, 'family': 'Inter, sans-serif'},
                'valueformat': '.1f'
            },
            gauge={
                'axis': {
                    'range': [0, match_ref * 1.35],
                    'tickwidth': 2,
                    'tickcolor': '#CBD5E1',
                    'tickfont': {'size': 10, 'color': '#64748B'}
                },
                'bar': {
                    'color': gauge_color,
                    'thickness': 0.7
                },
                'bgcolor': '#F1F5F9',
                'borderwidth': 3,
                'bordercolor': '#E2E8F0',
                'steps': [
                    {'range': [0, match_ref * 0.3], 'color': '#E0F2FE'},
                    {'range': [match_ref * 0.3, match_ref * 0.4], 'color': '#D1FAE5'},
                    {'range': [match_ref * 0.4, match_ref * 0.6], 'color': '#FEF3C7'},
                    {'range': [match_ref * 0.6, match_ref * 1.35], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': '#0F172A', 'width': 5},
                    'thickness': 0.85,
                    'value': match_ref
                }
            }
        ))

        # PORCENTAJE (MÁS PEQUEÑO - 20px en lugar de 36)
        fig.add_annotation(
            x=0.5, y=0.08,
            text=f"<b>{pct_load:.1f}%</b>",
            showarrow=False,
            font=dict(size=20, color=gauge_color, family='Inter, sans-serif'),
            xref='paper',
            yref='paper'
        )

        # ZONA DE CARGA (más prominente)
        fig.add_annotation(
            x=0.5, y=-0.02,
            text=f"<b>{load_emoji} {load_label} ({zone_letter})</b>",
            showarrow=False,
            font=dict(size=13, color=gauge_color, family='Inter, sans-serif'),
            xref='paper',
            yref='paper'
        )

        # DIFERENCIA VS DÍA ANTERIOR (más visible)
        fig.add_annotation(
            x=0.5, y=-0.12,
            text=f"<b>vs Anterior: {diff_text}</b>",
            showarrow=False,
            font=dict(size=11, color=diff_color, family='Inter, sans-serif'),
            xref='paper',
            yref='paper'
        )

        fig.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=70, b=50),
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            font={'family': 'Inter, sans-serif'}
        )

        # Contenedor del gauge CON DISEÑO PREMIUM
        gauges.append(
            html.Div(
                style={
                    'background': 'linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%)',
                    'borderRadius': '20px',
                    'padding': '24px',
                    'border': f'3px solid {gauge_color}',
                    'boxShadow': f'0 8px 32px rgba(0, 0, 0, 0.12), 0 0 0 1px {gauge_color}33',
                    'position': 'relative',
                    'overflow': 'hidden',
                    'transition': 'all 0.3s ease'
                },
                children=[
                    # Badge de zona en esquina superior derecha
                    html.Div(
                        style={
                            'position': 'absolute',
                            'top': '16px',
                            'right': '16px',
                            'background': gauge_bg,
                            'color': 'white',
                            'padding': '6px 12px',
                            'borderRadius': '20px',
                            'fontSize': '11px',
                            'fontWeight': '900',
                            'letterSpacing': '0.5px',
                            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.2)'
                        },
                        children=f"{zone_letter}"
                    ),
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ]
            )
        )

    # Panel de información de la sesión MEJORADO
    info_panel = html.Div(
        style={
            'background': 'linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)',
            'padding': '28px',
            'borderRadius': '20px',
            'marginBottom': '32px',
            'boxShadow': '0 12px 40px rgba(30, 58, 138, 0.3)',
            'border': '2px solid rgba(255, 255, 255, 0.2)'
        },
        children=[
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                    'gap': '24px'
                },
                children=[
                    html.Div(
                        style={
                            'backgroundColor': 'rgba(255, 255, 255, 0.15)',
                            'padding': '20px',
                            'borderRadius': '16px',
                            'textAlign': 'center',
                            'backdropFilter': 'blur(10px)',
                            'border': '1px solid rgba(255, 255, 255, 0.2)'
                        },
                        children=[
                            html.Div('📅', style={'fontSize': '28px', 'marginBottom': '8px'}),
                            html.Div('FECHA DE SESIÓN', 
                                    style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255, 255, 255, 0.8)', 
                                           'marginBottom': '8px', 'letterSpacing': '1px'}),
                            html.Div(selected_date_dt.strftime('%d/%m/%Y'), 
                                    style={'fontSize': '22px', 'fontWeight': '900', 'color': 'white'})
                        ]
                    ),
                    html.Div(
                        style={
                            'backgroundColor': 'rgba(220, 38, 38, 0.2)',
                            'padding': '20px',
                            'borderRadius': '16px',
                            'textAlign': 'center',
                            'backdropFilter': 'blur(10px)',
                            'border': '2px solid #DC2626'
                        },
                        children=[
                            html.Div('📌', style={'fontSize': '28px', 'marginBottom': '8px'}),
                            html.Div('TIPO DE SESIÓN', 
                                    style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255, 255, 255, 0.8)', 
                                           'marginBottom': '8px', 'letterSpacing': '1px'}),
                            html.Div(f'MD {md_value}', 
                                    style={'fontSize': '22px', 'fontWeight': '900', 'color': 'white'})
                        ]
                    ),
                    html.Div(
                        style={
                            'backgroundColor': 'rgba(255, 255, 255, 0.15)',
                            'padding': '20px',
                            'borderRadius': '16px',
                            'textAlign': 'center',
                            'backdropFilter': 'blur(10px)',
                            'border': '1px solid rgba(255, 255, 255, 0.2)'
                        },
                        children=[
                            html.Div('👥', style={'fontSize': '28px', 'marginBottom': '8px'}),
                            html.Div('ATLETAS EN SESIÓN', 
                                    style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255, 255, 255, 0.8)', 
                                           'marginBottom': '8px', 'letterSpacing': '1px'}),
                            html.Div(str(len(df_day)), 
                                    style={'fontSize': '22px', 'fontWeight': '900', 'color': 'white'})
                        ]
                    )
                ]
            )
        ]
    )

    # Info panel DENTRO del grid (4 columnas)
    info_card = html.Div(
        style={
            'background': 'linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)',
            'padding': '28px',
            'borderRadius': '20px',
            'boxShadow': '0 12px 40px rgba(30, 58, 138, 0.3)',
            'border': '3px solid rgba(255, 255, 255, 0.3)',
            'gridColumn': '1 / -1',
            'display': 'grid',
            'gridTemplateColumns': 'repeat(3, 1fr)',
            'gap': '20px'
        },
        children=[
            html.Div(
                style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.15)',
                    'padding': '24px',
                    'borderRadius': '16px',
                    'textAlign': 'center',
                    'backdropFilter': 'blur(10px)',
                    'border': '1px solid rgba(255, 255, 255, 0.2)'
                },
                children=[
                    html.Div('📅', style={'fontSize': '36px', 'marginBottom': '12px'}),
                    html.Div('FECHA DE SESIÓN', 
                            style={'fontSize': '12px', 'fontWeight': '800', 'color': 'rgba(255, 255, 255, 0.8)', 
                                   'marginBottom': '10px', 'letterSpacing': '1.5px'}),
                    html.Div(selected_date_dt.strftime('%d/%m/%Y'), 
                            style={'fontSize': '26px', 'fontWeight': '900', 'color': 'white'})
                ]
            ),
            html.Div(
                style={
                    'backgroundColor': 'rgba(220, 38, 38, 0.25)',
                    'padding': '24px',
                    'borderRadius': '16px',
                    'textAlign': 'center',
                    'backdropFilter': 'blur(10px)',
                    'border': '2px solid #DC2626'
                },
                children=[
                    html.Div('📌', style={'fontSize': '36px', 'marginBottom': '12px'}),
                    html.Div('TIPO DE SESIÓN', 
                            style={'fontSize': '12px', 'fontWeight': '800', 'color': 'rgba(255, 255, 255, 0.8)', 
                                   'marginBottom': '10px', 'letterSpacing': '1.5px'}),
                    html.Div(f'MD {md_value}', 
                            style={'fontSize': '26px', 'fontWeight': '900', 'color': 'white'})
                ]
            ),
            html.Div(
                style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.15)',
                    'padding': '24px',
                    'borderRadius': '16px',
                    'textAlign': 'center',
                    'backdropFilter': 'blur(10px)',
                    'border': '1px solid rgba(255, 255, 255, 0.2)'
                },
                children=[
                    html.Div('👥', style={'fontSize': '36px', 'marginBottom': '12px'}),
                    html.Div('ATLETAS EN SESIÓN', 
                            style={'fontSize': '12px', 'fontWeight': '800', 'color': 'rgba(255, 255, 255, 0.8)', 
                                   'marginBottom': '10px', 'letterSpacing': '1.5px'}),
                    html.Div(str(len(df_day)), 
                            style={'fontSize': '26px', 'fontWeight': '900', 'color': 'white'})
                ]
            )
        ]
    )

    # Grid de 4 columnas con info integrada
    all_elements = [info_card] + gauges

    return html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gap': '28px'
        },
        children=all_elements
    )

if __name__ == '__main__':
    app.run(
        debug=os.getenv('CAI_DASH_DEBUG', '1') == '1',
        host=os.getenv('CAI_DASH_HOST', '127.0.0.1'),
        port=int(os.getenv('CAI_DASH_PORT', '8050'))
    )
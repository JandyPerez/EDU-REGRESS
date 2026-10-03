"""
EDU·REGRESS — Regresión lineal de datos educativos  ·  versión CREMA
Requisitos: pip install pandas numpy matplotlib openpyxl scipy
Ejecutar:   python regresion_educacion.py

Los datos y la app tienen en cuenta el MES: columnas Año y Mes, filtro por rango de
meses y tres gráficos temporales. Tipos de gráfico (selector sobre el gráfico):
  1. Dispersión + regresión (con banda de confianza 95 %)
  2. Residuos vs ajustados
  3. Histogramas de X e Y (con curva de densidad)
  4. Distribución de residuos (histograma + Q-Q)
  5. Caja por provincia
  6. Violín por nivel
  7. Barras: media por provincia (con error estándar)
  8. Tendencia: media de Y según X
  9. Densidad hexagonal
 10. Mapa de calor de correlaciones
 11. Donut de observaciones
 12. Serie temporal mensual (con media móvil de 12 meses)
 13. Estacionalidad: media de Y por mes
 14. Mapa de calor año × mes
 15. Panel resumen (4 gráficos en uno)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from statistics import NormalDist

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from scipy.stats import linregress
except ImportError:
    linregress = None

# ---------- Paleta crema ----------
BG, PANEL, FIELD, BORDER = "#f3e9d2", "#fbf6e9", "#fffdf6", "#dccaa0"
TEXT, MUTED = "#43352a", "#8a7a63"
ACCENT, TERRA, PLUM, OLIVE = "#b5651d", "#c4573a", "#8a5a78", "#6f7d3c"
ON_ACCENT = "#fffaf0"
COLORES = [ACCENT, TERRA, "#6f8f72", "#d4a017", PLUM, "#4f7f9a", "#a0522d", "#7a8450"]
CMAP = LinearSegmentedColormap.from_list("crema", ["#fbf1d9", "#e8c987", "#c98b3c", "#8e4a1c"])
CMAP_DIV = LinearSegmentedColormap.from_list("crema_div", ["#4f7f9a", "#fbf6e9", "#c4573a"])
FONT = "Segoe UI"
FONT_TITULO = "Georgia"

TIPOS = [
    "Dispersión + regresión",
    "Residuos vs ajustados",
    "Histogramas de X e Y",
    "Distribución de residuos (hist + Q-Q)",
    "Caja por provincia",
    "Violín por nivel",
    "Barras: media por provincia",
    "Tendencia: media de Y según X",
    "Densidad hexagonal",
    "Mapa de calor de correlaciones",
    "Donut de observaciones",
    "Serie temporal mensual",
    "Estacionalidad: media de Y por mes",
    "Mapa de calor año × mes",
    "Panel resumen (4 gráficos)",
]


# =====================================================================
#  Datos y estadística
# =====================================================================
MESES_ABREV = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
_MES_TXT = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9,
            "set": 9, "oct": 10, "nov": 11, "dic": 12, "jan": 1, "apr": 4, "aug": 8, "dec": 12}


def mes_numerico(serie):
    """Devuelve el mes como número 1-12 (acepta números o nombres: enero, Feb, mar…)."""
    if pd.api.types.is_numeric_dtype(serie):
        m = serie.astype(float)
    else:
        m = serie.astype(str).str.strip().str.lower().str[:3].map(_MES_TXT)
    m = m.where(m.between(1, 12))
    if m.notna().mean() < 0.5:
        raise ValueError("La columna de Mes no tiene meses válidos (1–12 o nombres de mes)")
    return m


_PISTAS = {"nivel": ("nivel", "level"), "provincia": ("provincia", "region", "región", "prov"),
           "edad": ("edad", "age"), "mes": ("mes", "month"), "anio": ("anio", "año", "ano", "year")}


def detectar_columna(cols, clave):
    """Busca primero un nombre exacto y luego una columna que contenga la pista."""
    for h in _PISTAS[clave]:
        for c in cols:
            if c.lower() == h:
                return c
    for h in _PISTAS[clave]:
        for c in cols:
            if h in c.lower():
                return c
    return ""


def regresion(x, y):
    if linregress:
        r = linregress(x, y)
        return r.slope, r.intercept, r.rvalue ** 2, r.pvalue
    m, b = np.polyfit(x, y, 1)
    return m, b, np.corrcoef(x, y)[0, 1] ** 2, None


def datos_ejemplo(n=10_000, semilla=7):
    """10 000 filas con año y mes, relaciones claras y muy poco ruido (datos concentrados).

    - El gasto educativo sube de forma continua con el tiempo (año + mes).
    - La escolarización crece con el gasto y baja un poco en los meses de vacaciones.
    - El abandono baja cuando sube el gasto y sube algo en los meses de vacaciones.
    """
    rng = np.random.default_rng(semilla)
    provincias = np.array(["Norte", "Sur", "Este", "Oeste", "Centro", "Costa"])
    efecto_prov = dict(zip(provincias, [1.5, -1.0, 0.5, -0.5, 2.0, 0.0]))

    prov = rng.choice(provincias, n)
    anio = rng.integers(2010, 2024, n)
    mes = rng.integers(1, 13, n)
    nivel = rng.choice(["Primaria", "Secundaria"], n)
    edad = np.where(nivel == "Primaria", rng.integers(6, 12, n), rng.integers(12, 18, n))
    ep = np.array([efecto_prov[p] for p in prov])
    sec = (nivel == "Secundaria").astype(float)
    periodo = anio + (mes - 1) / 12
    vacac = np.cos(2 * np.pi * (mes - 1) / 12)          # +1 en enero (vacaciones), -1 en julio

    gasto = 3 + 0.25 * (periodo - 2010) + 0.04 * ep + rng.normal(0, 0.05, n)
    esc = 78 + 3 * gasto + 0.25 * ep - 0.6 * sec - 1.0 * vacac + rng.normal(0, 0.45, n)
    aband = 26 - 3 * gasto + 0.5 * sec + 0.8 * vacac + rng.normal(0, 0.4, n)

    df = pd.DataFrame({
        "anio": anio, "mes": mes, "periodo": np.round(periodo, 3),
        "provincia": prov, "nivel": nivel, "edad": edad,
        "tasa_escolarizacion": np.round(np.clip(esc, 0, 100), 2),
        "gasto_educativo_pib": np.round(gasto, 2),
        "tasa_abandono": np.round(np.clip(aband, 0, None), 2),
    })
    return df.sort_values(["anio", "mes"]).reset_index(drop=True)


def limpiar_numericas(df):
    """Convierte columnas de texto que en realidad son números (incluye coma decimal)."""
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
            if conv.notna().mean() > 0.8:
                df[c] = conv
    return df


def kde(v, n=200):
    """Densidad de kernel gaussiano (regla de Silverman), sin depender de scipy."""
    v = np.asarray(v, float)
    if len(v) > 3000:
        v = np.random.default_rng(0).choice(v, 3000, replace=False)
    s = v.std() or 1.0
    h = 1.06 * s * len(v) ** -0.2
    xs = np.linspace(v.min() - 3 * h, v.max() + 3 * h, n)
    d = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / h) ** 2).sum(1) / (len(v) * h * np.sqrt(2 * np.pi))
    return xs, d


def grupos_de(df, gcol, y):
    """Lista [(nombre, valores_y)] por grupo; un único grupo 'Todos' si no hay columna."""
    if gcol and gcol in df:
        return [(str(k), g[y].dropna().values) for k, g in df.groupby(gcol) if g[y].notna().any()]
    return [("Todos", df[y].dropna().values)]


def stat_linea(nombre, v):
    desv = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{str(nombre):<12} n={len(v)}  media={v.mean():.3f}  mediana={np.median(v):.3f}  desv={desv:.3f}"


# =====================================================================
#  Gráficos (funciones independientes de la interfaz)
# =====================================================================
def estilo_ax(ax, grid=True):
    ax.set_facecolor(PANEL)
    for k, s in ax.spines.items():
        s.set_color(BORDER)
        if k in ("top", "right"):
            s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if grid:
        ax.grid(color=BORDER, alpha=0.7, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def titulo(ax, txt):
    ax.set_title(txt, color=TEXT, fontsize=12, fontweight="bold", loc="left", fontfamily=FONT_TITULO)


def leyenda(ax, **kw):
    return ax.legend(facecolor=FIELD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, framealpha=0.95, **kw)


def barra_color(fig, ax, mappable, etiqueta):
    cb = fig.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(etiqueta, color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(BORDER)
    return cb


def _residuos(df, x, y):
    m, b, _, _ = regresion(df[x].values, df[y].values)
    ajust = m * df[x].values + b
    return ajust, df[y].values - ajust


def _etiquetas_grupo(ax, nombres):
    ax.set_xticks(range(1, len(nombres) + 1))
    ax.set_xticklabels(nombres, rotation=35 if len(nombres) > 6 else 0,
                       ha="right" if len(nombres) > 6 else "center")


# ---- 1. Dispersión + regresión
def g_dispersion(ax, df, x, y, gcol, agrupar):
    lineas = []
    grupos = list(df.groupby(gcol)) if (agrupar and gcol) else [("Todos", df)]
    for i, (nombre, g) in enumerate(grupos):
        if len(g) < 3 or g[x].nunique() < 2:
            continue
        col = COLORES[i % len(COLORES)]
        m, b, r2, p = regresion(g[x].values, g[y].values)
        ax.scatter(g[x], g[y], s=28, color=col, alpha=0.6, edgecolors=PANEL, linewidths=0.5, zorder=3)
        xs = np.linspace(g[x].min(), g[x].max(), 100)
        if len(grupos) == 1:  # banda de confianza 95 %
            n = len(g)
            res = g[y].values - (m * g[x].values + b)
            s = np.sqrt((res ** 2).sum() / (n - 2))
            xm = g[x].mean()
            sxx = ((g[x] - xm) ** 2).sum()
            se = s * np.sqrt(1 / n + (xs - xm) ** 2 / sxx)
            z = NormalDist().inv_cdf(0.975)
            ax.fill_between(xs, m * xs + b - z * se, m * xs + b + z * se, color=col, alpha=0.16,
                            zorder=2, label="IC 95 %")
        ax.plot(xs, m * xs + b, color=col, lw=2.3, zorder=5, label=f"{nombre}  R²={r2:.2f}")
        lineas.append(f"{str(nombre):<12} y = {m:.4f}·x + {b:.4f}   R²={r2:.3f}"
                      + (f"   p={p:.3g}" if p is not None else "") + f"   n={len(g)}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    titulo(ax, f"{y}  vs  {x}")
    leyenda(ax)
    return lineas


# ---- 2. Residuos vs ajustados
def g_residuos(ax, df, x, y):
    ajust, res = _residuos(df, x, y)
    ax.scatter(ajust, res, s=26, color=ACCENT, alpha=0.55, edgecolors=PANEL, linewidths=0.5, zorder=3)
    ax.axhline(0, color=TERRA, lw=2, ls="--", zorder=4)
    sd = res.std()
    for k in (-2, 2):
        ax.axhline(k * sd, color=MUTED, lw=1, ls=":", zorder=4)
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuos")
    titulo(ax, "Residuos vs valores ajustados")
    return [f"Residuos: media={res.mean():.4f}  desv={sd:.4f}  máx|e|={np.abs(res).max():.3f}"]


# ---- 3. Histogramas
def g_hist(ax, v, nombre, color):
    v = np.asarray(v, float)
    bins = int(min(40, max(8, np.sqrt(len(v)))))
    ax.hist(v, bins=bins, density=True, color=color, alpha=0.55, edgecolor=PANEL, zorder=3)
    xs, d = kde(v)
    ax.plot(xs, d, color=color, lw=2.2, zorder=4)
    ax.axvline(v.mean(), color=TEXT, lw=1.3, ls="--", zorder=5, label=f"media = {v.mean():.2f}")
    ax.set_xlabel(nombre)
    ax.set_ylabel("Densidad")
    titulo(ax, f"Distribución de {nombre}")
    leyenda(ax)


# ---- 4. Distribución de residuos + Q-Q
def g_resid_hist(ax, res):
    bins = int(min(40, max(8, np.sqrt(len(res)))))
    ax.hist(res, bins=bins, density=True, color=PLUM, alpha=0.5, edgecolor=PANEL, zorder=3)
    mu, sd = res.mean(), res.std() or 1.0
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=TERRA, lw=2.2, zorder=4, label="Normal teórica")
    ax.set_xlabel("Residuos")
    ax.set_ylabel("Densidad")
    titulo(ax, "Residuos")
    leyenda(ax)


def g_qq(ax, res):
    n = len(res)
    z = (np.sort(res) - res.mean()) / (res.std() or 1.0)
    nd = NormalDist()
    teo = np.array([nd.inv_cdf((i - 0.5) / n) for i in range(1, n + 1)])
    ax.scatter(teo, z, s=18, color=ACCENT, alpha=0.6, edgecolors=PANEL, linewidths=0.4, zorder=3)
    lim = [min(teo.min(), z.min()), max(teo.max(), z.max())]
    ax.plot(lim, lim, color=TERRA, lw=2, ls="--", zorder=4)
    ax.set_xlabel("Cuantiles teóricos (normal)")
    ax.set_ylabel("Cuantiles de los residuos")
    titulo(ax, "Gráfico Q-Q")


# ---- 5. Caja
def g_caja(ax, df, y, gcol):
    datos = grupos_de(df, gcol, y)
    bp = ax.boxplot([d for _, d in datos], patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color=TEXT, lw=2), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    rng = np.random.default_rng(0)
    for i, (caja, (_, d)) in enumerate(zip(bp["boxes"], datos)):
        col = COLORES[i % len(COLORES)]
        caja.set(facecolor=col, alpha=0.55, edgecolor=col)
        pts = d if len(d) <= 300 else rng.choice(d, 300, replace=False)
        ax.scatter(i + 1 + rng.normal(0, 0.06, len(pts)), pts, s=7, color=col, alpha=0.35, zorder=3)
    _etiquetas_grupo(ax, [n for n, _ in datos])
    ax.set_ylabel(y)
    titulo(ax, f"{y} por {gcol if gcol else 'grupo'}")
    return [stat_linea(n, d) for n, d in datos]


# ---- 6. Violín
def g_violin(ax, df, y, gcol):
    datos = [(n, d) for n, d in grupos_de(df, gcol, y) if len(d) >= 2]
    if not datos:
        raise ValueError("No hay suficientes datos por grupo para el violín")
    vp = ax.violinplot([d for _, d in datos], showmedians=True, showextrema=False, widths=0.8)
    for i, cuerpo in enumerate(vp["bodies"]):
        col = COLORES[i % len(COLORES)]
        cuerpo.set_facecolor(col)
        cuerpo.set_edgecolor(col)
        cuerpo.set_alpha(0.6)
    vp["cmedians"].set_color(TEXT)
    vp["cmedians"].set_linewidth(2)
    _etiquetas_grupo(ax, [n for n, _ in datos])
    ax.set_ylabel(y)
    titulo(ax, f"{y} por {gcol if gcol else 'grupo'}")
    return [stat_linea(n, d) for n, d in datos]


# ---- 7. Barras: media por grupo
def g_barras(ax, df, y, gcol):
    datos = sorted(grupos_de(df, gcol, y), key=lambda t: -t[1].mean())
    medias = np.array([d.mean() for _, d in datos])
    ee = np.array([d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0 for _, d in datos])
    pos = np.arange(len(datos))
    ax.bar(pos, medias, color=[COLORES[i % len(COLORES)] for i in range(len(datos))],
           alpha=0.8, edgecolor=PANEL, zorder=3)
    ax.errorbar(pos, medias, yerr=ee * 1.96, fmt="none", ecolor=TEXT, capsize=4, lw=1.4, zorder=4)
    for p, m in zip(pos, medias):
        ax.text(p, m * 0.5, f"{m:.1f}", ha="center", va="center", color=ON_ACCENT,
                fontsize=9, fontweight="bold", zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels([n for n, _ in datos], rotation=35 if len(datos) > 6 else 0,
                       ha="right" if len(datos) > 6 else "center")
    ax.set_ylabel(f"media de {y}")
    titulo(ax, f"Media de {y} por {gcol if gcol else 'grupo'}  (IC 95 %)")
    return [stat_linea(n, d) for n, d in datos]


# ---- 8. Tendencia: media de Y según X
def g_tendencia(ax, df, x, y):
    clave = df[x].values if df[x].nunique() <= 40 else pd.qcut(df[x], 20, duplicates="drop").values
    t = pd.DataFrame({"k": clave, "xv": df[x].values, "yv": df[y].values})
    g = t.groupby("k", observed=True).agg(xm=("xv", "mean"), ym=("yv", "mean"), ys=("yv", "std"),
                                          n=("yv", "size")).fillna(0).sort_values("xm")
    ax.fill_between(g["xm"], g["ym"] - g["ys"], g["ym"] + g["ys"], color=ACCENT, alpha=0.15,
                    zorder=2, label="± 1 desv. estándar")
    ax.plot(g["xm"], g["ym"], color=ACCENT, lw=2.4, marker="o", ms=6, mfc=PANEL, mew=2, zorder=4,
            label="media de Y")
    m, b, r2, _ = regresion(df[x].values, df[y].values)
    xs = np.array([g["xm"].min(), g["xm"].max()])
    ax.plot(xs, m * xs + b, color=TERRA, lw=1.8, ls="--", zorder=5, label=f"recta (R²={r2:.2f})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    titulo(ax, f"Tendencia de {y} según {x}")
    leyenda(ax)
    return [f"{len(g)} puntos de la tendencia · pendiente global = {m:.4f}"]


# ---- 9. Hexbin
def g_hex(fig, ax, df, x, y):
    hb = ax.hexbin(df[x], df[y], gridsize=25, cmap=CMAP, mincnt=1, linewidths=0.3, edgecolors=PANEL)
    barra_color(fig, ax, hb, "observaciones")
    m, b, _, _ = regresion(df[x].values, df[y].values)
    xs = np.array([df[x].min(), df[x].max()])
    ax.plot(xs, m * xs + b, color=TEXT, lw=2, ls="--", zorder=5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    titulo(ax, f"Densidad de observaciones: {y} vs {x}")


# ---- 10. Mapa de calor de correlaciones
def g_correlacion(fig, ax, df):
    num = df.select_dtypes("number")
    num = num.loc[:, num.std() > 0].iloc[:, :12]
    if num.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 columnas numéricas con variación")
    c = num.corr()
    im = ax.imshow(c.values, cmap=CMAP_DIV, vmin=-1, vmax=1)
    n = len(c)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(c.columns, rotation=40, ha="right")
    ax.set_yticklabels(c.columns)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    barra_color(fig, ax, im, "correlación de Pearson")
    titulo(ax, "Correlaciones entre variables numéricas")
    return [f"{n} variables · mayor |r| fuera de la diagonal = "
            f"{np.abs(c.values - np.eye(n)).max():.3f}"]


# ---- 11. Donut
def g_donut(ax, df, gcol):
    cuenta = df[gcol].astype(str).value_counts()
    if len(cuenta) > 8:
        cuenta = pd.concat([cuenta.iloc[:7], pd.Series({"Otros": cuenta.iloc[7:].sum()})])
    _, _, pct = ax.pie(cuenta.values, colors=COLORES[:len(cuenta)], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor=PANEL, linewidth=2),
                       autopct="%1.0f%%", pctdistance=0.79, textprops=dict(color=ON_ACCENT, fontsize=9))
    for t in pct:
        t.set_fontweight("bold")
    ax.text(0, 0.06, f"{len(df):,}", ha="center", va="center", color=TEXT, fontsize=20,
            fontweight="bold", fontfamily=FONT_TITULO)
    ax.text(0, -0.14, "observaciones", ha="center", va="center", color=MUTED, fontsize=9)
    ax.legend(cuenta.index, loc="center left", bbox_to_anchor=(1.0, 0.5), facecolor=FIELD,
              edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.set_aspect("equal")
    titulo(ax, f"Observaciones por {gcol}")
    return [f"{k:<12} {v:,}" for k, v in cuenta.items()]


# ---- 12-14. Gráficos con mes
def _mes_de(df, mcol):
    if not mcol:
        raise ValueError("Asigna la columna de Mes en «Columnas»")
    return mes_numerico(df[mcol])


def g_serie(ax, df, y, mcol, acol):
    if not acol:
        raise ValueError("Asigna las columnas de Año y Mes en «Columnas»")
    mes = _mes_de(df, mcol)
    t = pd.DataFrame({"t": df[acol].astype(float).values + (mes.values - 1) / 12, "y": df[y].values}).dropna()
    s = t.groupby("t")["y"].mean()
    if len(s) < 3:
        raise ValueError("Hay muy pocos meses distintos para la serie temporal")
    suav = s.rolling(12, min_periods=3, center=True).mean()
    ax.plot(s.index, s.values, color=ACCENT, lw=1.2, alpha=0.6, marker="o", ms=3, zorder=3,
            label="media mensual")
    ax.plot(suav.index, suav.values, color=TERRA, lw=2.8, zorder=5, label="media móvil 12 meses")
    m, b, r2, _ = regresion(s.index.values, s.values)
    xs = np.array([s.index.min(), s.index.max()])
    ax.plot(xs, m * xs + b, color=TEXT, lw=1.6, ls="--", zorder=4, label=f"tendencia (R²={r2:.2f})")
    ax.set_xlabel("Tiempo (año + mes)")
    ax.set_ylabel(y)
    titulo(ax, f"{y} a lo largo del tiempo (promedio mensual)")
    leyenda(ax)
    return [f"{len(s)} meses con datos · tendencia = {m:+.4f} por año · R²={r2:.3f}"]


def g_estacional(ax, df, y, mcol):
    mes = _mes_de(df, mcol)
    t = pd.DataFrame({"m": mes.values, "y": df[y].values}).dropna()
    g = t.groupby("m")["y"].agg(["mean", "std", "size"]).fillna(0)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=ACCENT, alpha=0.16,
                    zorder=2, label="± 1 desv. estándar")
    ax.plot(g.index, g["mean"], color=ACCENT, lw=2.6, marker="o", ms=8, mfc=PANEL, mew=2, zorder=4,
            label="media del mes")
    ax.axhline(t["y"].mean(), color=TERRA, lw=1.6, ls="--", zorder=3, label="media global")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MESES_ABREV)
    ax.set_xlim(0.6, 12.4)
    ax.set_xlabel("Mes")
    ax.set_ylabel(y)
    titulo(ax, f"Estacionalidad de {y}")
    leyenda(ax)
    return [f"{MESES_ABREV[int(m) - 1]:<12} n={int(r['size'])}  media={r['mean']:.3f}  desv={r['std']:.3f}"
            for m, r in g.iterrows()]


def g_heat_mes(fig, ax, df, y, mcol, acol):
    if not acol:
        raise ValueError("Asigna las columnas de Año y Mes en «Columnas»")
    mes = _mes_de(df, mcol)
    t = pd.DataFrame({"a": df[acol].values, "m": mes.values, "y": df[y].values}).dropna()
    pv = t.pivot_table(values="y", index="a", columns="m", aggfunc="mean").reindex(columns=range(1, 13))
    if len(pv) < 2:
        raise ValueError("Se necesitan al menos 2 años distintos para el mapa año × mes")
    im = ax.imshow(pv.values, aspect="auto", cmap=CMAP)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES_ABREV)
    paso = max(1, int(np.ceil(len(pv) / 15)))
    ax.set_yticks(range(0, len(pv), paso))
    ax.set_yticklabels([f"{a:g}" for a in pv.index[::paso]])
    vmin, vmax = np.nanmin(pv.values), np.nanmax(pv.values)
    if pv.size <= 130:
        for i in range(pv.shape[0]):
            for j in range(12):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                            color=ON_ACCENT if v > (vmin + vmax) / 2 else TEXT)
    for sp in ax.spines.values():
        sp.set_visible(False)
    barra_color(fig, ax, im, f"media de {y}")
    titulo(ax, f"{y}: promedio por año y mes")
    return [f"{len(pv)} años × 12 meses · mínimo={vmin:.2f} · máximo={vmax:.2f}"]


# =====================================================================
#  Despachador
# =====================================================================
def dibujar(fig, tipo, df, x, y, pcol, ncol, agrupar=False, mcol=None, acol=None):
    """Dibuja el gráfico elegido en `fig` y devuelve las líneas de detalle."""
    fig.clear()
    fig.set_facecolor(PANEL)
    pcol = pcol if pcol in df else None
    ncol = ncol if ncol in df else None
    mcol = mcol if mcol in df else None
    acol = acol if acol in df else None

    def nuevo(*pos, grid=True):
        ax = fig.add_subplot(*pos)
        estilo_ax(ax, grid)
        return ax

    lineas = []
    if tipo == TIPOS[0]:
        lineas = g_dispersion(nuevo(111), df, x, y, pcol, agrupar)
    elif tipo == TIPOS[1]:
        lineas = g_residuos(nuevo(111), df, x, y)
    elif tipo == TIPOS[2]:
        g_hist(nuevo(121), df[x].values, x, COLORES[0])
        g_hist(nuevo(122), df[y].values, y, COLORES[2])
        lineas = [stat_linea(x, df[x].values), stat_linea(y, df[y].values)]
    elif tipo == TIPOS[3]:
        _, res = _residuos(df, x, y)
        g_resid_hist(nuevo(121), res)
        g_qq(nuevo(122), res)
        lineas = [f"Residuos: media={res.mean():.4f}  desv={res.std():.4f}"]
    elif tipo == TIPOS[4]:
        lineas = g_caja(nuevo(111), df, y, pcol)
    elif tipo == TIPOS[5]:
        lineas = g_violin(nuevo(111), df, y, ncol)
    elif tipo == TIPOS[6]:
        lineas = g_barras(nuevo(111), df, y, pcol)
    elif tipo == TIPOS[7]:
        lineas = g_tendencia(nuevo(111), df, x, y)
    elif tipo == TIPOS[8]:
        ax = nuevo(111)
        g_hex(fig, ax, df, x, y)
    elif tipo == TIPOS[9]:
        lineas = g_correlacion(fig, nuevo(111, grid=False), df)
    elif tipo == TIPOS[10]:
        gcol = pcol or ncol
        if not gcol:
            raise ValueError("Asigna la columna de Provincia o Nivel para ver el donut")
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        lineas = g_donut(ax, df, gcol)
    elif tipo == TIPOS[11]:
        lineas = g_serie(nuevo(111), df, y, mcol, acol)
    elif tipo == TIPOS[12]:
        lineas = g_estacional(nuevo(111), df, y, mcol)
    elif tipo == TIPOS[13]:
        lineas = g_heat_mes(fig, nuevo(111, grid=False), df, y, mcol, acol)
    elif tipo == TIPOS[14]:
        g_dispersion(nuevo(221), df, x, y, pcol, False)
        g_hist(nuevo(222), df[y].values, y, COLORES[2])
        g_caja(nuevo(223), df, y, pcol)
        _, res = _residuos(df, x, y)
        ax = nuevo(224)
        ajust = _residuos(df, x, y)[0]
        ax.scatter(ajust, res, s=14, color=ACCENT, alpha=0.5, edgecolors="none")
        ax.axhline(0, color=TERRA, lw=1.8, ls="--")
        ax.set_xlabel("Ajustados")
        ax.set_ylabel("Residuos")
        titulo(ax, "Residuos")
        lineas = [stat_linea(y, df[y].values)]
    else:
        raise ValueError("Tipo de gráfico no reconocido")

    fig.tight_layout()
    return lineas


# =====================================================================
#  Aplicación
# =====================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EDU·REGRESS · Regresión lineal educativa")
        self.geometry("1360x880")
        self.configure(bg=BG)
        self.df = None
        self._job = None
        self.col_vars = {k: tk.StringVar() for k in ("nivel", "provincia", "edad", "anio", "mes")}
        self.x_var, self.y_var = tk.StringVar(), tk.StringVar()
        self.tipo_var = tk.StringVar(value=TIPOS[0])
        self.agrupar = tk.BooleanVar(value=False)
        self.edad_min, self.edad_max = tk.IntVar(value=0), tk.IntVar(value=99)
        self.mes_min, self.mes_max = tk.IntVar(value=1), tk.IntVar(value=12)
        self._estilos()
        self._ui()
        self.vacio("Carga un archivo o usa los datos de ejemplo")

    # ---------- Estilos ----------
    def _estilos(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=TEXT, font=(FONT, 10))
        s.configure("TFrame", background=PANEL)
        s.configure("TLabel", background=PANEL, foreground=TEXT)
        s.configure("TLabelframe", background=PANEL, bordercolor=BORDER, relief="solid")
        s.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT, font=(FONT, 9, "bold"))
        s.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                    arrowcolor=ACCENT, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
        s.configure("TSpinbox", fieldbackground=FIELD, foreground=TEXT, arrowcolor=ACCENT,
                    bordercolor=BORDER, background=FIELD)
        s.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        s.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", ACCENT)])
        s.configure("Cream.TButton", background=ACCENT, foreground=ON_ACCENT, font=(FONT, 10, "bold"),
                    borderwidth=0, padding=8)
        s.map("Cream.TButton", background=[("active", "#cf8138")])
        s.configure("Ghost.TButton", background=FIELD, foreground=ACCENT, bordercolor=ACCENT, padding=6)
        s.map("Ghost.TButton", background=[("active", "#f4e6c6")])
        s.configure("Vertical.TScrollbar", background=BORDER, troughcolor=PANEL, arrowcolor=ACCENT,
                    bordercolor=PANEL)
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)

    # ---------- Interfaz ----------
    def _ui(self):
        cab = tk.Frame(self, bg=BG)
        cab.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(cab, text="✦ EDU·REGRESS", bg=BG, fg=ACCENT, font=(FONT_TITULO, 21, "bold")).pack(side="left")
        tk.Label(cab, text="  análisis de regresión lineal · educación", bg=BG, fg=MUTED,
                 font=(FONT, 10)).pack(side="left", pady=(10, 0))
        self.estado = tk.Label(cab, text="", bg=BG, fg=TERRA, font=(FONT, 9))
        self.estado.pack(side="right")
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x", padx=16)

        cuerpo = tk.Frame(self, bg=BG)
        cuerpo.pack(fill="both", expand=True, padx=16, pady=12)

        # Panel izquierdo
        panel = tk.Frame(cuerpo, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, width=330)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)
        inner = ttk.Frame(panel, padding=10)
        inner.pack(fill="both", expand=True)

        f = ttk.LabelFrame(inner, text=" 01 · DATOS ", padding=8)
        f.pack(fill="x", pady=4)
        ttk.Button(f, text="⇪  Cargar CSV / Excel", style="Cream.TButton", command=self.cargar).pack(fill="x")
        ttk.Button(f, text="Datos de ejemplo", style="Ghost.TButton",
                   command=lambda: self.set_df(datos_ejemplo())).pack(fill="x", pady=(5, 0))
        self.lbl_archivo = ttk.Label(f, text="Sin datos", foreground=MUTED)
        self.lbl_archivo.pack(anchor="w", pady=(5, 0))

        f = ttk.LabelFrame(inner, text=" 02 · COLUMNAS ", padding=8)
        f.pack(fill="x", pady=4)
        self.combos = {}
        for i, (k, etq) in enumerate([("nivel", "Nivel"), ("provincia", "Provincia"), ("edad", "Edad"),
                                ("anio", "Año"), ("mes", "Mes")]):
            ttk.Label(f, text=etq).grid(row=i, column=0, sticky="w")
            c = ttk.Combobox(f, textvariable=self.col_vars[k], state="readonly", width=20)
            c.grid(row=i, column=1, pady=2, padx=6)
            c.bind("<<ComboboxSelected>>", lambda e: self.poblar_filtros())
            self.combos[k] = c

        f = ttk.LabelFrame(inner, text=" 03 · FILTROS ", padding=8)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text="Nivel (primario / secundario)", foreground=MUTED).pack(anchor="w")
        self.lb_nivel = self._listbox(f, 2)
        ttk.Label(f, text="Provincias", foreground=MUTED).pack(anchor="w")
        self.lb_prov = self._listbox(f, 5)
        for etq, vmin, vmax, lo, hi in (("Edad", self.edad_min, self.edad_max, 0, 99),
                                        ("Mes ", self.mes_min, self.mes_max, 1, 12)):
            r = ttk.Frame(f)
            r.pack(fill="x", pady=3)
            ttk.Label(r, text=etq, width=5).pack(side="left")
            for var in (vmin, vmax):
                sp = ttk.Spinbox(r, from_=lo, to=hi, textvariable=var, width=4, command=self.auto)
                sp.pack(side="left", padx=4)
                sp.bind("<KeyRelease>", lambda e: self.auto())
                if var is vmin:
                    ttk.Label(r, text="→").pack(side="left")

        f = ttk.LabelFrame(inner, text=" 04 · REGRESIÓN ", padding=8)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text="Variable X").grid(row=0, column=0, sticky="w")
        self.cb_x = ttk.Combobox(f, textvariable=self.x_var, state="readonly", width=20)
        self.cb_x.grid(row=0, column=1, pady=2, padx=6)
        ttk.Label(f, text="Variable Y").grid(row=1, column=0, sticky="w")
        self.cb_y = ttk.Combobox(f, textvariable=self.y_var, state="readonly", width=20)
        self.cb_y.grid(row=1, column=1, pady=2, padx=6)
        for cb in (self.cb_x, self.cb_y):
            cb.bind("<<ComboboxSelected>>", lambda e: self.auto())
        ttk.Checkbutton(f, text="Una recta por provincia", variable=self.agrupar,
                        command=self.auto).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        r = ttk.Frame(inner)
        r.pack(fill="x", pady=(10, 0))
        ttk.Button(r, text="💾 Gráfico", style="Ghost.TButton", command=self.guardar_png).pack(
            side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(r, text="📄 Datos", style="Ghost.TButton", command=self.exportar_csv).pack(
            side="left", expand=True, fill="x", padx=(3, 0))

        # Zona derecha: tarjetas + selector + gráfico + detalle
        der = tk.Frame(cuerpo, bg=BG)
        der.pack(side="right", fill="both", expand=True, padx=(14, 0))

        tarjetas = tk.Frame(der, bg=BG)
        tarjetas.pack(fill="x")
        self.cards = {}
        for i, (k, t, col) in enumerate([("r2", "R²", ACCENT), ("m", "PENDIENTE", TERRA),
                                         ("b", "INTERCEPTO", PLUM), ("n", "OBSERVACIONES", OLIVE)]):
            c = tk.Frame(tarjetas, bg=PANEL, highlightbackground=col, highlightthickness=1)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            tarjetas.columnconfigure(i, weight=1)
            tk.Label(c, text=t, bg=PANEL, fg=MUTED, font=(FONT, 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
            v = tk.Label(c, text="—", bg=PANEL, fg=col, font=(FONT_TITULO, 20, "bold"))
            v.pack(anchor="w", padx=12, pady=(0, 8))
            self.cards[k] = v

        # Selector de tipo de gráfico
        barra = tk.Frame(der, bg=BG)
        barra.pack(fill="x", pady=(10, 0))
        tk.Label(barra, text="TIPO DE GRÁFICO", bg=BG, fg=ACCENT, font=(FONT, 9, "bold")).pack(side="left")
        self.cb_tipo = ttk.Combobox(barra, textvariable=self.tipo_var, values=TIPOS, state="readonly", width=42)
        self.cb_tipo.pack(side="left", padx=10)
        self.cb_tipo.bind("<<ComboboxSelected>>", lambda e: self.auto())
        for txt, paso in (("◀", -1), ("▶", 1)):
            ttk.Button(barra, text=txt, style="Ghost.TButton", width=3,
                       command=lambda p=paso: self.cambiar_tipo(p)).pack(side="left", padx=2)

        marco = tk.Frame(der, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        marco.pack(fill="both", expand=True, pady=10)
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=marco)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        self.detalle = tk.Label(der, text="", bg=BG, fg=MUTED, font=("Consolas", 9), justify="left", anchor="w")
        self.detalle.pack(fill="x")

    def _listbox(self, parent, h):
        fr = ttk.Frame(parent)
        fr.pack(fill="x", pady=(0, 6))
        lb = tk.Listbox(fr, selectmode="extended", height=h, exportselection=False, bg=FIELD, fg=TEXT,
                        selectbackground=ACCENT, selectforeground=ON_ACCENT, relief="flat",
                        highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
                        activestyle="none", font=(FONT, 10))
        sb = ttk.Scrollbar(fr, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        lb.bind("<<ListboxSelect>>", lambda e: self.auto())
        return lb

    def cambiar_tipo(self, paso):
        i = (TIPOS.index(self.tipo_var.get()) + paso) % len(TIPOS)
        self.tipo_var.set(TIPOS[i])
        self.auto()

    # ---------- Datos ----------
    def cargar(self):
        ruta = filedialog.askopenfilename(filetypes=[("Datos", "*.csv *.xlsx *.xls *.txt"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            if ruta.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(ruta)
            else:
                df = pd.read_csv(ruta, sep=None, engine="python", encoding_errors="replace")
        except Exception as e:
            return messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")
        self.set_df(df, ruta.replace("\\", "/").split("/")[-1])

    def set_df(self, df, nombre="datos de ejemplo"):
        self.df = limpiar_numericas(df.copy())
        cols = [str(c) for c in self.df.columns]
        self.df.columns = cols
        self.lbl_archivo.config(text=f"{nombre} · {len(self.df)} filas", foreground=ACCENT)
        for k, c in self.combos.items():
            c["values"] = cols
            self.col_vars[k].set(detectar_columna(cols, k))
        nums = list(self.df.select_dtypes("number").columns)
        self.cb_x["values"] = self.cb_y["values"] = nums
        if len(nums) >= 2:
            self.x_var.set(next((c for c in nums if c.lower() in ("periodo", "anio", "año", "year")), nums[0]))
            self.y_var.set(next(c for c in reversed(nums) if c != self.x_var.get()))
        self.poblar_filtros()

    def poblar_filtros(self):
        if self.df is None:
            return
        for lb, k in ((self.lb_nivel, "nivel"), (self.lb_prov, "provincia")):
            lb.delete(0, "end")
            col = self.col_vars[k].get()
            if col in self.df:
                for v in sorted(self.df[col].dropna().unique(), key=str):
                    lb.insert("end", v)
                lb.select_set(0, "end")
        col = self.col_vars["edad"].get()
        if col in self.df and pd.api.types.is_numeric_dtype(self.df[col]) and self.df[col].notna().any():
            self.edad_min.set(int(self.df[col].min()))
            self.edad_max.set(int(self.df[col].max()))
        self.mes_min.set(1)
        self.mes_max.set(12)
        self.auto()

    def filtrar(self):
        df = self.df
        for lb, k in ((self.lb_nivel, "nivel"), (self.lb_prov, "provincia")):
            col = self.col_vars[k].get()
            if col in df and lb.size():
                sel = {str(lb.get(i)) for i in lb.curselection()}
                df = df[df[col].astype(str).isin(sel)]
        col = self.col_vars["edad"].get()
        if col in df and pd.api.types.is_numeric_dtype(df[col]):
            try:
                df = df[df[col].between(self.edad_min.get(), self.edad_max.get())]
            except tk.TclError:
                pass
        col = self.col_vars["mes"].get()
        if col in df and len(df):
            try:
                df = df[mes_numerico(df[col]).between(self.mes_min.get(), self.mes_max.get())]
            except (ValueError, tk.TclError):
                pass
        return df

    # ---------- Gráfico ----------
    def auto(self):
        """Actualiza el gráfico automáticamente (con pequeño retardo para no saturar)."""
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(80, self.calcular)

    def vacio(self, msg):
        self.fig.clear()
        self.fig.set_facecolor(PANEL)
        ax = self.fig.add_subplot(111)
        estilo_ax(ax, grid=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=13, transform=ax.transAxes)
        self.canvas.draw()
        for v in self.cards.values():
            v.config(text="—")
        self.detalle.config(text="")

    def calcular(self):
        self._job = None
        if self.df is None:
            return self.vacio("Carga un archivo o usa los datos de ejemplo")
        x, y = self.x_var.get(), self.y_var.get()
        if x not in self.df or y not in self.df:
            return self.vacio("Elige las variables X e Y")
        try:
            df = self.filtrar().dropna(subset=[x, y])
            if len(df) < 3 or df[x].nunique() < 2:
                return self.vacio("Con estos filtros hay muy pocos datos (mín. 3)")

            lineas = dibujar(self.fig, self.tipo_var.get(), df, x, y,
                             self.col_vars["provincia"].get(), self.col_vars["nivel"].get(),
                             self.agrupar.get(), self.col_vars["mes"].get(), self.col_vars["anio"].get())
            self.canvas.draw()

            m, b, r2, p = regresion(df[x].values, df[y].values)   # tarjetas: ajuste global
            self.cards["r2"].config(text=f"{r2:.3f}")
            self.cards["m"].config(text=f"{m:.4g}")
            self.cards["b"].config(text=f"{b:.4g}")
            self.cards["n"].config(text=f"{len(df):,}")
            self.detalle.config(text="\n".join(lineas[:8]))
            self.estado.config(text="● actualizado")
        except ValueError as e:
            self.vacio(str(e))
        except Exception as e:
            self.vacio(f"Error al calcular:\n{e}")

    # ---------- Exportar ----------
    def guardar_png(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if ruta:
            self.fig.savefig(ruta, dpi=200, facecolor=self.fig.get_facecolor())

    def exportar_csv(self):
        if self.df is None:
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if ruta:
            self.filtrar().to_csv(ruta, index=False)


if __name__ == "__main__":
    App().mainloop()

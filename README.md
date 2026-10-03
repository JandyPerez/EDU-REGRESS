# EDU-REGRESS

Aplicación de escritorio en Python para analizar datos educativos con **regresión lineal**. Permite ver si una variable (por ejemplo, el gasto educativo) explica a otra (por ejemplo, la tasa de abandono), filtrar los datos y explorarlos con 15 tipos de gráfico, todo con una interfaz en tonos crema.

## Características

- Carga de archivos **CSV** y **Excel** (`.xlsx`), o uso de datos de ejemplo incluidos.
- Regresión lineal simple entre dos variables numéricas, con **R²**, **pendiente**, **intercepto** y **número de observaciones**.
- **Filtros** por nivel, provincia, rango de edad y rango de meses.
- Una recta de regresión por provincia (opcional).
- **15 tipos de gráfico**, que se actualizan automáticamente al cambiar cualquier opción.
- Tiene en cuenta el **mes**: filtro por meses, serie temporal, estacionalidad y mapa de calor año × mes.
- Exportación del gráfico a **PNG** y de los datos filtrados a **CSV**.

## Requisitos

- Python 3.9 o superior
- Tkinter (viene con Python en Windows y macOS; en Linux: `sudo apt install python3-tk`)

```bash
pip install pandas numpy matplotlib openpyxl scipy
```

`scipy` es opcional: si no está instalado, la app calcula la regresión con NumPy (solo se pierde el valor *p*).

## Uso

```bash
python regresion_educacion.py
```

1. Pulsa **Cargar CSV / Excel** o **Datos de ejemplo**.
2. En **Columnas**, indica cuáles son Nivel, Provincia, Edad, Año y Mes (la app intenta detectarlas por el nombre).
3. En **Filtros**, elige niveles, provincias, rango de edad y rango de meses.
4. En **Regresión**, elige la variable **X** y la variable **Y**.
5. Elige el **tipo de gráfico** sobre el área del gráfico (o recórrelos con ◀ ▶).
6. Guarda el gráfico (💾) o exporta los datos filtrados (📄).

## Tipos de gráfico

| # | Gráfico | Qué muestra |
|---|---|---|
| 1 | Dispersión + regresión | Puntos, recta y banda de confianza del 95 % |
| 2 | Residuos vs ajustados | Si el modelo deja patrones sin explicar |
| 3 | Histogramas de X e Y | Distribución de cada variable con curva de densidad |
| 4 | Distribución de residuos | Histograma de residuos y gráfico Q-Q |
| 5 | Caja por provincia | Mediana, cuartiles y dispersión por provincia |
| 6 | Violín por nivel | Forma de la distribución en primaria y secundaria |
| 7 | Barras por provincia | Media de Y con intervalo de confianza del 95 % |
| 8 | Tendencia | Media de Y según X, con banda de ± 1 desviación |
| 9 | Densidad hexagonal | Dónde se concentran las observaciones |
| 10 | Mapa de calor de correlaciones | Correlación entre todas las variables numéricas |
| 11 | Donut de observaciones | Reparto de filas por provincia o nivel |
| 12 | Serie temporal mensual | Evolución de Y en el tiempo, con media móvil de 12 meses |
| 13 | Estacionalidad | Media de Y por mes del año |
| 14 | Mapa de calor año × mes | Promedio de Y para cada combinación de año y mes |
| 15 | Panel resumen | Dispersión, histograma, caja y residuos en una sola vista |

Los gráficos 12 a 14 necesitan que las columnas **Año** y **Mes** estén asignadas. El mes puede ser un número (1–12) o un nombre (enero, Feb, mar…).

## Datos de ejemplo

El botón **Datos de ejemplo** genera **10.000 filas sintéticas** (también incluidas en `datos_ejemplo_10000.csv`):

| Columna | Descripción |
|---|---|
| `anio` | Año, de 2010 a 2023 |
| `mes` | Mes, de 1 a 12 |
| `periodo` | Año + mes en formato decimal (por ejemplo, 2015,25) |
| `provincia` | Norte, Sur, Este, Oeste, Centro o Costa |
| `nivel` | Primaria o Secundaria |
| `edad` | De 6 a 11 años (primaria) y de 12 a 17 (secundaria) |
| `tasa_escolarizacion` | Porcentaje de escolarización |
| `gasto_educativo_pib` | Gasto educativo como porcentaje del PIB |
| `tasa_abandono` | Porcentaje de abandono escolar |

Las relaciones están definidas a propósito: el gasto sube con el tiempo, la escolarización crece con el gasto, el abandono baja con él, y hay una pequeña estacionalidad en los meses de vacaciones. Con poco ruido, la regresión ajusta con un R² cercano a 0,9.

> ⚠️ Son datos inventados para probar la aplicación. No sirven para sacar conclusiones sobre educación real.

## Usar tus propios datos

- Cada fila es una observación y cada columna una variable.
- Las columnas numéricas con coma decimal (`3,5`) se convierten automáticamente.
- Para la regresión hacen falta al menos dos columnas numéricas.
- El separador del CSV se detecta solo.
- Los archivos `.xls` antiguos pueden requerir `pip install xlrd`.

## Estructura del código

Todo está en `regresion_educacion.py`:

- **Datos y estadística:** `datos_ejemplo()`, `limpiar_numericas()`, `mes_numerico()`, `regresion()`.
- **Gráficos:** funciones `g_*` independientes de la interfaz, más el despachador `dibujar(fig, tipo, df, x, y, ...)`.
- **Interfaz:** la clase `App` (Tkinter), con la paleta de colores definida al inicio del archivo.

### Añadir un gráfico nuevo

1. Escribe una función `g_mi_grafico(ax, df, x, y)` que dibuje en `ax` y devuelva una lista de líneas de texto con el detalle.
2. Añade su nombre a la lista `TIPOS`.
3. Añade una rama `elif` en `dibujar()` que la llame.

### Cambiar los colores

Edita las constantes del inicio del archivo: `BG`, `PANEL`, `FIELD`, `BORDER`, `TEXT`, `ACCENT`, `COLORES` y los mapas de color `CMAP` y `CMAP_DIV`.

## Limitaciones

- La regresión es **lineal simple** (una variable X y una Y); no incluye regresión múltiple.
- El modelo describe asociaciones, no demuestra causas.
- Los valores atípicos influyen en la recta y no se eliminan automáticamente.

## Licencia

MIT

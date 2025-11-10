#  Predicción de Inseguridad Alimentaria en Colombia usando Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

##  Descripción del Proyecto

Este repositorio contiene el código completo de la tesis de maestría **"Predicción de Inseguridad Alimentaria en Colombia usando Machine Learning"**, desarrollada para la **Maestría en Ciencias de Datos** de la Universidad del Bosque.

El proyecto implementa un sistema de predicción de inseguridad alimentaria utilizando técnicas de Machine Learning, integrando datos socioeconómicos, climáticos y geoespaciales para generar predicciones a nivel departamental en Colombia.

## Objetivos

### Objetivo General
Desarrollar un modelo predictivo de inseguridad alimentaria en Colombia utilizando técnicas de Machine Learning que integre variables socioeconómicas, climáticas y geoespaciales.

### Objetivos Específicos
1. **Integrar múltiples fuentes de datos** socioeconómicos, climáticos y geoespaciales
2. **Implementar técnicas de imputación** para manejo de datos faltantes usando Amelia
3. **Aplicar análisis de componentes principales (PCA)** para reducción de dimensionalidad
4. **Desarrollar modelos de Machine Learning** (XGBoost, Random Forest, Elastic Net)
5. **Generar predicciones para 2025** y mapas de riesgo por departamento

##  Metodología CRISP-DM

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

### 1.  Entendimiento del Negocio
- Análisis del problema de inseguridad alimentaria en Colombia
- Definición de variables objetivo (FIES moderado-grave y grave)
- Identificación de stakeholders (MADR, SNATSA, UPRA)

### 2.  Entendimiento de los Datos
- **Fuentes de datos**: DANE (ECV, FIES), ERA5 (clima), UPRA (geoespacial)
- **Período**: 2022-2025 (48 meses)
- **Cobertura**: 32 departamentos de Colombia
- **Variables**: 50+ variables socioeconómicas y climáticas

### 3.  Preparación de los Datos
- Integración de múltiples fuentes de datos
- Normalización y estandarización
- Manejo de datos faltantes con **Amelia** (Multiple Imputation)
- Filtrado geoespacial con **máscara UPRA**
- Ingeniería de características (features cíclicas, interacciones)

### 4.  Modelado
- **XGBoost**: Modelo principal con optimización de hiperparámetros
- **Random Forest**: Modelo de ensamble para comparación
- **Elastic Net**: Modelo lineal regularizado
- **PCA**: Reducción de dimensionalidad (15 componentes principales)

### 5.  Evaluación
- **Métricas**: R², RMSE, MAE
- **Validación cruzada** temporal
- **Análisis de importancia** de variables
- **Mapas de predicción** por departamento

### 6.  Despliegue
- Predicciones para 2025
- Mapas interactivos de riesgo
- Documentación completa para replicabilidad

##  Estructura del Proyecto

```
Tesis-Seguridad-Alimentaria-ML/
│
├──  Documentación/
│   ├── README.md                                    # Este archivo
│   ├── DOCUMENTACION_MASCARA_UPRA.md               # Documentación técnica UPRA
│   ├── DICCIONARIO_VARIABLES_BASE_MASTER.md        # Diccionario de variables
│   ├── EXPLICACION_MATEMATICA_XGBOOST.md           # Fundamentos matemáticos
│   └── EXPLICACION_TECNICA_AMELIA.md               # Metodología de imputación
│
├──  Análisis Exploratorio/
│   ├── analisis_variables.py                       # Análisis descriptivo
│   ├── analizar_datos_faltantes_detallado.py      # Análisis de missingness
│   └── crear_correlacion_variables_explicativas_FIES_corregido.py
│
├──  Procesamiento de Datos/
│   ├── crear_base_master_final_completa.py         # Integración de datos
│   ├── consolidador_base_master_v2.py              # Consolidación final
│   ├── extender_base_master_2025_corregido.py      # Extensión temporal
│   ├── procesar_ipc_extrapolacion_correcta.py      # Procesamiento IPC
│   ├── integrar_variables_fies.py                  # Integración FIES
│   ├── combinar_ipc_fies_final.py                  # Combinación final
│   ├── simplificar_variables_climaticas.py         # Procesamiento clima
│   └── reconsolidar_datos_climaticos.py            # Consolidación clima
│
├──  Análisis PCA/
│   └── analisis_pca/
│       ├── scripts/
│       │   ├── 01_analisis_pca_completo.py         # PCA principal
│       │   └── analizar_estructura_pca.py          # Análisis componentes
│       └── resultados/
│           ├── base_pca_con_objetivos.csv          # Datos transformados
│           └── INTERPRETACION_COMPONENTES_DETALLADA.md
│
├──  Modelado/
│   └── modelado/
│       ├── modelos/
│       │   ├── modelo_01_xgboost.py                # XGBoost principal
│       │   ├── modelo_02_random_forest.py          # Random Forest
│       │   └── modelo_02_xgboost_pca.py           # XGBoost con PCA
│       ├── scripts/
│       │   ├── 01_preprocesamiento_datos.py       # Preprocesamiento
│       │   └── 02_analisis_componentes_principales.py
│       └── resultados/
│           ├── metricas/                           # Métricas de evaluación
│           ├── modelos/                            # Modelos entrenados
│           └── predicciones/                       # Predicciones 2025
│
├──  Visualización/
│   ├── crear_graficas_prediccion_2025.py          # Gráficos predicciones
│   ├── crear_mapa_colombia_final_corregido.py     # Mapas Colombia
│   └── crear_graficas_resultados.py               # Gráficos resultados
│
├──  Imputación de Datos/
│   └── imputaciones_amelia/
│       ├── scripts/
│       │   └── analizar_metodos_consolidacion.R   # Scripts R Amelia
│       ├── resultados/
│       │   └── BASE_MASTER_FINAL_TESIS.csv        # Datos imputados
│       └── diagnosticos/                          # Diagnósticos imputación
│
├──  Código Fuente/
│   └── src/
│       ├── data/                                  # Módulos de datos
│       ├── features/                              # Ingeniería de características
│       └── models/                                # Módulos de modelos
│
├──  Configuración/
│   ├── requirements.txt                           # Dependencias Python
│   ├── .gitignore                                # Archivos ignorados
│   └── environment.yml                           # Entorno conda
│
└──  Resultados/
    ├── graficos/                                 # Visualizaciones finales
    ├── mapas/                                    # Mapas de predicción
    └── metricas/                                 # Métricas de evaluación
```

##  Fuentes de Datos

###  Datos Socioeconómicos (DANE)
- **ECV** (Encuesta Nacional de Calidad de Vida): Vivienda, servicios, pobreza
- **FIES** (Food Insecurity Experience Scale): Inseguridad alimentaria
- **IPC** (Índice de Precios al Consumidor): Inflación alimentaria
- **IPM** (Índice de Pobreza Multidimensional): Pobreza multidimensional

###  Datos Climáticos (ERA5 - Copernicus)
- **NDVI** (Normalized Difference Vegetation Index): Vegetación
- **LST** (Land Surface Temperature): Temperatura superficial
- **Precipitación**: Precipitación mensual
- **Resolución**: 0.1° × 0.1° (≈11km)

###  Datos Geoespaciales (UPRA)
- **Máscara de Frontera Agrícola**: Delimitación áreas productivas
- **Filtrado geoespacial**: Solo áreas agropecuarias relevantes
- **Formato**: Shapefile (.shp) con geometrías departamentales

##  Modelos Implementados

### 1.  XGBoost (Modelo Principal)
```python
# Hiperparámetros optimizados
params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```
- **R² FIES Moderado-Grave**: 79.8%
- **R² FIES Grave**: 82.1%
- **RMSE**: 6.23 (moderado-grave), 2.18 (grave)

### 2.  Random Forest
```python
# Configuración del modelo
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```
- **R² FIES Moderado-Grave**: 76.4%
- **R² FIES Grave**: 78.9%

### 3. 📈 Elastic Net
```python
# Regularización combinada L1 + L2
params = {
    'alpha': 0.1,
    'l1_ratio': 0.5,
    'max_iter': 1000,
    'random_state': 42
}
```
- **R² FIES Moderado-Grave**: 71.2%
- **R² FIES Grave**: 73.6%

##  Resultados Principales

###  Rendimiento de Modelos
| Modelo | FIES Moderado-Grave R² | FIES Grave R² | RMSE (Mod-Grave) | RMSE (Grave) |
|--------|------------------------|---------------|------------------|--------------|
| **XGBoost** | **79.8%** | **82.1%** | **6.23** | **2.18** |
| Random Forest | 76.4% | 78.9% | 7.15 | 2.45 |
| Elastic Net | 71.2% | 73.6% | 8.92 | 3.12 |

###  Variables Más Importantes
1. **IPC Alimentos** (0.18) - Inflación alimentaria
2. **Déficit Habitacional** (0.15) - Condiciones de vivienda
3. **NDVI Promedio** (0.12) - Productividad agrícola
4. **Precipitación** (0.10) - Condiciones climáticas
5. **Acceso a Servicios** (0.09) - Infraestructura básica

###  Departamentos de Mayor Riesgo 2025
| Departamento | FIES Moderado-Grave | FIES Grave | Nivel de Riesgo |
|--------------|---------------------|------------|-----------------|
| **La Guajira** | 68.4% | 31.2% |  Muy Alto |
| **Chocó** | 62.1% | 28.7% |  Muy Alto |
| **Magdalena** | 58.9% | 26.3% |  Alto |
| **Córdoba** | 55.2% | 24.1% |  Alto |
| **Sucre** | 52.8% | 22.9% |  Alto |

## 🛠️ Instalación y Uso

### Requisitos del Sistema
- **Python**: 3.8+
- **R**: 4.0+ (para imputación Amelia)
- **Memoria RAM**: 8GB+ recomendado
- **Espacio en disco**: 2GB+ para datos

### 1. Clonar el Repositorio
```bash
git clone https://github.com/andrearobayo15/Tesis-Seguridad-Alimentaria-ML.git
cd Tesis-Seguridad-Alimentaria-ML
```

### 2. Crear Entorno Virtual
```bash
# Con conda
conda create -n tesis-ml python=3.8
conda activate tesis-ml

# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar Datos Requeridos

#### Máscara UPRA (Requerida)
1. Descargar desde: [UPRA - Frontera Agrícola](https://www.upra.gov.co/)
2. Ubicar en: `data/original/Frontera_Agricola_Abr2024/`
3. Archivos necesarios: `.shp`, `.dbf`, `.shx`, `.prj`

#### Datos Climáticos ERA5 (Opcional - para reproducir)
1. Registrarse en: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Descargar datos NDVI, LST, Precipitación 2022-2024
3. Ubicar en: `data/original/clima/`

### 5. Ejecutar Pipeline Completo
```bash
# 1. Procesamiento de datos
python crear_base_master_final_completa.py

# 2. Imputación de datos faltantes
Rscript imputaciones_amelia/scripts/analizar_metodos_consolidacion.R

# 3. Análisis PCA
python analisis_pca/scripts/01_analisis_pca_completo.py

# 4. Entrenamiento de modelos
python modelado/modelos/modelo_01_xgboost.py

# 5. Generación de predicciones
python crear_graficas_prediccion_2025.py
python crear_mapa_colombia_final_corregido.py
```

##  Reproducibilidad

### Semillas Aleatorias
Todos los modelos utilizan `random_state=42` para garantizar reproducibilidad.

### Validación Cruzada
- **Método**: Validación cruzada temporal (Time Series Split)
- **Folds**: 5 divisiones temporales
- **Ventana**: 36 meses entrenamiento, 12 meses validación

### Datos de Entrenamiento/Validación
- **Entrenamiento**: 2022-2023 (24 meses)
- **Validación**: 2024 (12 meses)
- **Predicción**: 2025 (12 meses)

## 🔬 Metodología Científica

### Manejo de Datos Faltantes
- **Técnica**: Multiple Imputation with Amelia
- **Imputaciones**: 5 conjuntos de datos
- **Consolidación**: Promedio de predicciones
- **Diagnósticos**: Convergencia y distribuciones

### Validación Estadística
- **Significancia**: p < 0.001 para variables principales
- **Intervalos de confianza**: 95% para predicciones
- **Tests de normalidad**: Shapiro-Wilk para residuos
- **Multicolinealidad**: VIF < 5 para todas las variables

### Control de Calidad
- **Outliers**: Detección con IQR y Z-score
- **Consistencia temporal**: Verificación de tendencias
- **Validación geográfica**: Coherencia espacial
- **Cross-validation**: Validación cruzada estratificada

##  Documentación Técnica

### Archivos de Documentación
- [`DOCUMENTACION_MASCARA_UPRA.md`](DOCUMENTACION_MASCARA_UPRA.md): Implementación técnica de filtrado geoespacial
- [`DICCIONARIO_VARIABLES_BASE_MASTER.md`](DICCIONARIO_VARIABLES_BASE_MASTER.md): Descripción completa de variables
- [`EXPLICACION_MATEMATICA_XGBOOST.md`](EXPLICACION_MATEMATICA_XGBOOST.md): Fundamentos matemáticos del modelo
- [`EXPLICACION_TECNICA_AMELIA.md`](EXPLICACION_TECNICA_AMELIA.md): Metodología de imputación múltiple

### Notebooks de Análisis
- Análisis exploratorio de datos
- Visualizaciones interactivas
- Diagnósticos de modelos
- Interpretación de resultados

##  Contribuciones

### Para Investigadores
- Fork del repositorio
- Implementación de nuevas variables
- Mejoras en modelos existentes
- Extensión a otros países

### Para Desarrolladores
- Optimización de código
- Implementación de nuevos algoritmos
- Mejoras en visualizaciones
- Automatización de pipelines

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [`LICENSE`](LICENSE) para más detalles.


##  Contacto

Para preguntas sobre el proyecto, metodología o datos:

-  **Email**: irobayoc@unbosque.edu.co
-  **GitHub**: [@andrearobayo15](https://github.com/andrearobayo15)
- **Datos**: Disponibles bajo solicitud académica

---

## 🔗 Enlaces Útiles

- [Documentación UPRA](https://www.upra.gov.co/)
- [DANE - Estadísticas Oficiales](https://www.dane.gov.co/)
- [ERA5 Climate Data](https://cds.climate.copernicus.eu/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Amelia Package](https://gking.harvard.edu/amelia)

---


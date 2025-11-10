# Predicción de Inseguridad Alimentaria en Colombia usando Machine Learning

## 📋 Descripción del Proyecto

Este proyecto desarrolla modelos de Machine Learning para predecir la inseguridad alimentaria en los departamentos de Colombia utilizando factores sociales, económicos y climáticos. El estudio se basa en la Escala de Experiencia de Inseguridad Alimentaria (FIES) del DANE y abarca el período 2022-2025.

## 🎯 Objetivos

### Objetivo Principal
Construir un modelo basado en Machine Learning que permita predecir la inseguridad alimentaria tomando como base datos relacionados con factores de tipo sociales, económicos y climáticos en los departamentos de Colombia para los períodos comprendidos entre 2022 y 2025.

### Objetivos Específicos
1. Identificar los factores sociales, económicos y climáticos asociados a la inseguridad alimentaria en los departamentos de Colombia
2. Construir modelos de Machine Learning que integren dichos factores y permitan predecir escenarios de riesgo de inseguridad alimentaria a nivel territorial
3. Evaluar y comparar el desempeño de diferentes algoritmos de aprendizaje automático

## 📊 Variables del Estudio

### Variables Objetivo (FIES)
- **FIES_moderado_grave**: Inseguridad alimentaria moderada a grave
- **FIES_grave**: Inseguridad alimentaria grave

### Variables Explicativas
- **Sociales**: Índice de Pobreza Multidimensional (IPM), variables de Encuesta de Calidad de Vida (ECV)
- **Económicas**: Índice de Precios al Consumidor (IPC) de alimentos
- **Climáticas**: NDVI, precipitación, temperatura superficial (LST) vía Google Earth Engine

## 🔧 Metodología

### Modelos Implementados
1. **XGBoost** - Gradient Boosting optimizado
2. **Random Forest** - Ensamble de árboles de decisión
3. **Support Vector Machine (SVM)** - Máquinas de vectores de soporte
4. **Elastic Net** - Regresión regularizada (L1 + L2)
5. **Análisis de Componentes Principales (PCA)** - Reducción de dimensionalidad

### Proceso de Modelado
- **Metodología**: CRISP-DM (Cross Industry Standard Process for Data Mining)
- **Período de entrenamiento**: 2022-2024
- **Período de predicción**: 2025
- **Cobertura geográfica**: 32 departamentos de Colombia
- **Técnicas de validación**: Validación cruzada, métricas de regresión

## 📁 Estructura del Proyecto

```
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias de Python
├── .gitignore               # Archivos excluidos del repositorio
│
├── src/                     # Código fuente
│   ├── data_processing/     # Scripts de procesamiento de datos
│   ├── modeling/           # Modelos de Machine Learning
│   └── visualization/      # Scripts de visualización
│
├── data/                   # Datos del proyecto
│   ├── processed/         # Datos procesados
│   └── sample/           # Datos de ejemplo
│
├── models/                # Modelos entrenados
│
├── results/              # Resultados del análisis
│   ├── figures/         # Gráficos y visualizaciones
│   ├── tables/          # Tablas de resultados
│   └── reports/         # Reportes de análisis
│
├── notebooks/           # Jupyter Notebooks exploratorios
│
└── docs/               # Documentación adicional
    ├── methodology/    # Documentación metodológica
    └── analysis/      # Análisis detallados
```

## 📁 Datos Requeridos

### Máscara UPRA (Frontera Agrícola)
Este proyecto requiere la máscara de Frontera Agrícola de UPRA para filtrar áreas agropecuarias:

1. **Descargar desde**: [UPRA - Frontera Agrícola](https://www.upra.gov.co/uso-y-adecuacion-de-tierras/evaluaciones-de-tierras/zonificacion-de-tierras/evaluacion-de-tierras-para-la-agricultura-de-clima-calido-y-medio/5666)
2. **Ubicar en**: `data/original/Frontera_Agricola_Abr2024/`
3. **Archivos necesarios**: 
   - `Frontera_Agricola_Abr2024.shp`
   - `Frontera_Agricola_Abr2024.dbf`
   - `Frontera_Agricola_Abr2024.shx`
   - `Frontera_Agricola_Abr2024.prj`

**Nota**: Estos archivos no están incluidos en el repositorio debido a su gran tamaño (>750MB).

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.8+
- R 4.0+ (para imputación con Amelia)
- Git

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tesis-fies-ml.git
cd tesis-fies-ml

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Uso Básico
```python
# Ejemplo de uso de los modelos
from src.modeling.modelo_elastic_net import ElasticNetFIES

# Cargar y entrenar modelo
model = ElasticNetFIES()
model.train(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)
```

## 📈 Resultados Principales

### Mejor Modelo: Elastic Net
- **R² Score**: 79.88%
- **RMSE**: Optimizado para ambas variables FIES
- **Interpretabilidad**: Alta, con coeficientes explicables

### Hallazgos Clave
- **Departamento más crítico**: La Guajira (54.1% inseguridad alimentaria moderada-grave)
- **Población en riesgo 2025**: Aproximadamente 9.2 millones de personas
- **Variables más predictivas**: IPM, variables climáticas, indicadores de pobreza

## 📊 Visualizaciones

El proyecto incluye múltiples visualizaciones:
- Mapas de Colombia con predicciones por departamento
- Análisis de correlaciones entre variables
- Evolución temporal de la inseguridad alimentaria
- Comparación de desempeño de modelos

## 🔬 Metodología Científica

### Tratamiento de Datos Faltantes
- **Método**: Multiple Imputation usando Amelia (R)
- **Variables imputadas**: FIES 2022 (8 variables detalladas)
- **Validación**: Diagnósticos de convergencia y calidad

### Validación de Modelos
- Validación cruzada temporal
- Métricas de regresión (R², RMSE, MAE)
- Análisis de residuos
- Pruebas de significancia estadística

## 📚 Referencias Académicas

Basado en la metodología de:
- Martini et al. (2022) Nature Food - Modelos predictivos de seguridad alimentaria
- DANE (2023) - Escala FIES Colombia
- FAO (2021) - Metodologías de seguridad alimentaria

## 🤝 Contribuciones

Este proyecto es parte de una tesis de maestría. Para contribuciones o colaboraciones académicas, por favor contactar al autor.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Ivonne Andrea Robayo Cante**
- Maestría en Ciencias de Datos
- Universidad del Bosque
- Email: [tu-email@ejemplo.com]
- LinkedIn: [tu-perfil-linkedin]

## 🙏 Agradecimientos

- Universidad del Bosque - Programa de Maestría en Ciencias de Datos
- DANE - Por proporcionar los datos de FIES
- Google Earth Engine - Por los datos climáticos
- Comunidad de código abierto de Python y R

---

*Este proyecto contribuye al entendimiento de la inseguridad alimentaria en Colombia mediante técnicas avanzadas de Machine Learning, proporcionando herramientas predictivas para la toma de decisiones en políticas públicas.*

# ESTRATEGIA DE MODELADO - 5 MODELOS PARA PREDICCIÓN FIES

## OBJETIVO
Predecir las 8 variables FIES (Escala de Experiencia de Inseguridad Alimentaria) para 2022 usando datos de entrenamiento 2023-2024 y variables auxiliares socioeconómicas y climáticas.

## VARIABLES OBJETIVO (8 VARIABLES FIES)
1. **FIES_preocupacion_alimentos** - Preocupación por no tener suficientes alimentos
2. **FIES_no_alimentos_saludables** - No poder comer alimentos saludables y nutritivos
3. **FIES_poca_variedad_alimentos** - Comer solo algunos tipos de alimentos
4. **FIES_saltar_comida** - Tener que saltar una comida
5. **FIES_comio_menos** - Comer menos de lo que pensaba que debía
6. **FIES_sin_alimentos** - Quedarse sin alimentos
7. **FIES_hambre_sin_comer** - Sentir hambre pero no comer
8. **FIES_no_comio_dia_entero** - No comer en todo un día

## VARIABLES PREDICTORAS DISPONIBLES

### **Variables Socioeconómicas (IPM)**
- Analfabetismo, Bajo_logro_educativo, Barreras_acceso_salud
- Desempleo_larga_duracion, Inasistencia_escolar, Rezago_escolar
- Sin_aseguramiento_salud, Trabajo_informal, IPM_Total

### **Variables de Calidad de Vida (ECV)**
- Vida_general, Salud, Seguridad, Trabajo_actividad, Tiempo_libre, Ingreso
- Variables de vivienda y servicios públicos
- Pobreza_monetaria, IPC_Total

### **Variables Climáticas**
- precipitacion_promedio, temperatura_promedio, ndvi_promedio

### **Variables Temporales**
- departamento, año, mes, fecha

## ESTRATEGIA DE 5 MODELOS

### **MODELO 1: REGRESIÓN LINEAL MÚLTIPLE (MLR)**
**Características:**
- Modelo base interpretable y robusto
- Relaciones lineales entre predictores y variables FIES
- Fácil interpretación de coeficientes

**Funcionamiento:**
```
FIES_variable = β₀ + β₁*IPM_Total + β₂*Pobreza_monetaria + β₃*IPC_Total + 
                β₄*precipitacion + β₅*temperatura + β₆*ndvi + 
                β₇*departamento_dummies + β₈*mes_dummies + ε
```

**Ventajas:**
- Interpretabilidad alta
- Baseline para comparación
- Identificación de variables más importantes

### **MODELO 2: RANDOM FOREST (RF)**
**Características:**
- Ensemble de árboles de decisión
- Captura interacciones no lineales
- Robusto a outliers y multicolinealidad

**Funcionamiento:**
- Construye múltiples árboles con muestras bootstrap
- Cada árbol usa subconjunto aleatorio de variables
- Predicción final = promedio de todos los árboles
- Importancia de variables automática

**Ventajas:**
- Maneja relaciones complejas
- No requiere normalización
- Proporciona importancia de variables

### **MODELO 3: GRADIENT BOOSTING (XGBoost)**
**Características:**
- Ensemble secuencial de árboles débiles
- Optimización de gradiente para minimizar error
- Regularización L1 y L2 incorporada

**Funcionamiento:**
- Entrena árboles secuencialmente
- Cada árbol corrige errores del anterior
- Función objetivo: Loss + Regularización
- Hiperparámetros: learning_rate, max_depth, n_estimators

**Ventajas:**
- Alta precisión predictiva
- Maneja missing values
- Control de overfitting

### **MODELO 4: SUPPORT VECTOR REGRESSION (SVR)**
**Características:**
- Mapeo a espacio de alta dimensión
- Kernel trick para relaciones no lineales
- Robusto a outliers con ε-insensitive loss

**Funcionamiento:**
- Encuentra hiperplano óptimo en espacio transformado
- Kernels: RBF, polynomial, linear
- Parámetros: C (regularización), γ (kernel), ε (tolerancia)

**Ventajas:**
- Efectivo en alta dimensión
- Memoria eficiente
- Versátil con diferentes kernels

### **MODELO 5: REDES NEURONALES (MULTILAYER PERCEPTRON)**
**Características:**
- Arquitectura de capas densas
- Activaciones no lineales (ReLU, tanh)
- Backpropagation para entrenamiento

**Funcionamiento:**
```
Arquitectura propuesta:
Input Layer (n_features) → 
Hidden Layer 1 (128 neurons, ReLU) → 
Hidden Layer 2 (64 neurons, ReLU) → 
Hidden Layer 3 (32 neurons, ReLU) → 
Output Layer (1 neuron, linear)
```

**Ventajas:**
- Aproximación universal de funciones
- Captura patrones complejos
- Escalable a grandes datasets

## ESTRATEGIA DE IMPLEMENTACIÓN

### **FASE 1: PREPARACIÓN DE DATOS**
1. **División temporal:**
   - Entrenamiento: 2023-2024 (768 registros)
   - Predicción: 2022 (384 registros)

2. **Ingeniería de características:**
   - Variables dummy para departamentos
   - Variables estacionales (mes)
   - Interacciones importantes
   - Normalización para modelos que lo requieran

### **FASE 2: ENTRENAMIENTO Y VALIDACIÓN**
1. **Validación cruzada temporal:**
   - 2023 para entrenamiento
   - 2024 para validación
   - Métricas: RMSE, MAE, R²

2. **Optimización de hiperparámetros:**
   - Grid Search o Random Search
   - Validación cruzada k-fold
   - Early stopping para evitar overfitting

### **FASE 3: EVALUACIÓN Y COMPARACIÓN**
1. **Métricas de evaluación:**
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - R² (Coeficiente de determinación)
   - MAPE (Mean Absolute Percentage Error)

2. **Análisis de residuos:**
   - Distribución de errores
   - Patrones temporales
   - Análisis por departamento

### **FASE 4: ENSEMBLE Y PREDICCIÓN FINAL**
1. **Ensemble de modelos:**
   - Promedio ponderado por performance
   - Stacking con meta-learner
   - Voting regressor

2. **Predicción 2022:**
   - Aplicación a datos 2022
   - Intervalos de confianza
   - Análisis de incertidumbre

## CRITERIOS DE ÉXITO
1. **RMSE < 5.0** para variables FIES principales
2. **R² > 0.70** en validación cruzada
3. **Residuos sin patrones sistemáticos**
4. **Predicciones dentro de bounds [0-100]**
5. **Consistencia temporal y geográfica**

## PRÓXIMOS PASOS
1. Implementar pipeline de preprocesamiento
2. Desarrollar cada modelo individualmente
3. Comparar performance y seleccionar mejores
4. Crear ensemble final
5. Generar predicciones 2022 con intervalos de confianza

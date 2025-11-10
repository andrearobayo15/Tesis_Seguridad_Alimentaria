# ESPECIFICACIONES Y REQUISITOS PARA MODELADO FIES

## INSTRUCCIONES DEL PROFESOR

### **REQUISITOS ESPECÍFICOS MENCIONADOS:**
1. **Características del modelo y cómo funciona** - Cada modelo debe tener documentación detallada
2. **Trabajo en 5 modelos** - Implementar exactamente 5 modelos diferentes
3. **Paso a paso** - Desarrollo incremental y evaluación progresiva
4. **Para la escritura de tesis** - Documentación académica completa

### **PREGUNTAS PENDIENTES PARA ACLARAR:**
- [ ] ¿Qué métricas específicas requiere el profesor?
- [ ] ¿Qué nivel de interpretabilidad necesita cada modelo?
- [ ] ¿Requiere comparación con literatura específica (ej. Martini)?
- [ ] ¿Qué formato de reporte final necesita?
- [ ] ¿Requiere validación cruzada específica?

## ESTRUCTURA ORGANIZACIONAL PROPUESTA

### **DIRECTORIO DE TRABAJO:**
```
d:/Tesis maestria/Tesis codigo/modelado/
├── scripts/                          # Scripts de preprocesamiento
│   └── 01_preprocesamiento_datos.py
├── modelos/                          # Un script por modelo
│   ├── modelo_01_xgboost.py
│   ├── modelo_02_random_forest.py
│   ├── modelo_03_regresion_logistica.py
│   ├── modelo_04_svm.py
│   └── modelo_05_elastic_net.py
├── resultados/                       # Outputs y reportes
│   ├── metricas/
│   ├── predicciones/
│   ├── graficos/
│   └── reportes/
├── utils/                           # Funciones auxiliares
│   ├── evaluacion.py
│   ├── visualizacion.py
│   └── validacion.py
└── documentacion/                   # Documentación académica
    ├── metodologia_por_modelo.md
    ├── resultados_comparativos.md
    └── interpretacion_academica.md
```

### **PRINCIPIOS DE ORGANIZACIÓN:**
1. **Un script por modelo** - Evita confusiones y permite desarrollo independiente
2. **Modularidad** - Cada componente es independiente y reutilizable
3. **Trazabilidad** - Cada resultado puede rastrearse a su origen
4. **Documentación paralela** - Documentación académica junto con código
5. **Versionado** - Control de cambios en cada modelo

## ESPECIFICACIONES TÉCNICAS

### **DATOS DE ENTRADA:**
- **Archivo:** `BASE_MASTER_FINAL_TESIS.csv`
- **Entrenamiento:** 2022-2024 (1,152 registros)
- **Predicción:** 2025 (384 registros)
- **Variables objetivo:** 50 variables (IPM, ECV, FIES, climáticas, IPC)

### **VARIABLES PREDICTORAS:**
- Variables temporales: departamento, mes
- Variables socioeconómicas disponibles en 2022-2024
- Variables climáticas históricas
- Variables dummy generadas

### **MÉTRICAS DE EVALUACIÓN ESTÁNDAR:**
1. **RMSE** (Root Mean Square Error)
2. **MAE** (Mean Absolute Error)
3. **R²** (Coeficiente de determinación)
4. **MAPE** (Mean Absolute Percentage Error)
5. **Validación cruzada temporal**

## TEMPLATE ESTÁNDAR PARA CADA MODELO

### **ESTRUCTURA DE SCRIPT POR MODELO:**
```python
"""
MODELO X: [NOMBRE DEL MODELO]
========================

Descripción: [Descripción del modelo]
Características: [Características principales]
Funcionamiento: [Cómo funciona el modelo]
Ventajas: [Ventajas específicas]
Limitaciones: [Limitaciones conocidas]

Autor: Tesis Maestría
Fecha: Agosto 2025
Referencia: [Si aplica, ej. Martini para XGBoost]
"""

# 1. IMPORTACIONES Y CONFIGURACIÓN
# 2. CARGA DE DATOS PREPROCESADOS
# 3. CONFIGURACIÓN DEL MODELO
# 4. ENTRENAMIENTO
# 5. VALIDACIÓN CRUZADA
# 6. PREDICCIÓN 2025
# 7. EVALUACIÓN Y MÉTRICAS
# 8. VISUALIZACIONES
# 9. GUARDADO DE RESULTADOS
# 10. REPORTE ACADÉMICO
```

### **OUTPUTS ESTÁNDAR POR MODELO:**
1. **Modelo entrenado** (.pkl)
2. **Predicciones 2025** (.csv)
3. **Métricas de evaluación** (.json)
4. **Gráficos de diagnóstico** (.png)
5. **Reporte académico** (.md)
6. **Importancia de variables** (.csv)

## CRONOGRAMA DE DESARROLLO

### **FASE 1: PREPARACIÓN (Completada)**
- [x] Preprocesamiento de datos
- [x] Definición de estrategia
- [x] Estructura organizacional

### **FASE 2: DESARROLLO DE MODELOS**
**Semana 1:**
- [ ] Modelo 1: XGBoost (referencia Martini)
- [ ] Modelo 2: Random Forest

**Semana 2:**
- [ ] Modelo 3: Regresión Logística
- [ ] Modelo 4: SVM
- [ ] Modelo 5: Elastic Net

### **FASE 3: ANÁLISIS COMPARATIVO**
- [ ] Ensemble de modelos
- [ ] Análisis de resultados
- [ ] Documentación académica final

## CRITERIOS DE CALIDAD

### **CÓDIGO:**
- [ ] Documentación completa en cada script
- [ ] Manejo de errores y validaciones
- [ ] Reproducibilidad (random_state fijo)
- [ ] Eficiencia computacional
- [ ] Código limpio y comentado

### **RESULTADOS:**
- [ ] Métricas consistentes entre modelos
- [ ] Validación cruzada robusta
- [ ] Predicciones dentro de rangos válidos
- [ ] Interpretabilidad clara
- [ ] Comparabilidad académica

### **DOCUMENTACIÓN ACADÉMICA:**
- [ ] Descripción metodológica detallada
- [ ] Justificación de hiperparámetros
- [ ] Análisis de resultados
- [ ] Comparación con literatura
- [ ] Limitaciones y recomendaciones

## CHECKLIST ANTES DE COMENZAR

### **CONFIRMACIONES NECESARIAS:**
- [ ] ¿Los 5 modelos seleccionados son apropiados?
- [ ] ¿La estructura organizacional es adecuada?
- [ ] ¿Las métricas de evaluación son suficientes?
- [ ] ¿El template de script es completo?
- [ ] ¿Los outputs cubren necesidades académicas?

### **RECURSOS DISPONIBLES:**
- [ ] Datos preprocesados listos
- [ ] Librerías Python instaladas
- [ ] Espacio de almacenamiento suficiente
- [ ] Tiempo estimado para desarrollo

## PRÓXIMOS PASOS

1. **Revisar y aprobar** estas especificaciones
2. **Confirmar requisitos** adicionales del profesor
3. **Crear template base** para scripts de modelos
4. **Comenzar con Modelo 1: XGBoost**
5. **Desarrollo iterativo** con validación continua

---

**NOTA:** Este documento debe ser revisado y aprobado antes de comenzar el desarrollo para asegurar alineación con expectativas académicas y técnicas.

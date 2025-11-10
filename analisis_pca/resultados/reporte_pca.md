# REPORTE ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)

## RESUMEN EJECUTIVO
- **Variables originales**: 50
- **Componentes principales (80% varianza)**: 7
- **Componentes principales (90% varianza)**: 12
- **Reducción dimensional**: 86.0%
- **Correlaciones altas identificadas**: 64

## VARIANZA EXPLICADA POR COMPONENTE
- **PC1**: 0.419 (41.9%) - Acumulada: 0.419
- **PC2**: 0.131 (13.1%) - Acumulada: 0.550
- **PC3**: 0.086 (8.6%) - Acumulada: 0.636
- **PC4**: 0.065 (6.5%) - Acumulada: 0.701
- **PC5**: 0.047 (4.7%) - Acumulada: 0.748
- **PC6**: 0.036 (3.6%) - Acumulada: 0.784
- **PC7**: 0.026 (2.6%) - Acumulada: 0.810
- **PC8**: 0.023 (2.3%) - Acumulada: 0.833
- **PC9**: 0.021 (2.1%) - Acumulada: 0.855
- **PC10**: 0.018 (1.8%) - Acumulada: 0.872

## INTERPRETACIÓN DE COMPONENTES PRINCIPALES

### PC1
Variables más importantes:
- Pobreza_monetaria: 0.193
- No_pobres: -0.193
- Alcantarillado: -0.192
- En_arriendo_o_subarriendo: -0.190
- Deficit_habitacional: 0.188
- IPM_Total: 0.188
- Recoleccion_basura: -0.185
- No_alcanzan_gastos_minimos: 0.184

### PC2
Variables más importantes:
- FIES_no_alimentos_saludables: 0.283
- FIES_poca_variedad_alimentos: 0.235
- Inasistencia_escolar: -0.220
- Con_permiso_sin_pago: 0.218
- Propiedad_colectiva: -0.217
- Propia_totalmente_pagada: 0.214
- FIES_comio_menos: 0.212
- Energia: 0.209

### PC3
Variables más importantes:
- Salud: 0.318
- Tiempo_libre: 0.294
- Vida_general: 0.264
- Trabajo_actividad: 0.232
- Ingreso: 0.211
- Seguridad: 0.198
- FIES_no_comio_dia_entero: 0.193
- FIES_comio_menos: 0.189

### PC4
Variables más importantes:
- Sin_aseguramiento_salud: 0.317
- Salud: -0.295
- Propia_la_estan_pagando: 0.270
- Seguridad: -0.269
- Vida_general: -0.254
- Tiempo_libre: -0.250
- Telefono_fijo: 0.223
- Trabajo_informal: -0.200

### PC5
Variables más importantes:
- Deficit_cuantitativo: 0.358
- Deficit_cualitativo: -0.326
- Propiedad_colectiva: 0.320
- Barreras_acceso_salud: 0.286
- Con_permiso_sin_pago: -0.250
- Inasistencia_escolar: -0.206
- Posesion_sin_titulo: -0.195
- Analfabetismo: -0.195

### PC6
Variables más importantes:
- Barreras_acceso_salud: 0.475
- IPC_Total: 0.293
- Rezago_escolar: -0.254
- Desempleo_larga_duracion: -0.236
- FIES_preocupacion_alimentos: 0.216
- FIES_no_alimentos_saludables: 0.210
- Propia_totalmente_pagada: -0.204
- Posesion_sin_titulo: 0.203

### PC7
Variables más importantes:
- Desempleo_larga_duracion: 0.591
- ndvi_promedio: 0.360
- IPC_Total: 0.262
- Sin_aseguramiento_salud: -0.258
- temperatura_promedio: -0.258
- Rezago_escolar: -0.204
- Propia_totalmente_pagada: 0.200
- Seguridad: 0.199

## RECOMENDACIONES
1. **Usar 7 componentes principales** para capturar 80% de la varianza
2. **Reducción significativa** de dimensionalidad: 50 → 7 variables
3. **Solución al sobreajuste**: Menos parámetros a estimar en modelos ML
4. **Eliminación de multicolinealidad**: 64 correlaciones altas identificadas

## ARCHIVOS GENERADOS
- `datos_pca_transformados.csv`: Dataset con componentes principales
- `pca_varianza.png`: Visualizaciones de varianza explicada
- `reporte_pca.md`: Este reporte

## PRÓXIMOS PASOS
1. Re-entrenar XGBoost con componentes principales
2. Comparar performance con modelo original
3. Evaluar reducción de sobreajuste

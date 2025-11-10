"""
Análisis y Justificación de Parámetros XGBoost para Nuestro Caso Específico
Comparación entre configuración de Martini y configuración optimizada para PCA
"""

import pandas as pd
import numpy as np

def analizar_contexto_datos():
    """Analizar características específicas de nuestros datos"""
    print("=" * 70)
    print("ANÁLISIS DEL CONTEXTO DE NUESTROS DATOS")
    print("=" * 70)
    
    contexto = {
        'tamaño_muestra': {
            'total_registros': 1536,
            'entrenamiento': 1152,  # 2022-2024
            'test': 384,           # 2025
            'ratio_train_test': 1152/384
        },
        'dimensionalidad': {
            'variables_originales': 50,
            'componentes_pca': 7,
            'reduccion_dimensional': (50-7)/50 * 100,
            'varianza_explicada': 81.0
        },
        'targets': {
            'variables_objetivo': 2,  # FIES_moderado_grave, FIES_grave
            'completitud_datos': 75.0,  # 1152/1536
            'tipo_problema': 'Regresión multivariada'
        },
        'estructura_temporal': {
            'años_entrenamiento': 3,
            'departamentos': 32,
            'meses_por_año': 12,
            'estructura': 'Panel de datos'
        }
    }
    
    print("CARACTERÍSTICAS DE NUESTROS DATOS:")
    print(f"\n1. TAMAÑO DE MUESTRA:")
    print(f"   - Total registros: {contexto['tamaño_muestra']['total_registros']:,}")
    print(f"   - Entrenamiento: {contexto['tamaño_muestra']['entrenamiento']:,}")
    print(f"   - Test: {contexto['tamaño_muestra']['test']:,}")
    print(f"   - Ratio train/test: {contexto['tamaño_muestra']['ratio_train_test']:.1f}")
    
    print(f"\n2. DIMENSIONALIDAD:")
    print(f"   - Variables originales: {contexto['dimensionalidad']['variables_originales']}")
    print(f"   - Componentes PCA: {contexto['dimensionalidad']['componentes_pca']}")
    print(f"   - Reducción dimensional: {contexto['dimensionalidad']['reduccion_dimensional']:.1f}%")
    print(f"   - Varianza explicada: {contexto['dimensionalidad']['varianza_explicada']:.1f}%")
    
    print(f"\n3. VARIABLES OBJETIVO:")
    print(f"   - Número de targets: {contexto['targets']['variables_objetivo']}")
    print(f"   - Completitud: {contexto['targets']['completitud_datos']:.1f}%")
    print(f"   - Tipo: {contexto['targets']['tipo_problema']}")
    
    print(f"\n4. ESTRUCTURA TEMPORAL:")
    print(f"   - Años entrenamiento: {contexto['estructura_temporal']['años_entrenamiento']}")
    print(f"   - Departamentos: {contexto['estructura_temporal']['departamentos']}")
    print(f"   - Tipo: {contexto['estructura_temporal']['estructura']}")
    
    return contexto

def comparar_configuraciones():
    """Comparar configuración de Martini vs nuestra configuración"""
    print(f"\n" + "=" * 70)
    print("COMPARACIÓN DE CONFIGURACIONES XGBOOST")
    print("=" * 70)
    
    configuraciones = {
        'martini': {
            'objetivo': 'reg:logistic',
            'n_estimators': [100, 150, 200],  # Grid search
            'max_depth': [4, 5, 6],           # Grid search
            'learning_rate': [0.05, 0.1, 0.3], # Grid search
            'subsample': None,
            'colsample_bytree': None,
            'reg_alpha': None,
            'reg_lambda': None,
            'n_jobs': 2,
            'random_state': None,
            'bootstrap': 100,
            'cv_folds': 4
        },
        'nuestra_actual': {
            'objetivo': 'MultiOutputRegressor',
            'n_estimators': 100,              # Fijo
            'max_depth': 6,                   # Fijo
            'learning_rate': 0.1,             # Fijo
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'bootstrap': None,
            'cv_folds': None
        }
    }
    
    print("CONFIGURACIÓN DE MARTINI:")
    print("  - Objetivo: Regresión logística (valores 0-1)")
    print("  - Grid Search: 3 parámetros x 3 valores = 27 combinaciones")
    print("  - Bootstrap: 100 iteraciones para robustez")
    print("  - Validación: Temporal con 4 folds")
    print("  - Regularización: Solo implícita (sin L1/L2)")
    
    print(f"\nNUESTRA CONFIGURACIÓN ACTUAL:")
    print("  - Objetivo: Multi-output regression (2 targets simultáneos)")
    print("  - Parámetros fijos (sin grid search)")
    print("  - Regularización explícita (L1=0.1, L2=0.1)")
    print("  - Sampling: subsample=0.8, colsample=0.8")
    print("  - Reproducibilidad: random_state=42")
    
    return configuraciones

def justificar_parametros_optimos():
    """Justificar parámetros óptimos para nuestro caso específico"""
    print(f"\n" + "=" * 70)
    print("JUSTIFICACIÓN DE PARÁMETROS ÓPTIMOS PARA NUESTRO CASO")
    print("=" * 70)
    
    justificaciones = {
        'objetivo': {
            'recomendado': 'reg:squarederror',
            'justificacion': [
                "FIES son porcentajes continuos (0-100), no probabilidades (0-1)",
                "reg:logistic es para clasificación binaria o probabilidades",
                "reg:squarederror es estándar para regresión continua",
                "MultiOutputRegressor permite 2 targets simultáneos"
            ]
        },
        'n_estimators': {
            'recomendado': [50, 100, 150],
            'justificacion': [
                "Datos limitados (1,152 registros) requieren menos árboles",
                "PCA redujo dimensionalidad (7 features vs 50 originales)",
                "Grid search para encontrar óptimo específico",
                "Rango conservador para evitar overfitting"
            ]
        },
        'max_depth': {
            'recomendado': [3, 4, 5, 6],
            'justificacion': [
                "Datos PCA son menos complejos que originales",
                "7 features permiten árboles menos profundos",
                "Incluir rango de Martini (4-6) + opción conservadora (3)",
                "Evitar overfitting con muestra limitada"
            ]
        },
        'learning_rate': {
            'recomendado': [0.01, 0.05, 0.1, 0.2],
            'justificacion': [
                "Incluir rango de Martini (0.05-0.3)",
                "Agregar learning rate menor (0.01) para más estabilidad",
                "Datos PCA pueden beneficiarse de aprendizaje más gradual",
                "Grid search determinará óptimo"
            ]
        },
        'subsample': {
            'recomendado': [0.8, 0.9, 1.0],
            'justificacion': [
                "Datos limitados requieren usar la mayoría de observaciones",
                "0.8-0.9 introduce variabilidad sin perder mucha información",
                "1.0 como baseline (usar todos los datos)",
                "Ayuda a prevenir overfitting"
            ]
        },
        'colsample_bytree': {
            'recomendado': [0.8, 1.0],
            'justificacion': [
                "Solo 7 features disponibles (vs 50 originales)",
                "Usar la mayoría de features disponibles",
                "0.8 = usar 5-6 features por árbol",
                "1.0 = usar todas las 7 features"
            ]
        },
        'regularizacion': {
            'recomendado': {'reg_alpha': [0, 0.01, 0.1], 'reg_lambda': [0, 0.01, 0.1]},
            'justificacion': [
                "Datos limitados son propensos a overfitting",
                "PCA ya redujo multicolinealidad, pero regularización ayuda",
                "L1 (alpha) para selección de features",
                "L2 (lambda) para suavizar pesos",
                "Incluir 0 para comparar con enfoque de Martini"
            ]
        },
        'validacion': {
            'recomendado': 'TimeSeriesSplit con 5 folds',
            'justificacion': [
                "Datos tienen estructura temporal (2022-2024)",
                "TimeSeriesSplit respeta orden temporal",
                "5 folds para datos limitados (vs 4 de Martini)",
                "Validación más robusta que train/test simple"
            ]
        }
    }
    
    for param, info in justificaciones.items():
        print(f"\n{param.upper()}:")
        print(f"  Recomendado: {info['recomendado']}")
        print("  Justificación:")
        for just in info['justificacion']:
            print(f"    - {just}")
    
    return justificaciones

def generar_configuracion_final():
    """Generar configuración final justificada"""
    print(f"\n" + "=" * 70)
    print("CONFIGURACIÓN FINAL RECOMENDADA")
    print("=" * 70)
    
    config_final = {
        'modelo_base': {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        },
        'grid_search': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        },
        'validacion': {
            'cv_method': 'TimeSeriesSplit',
            'n_splits': 5,
            'scoring': ['r2', 'neg_mean_absolute_error'],
            'refit': 'r2'
        },
        'multioutput': {
            'wrapper': 'MultiOutputRegressor',
            'targets': ['FIES_moderado_grave', 'FIES_grave']
        }
    }
    
    print("CONFIGURACIÓN MODELO BASE:")
    for param, valor in config_final['modelo_base'].items():
        print(f"  {param}: {valor}")
    
    print(f"\nGRID SEARCH ({len(config_final['grid_search'])} parámetros):")
    total_combinaciones = 1
    for param, valores in config_final['grid_search'].items():
        print(f"  {param}: {valores}")
        total_combinaciones *= len(valores)
    
    print(f"\nTotal combinaciones: {total_combinaciones:,}")
    print(f"Con 5-fold CV: {total_combinaciones * 5:,} entrenamientos")
    
    print(f"\nVALIDACIÓN:")
    for param, valor in config_final['validacion'].items():
        print(f"  {param}: {valor}")
    
    return config_final

def comparar_con_literatura():
    """Comparar nuestra aproximación con literatura"""
    print(f"\n" + "=" * 70)
    print("COMPARACIÓN CON LITERATURA CIENTÍFICA")
    print("=" * 70)
    
    comparacion = {
        'martini_2022': {
            'contexto': 'Seguridad alimentaria global',
            'datos': 'Múltiples países, datos secundarios',
            'enfoque': 'Grid search conservador, bootstrap robusto',
            'ventajas': 'Validado en literatura, robusto',
            'limitaciones': 'No optimizado para PCA, reg:logistic inadecuado'
        },
        'nuestro_enfoque': {
            'contexto': 'Seguridad alimentaria Colombia (departamental)',
            'datos': 'Panel temporal, datos PCA',
            'enfoque': 'Grid search expandido, validación temporal',
            'ventajas': 'Optimizado para PCA, multi-target, regularización',
            'limitaciones': 'Más complejo computacionalmente'
        }
    }
    
    print("MARTINI ET AL. (2022):")
    for aspecto, descripcion in comparacion['martini_2022'].items():
        print(f"  {aspecto}: {descripcion}")
    
    print(f"\nNUESTRO ENFOQUE:")
    for aspecto, descripcion in comparacion['nuestro_enfoque'].items():
        print(f"  {aspecto}: {descripcion}")
    
    print(f"\nJUSTIFICACIÓN DE DIFERENCIAS:")
    print("  1. Objetivo diferente: reg:squarederror vs reg:logistic")
    print("     - FIES son porcentajes continuos, no probabilidades")
    print("  2. Grid search expandido:")
    print("     - Más parámetros para optimizar específicamente para PCA")
    print("  3. Regularización explícita:")
    print("     - Datos limitados requieren prevención de overfitting")
    print("  4. Multi-target:")
    print("     - Predicción simultánea de 2 variables FIES")
    print("  5. Validación temporal:")
    print("     - Respeta estructura de panel de datos")

def generar_codigo_implementacion():
    """Generar código de implementación con configuración justificada"""
    print(f"\n" + "=" * 70)
    print("GENERANDO CÓDIGO DE IMPLEMENTACIÓN")
    print("=" * 70)
    
    codigo = '''
# Configuración XGBoost Justificada para Datos PCA
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

def configurar_xgboost_pca():
    """Configuración XGBoost optimizada para datos PCA"""
    
    # Modelo base con parámetros justificados
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',  # Para regresión continua (FIES 0-100)
        random_state=42,               # Reproducibilidad
        n_jobs=-1,                     # Usar todos los cores
        verbosity=1                    # Información de progreso
    )
    
    # Grid search con parámetros justificados
    param_grid = {
        'estimator__n_estimators': [50, 100, 150],      # Conservador para datos limitados
        'estimator__max_depth': [3, 4, 5, 6],           # Incluye rango Martini + conservador
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Incluye rango Martini + gradual
        'estimator__subsample': [0.8, 0.9, 1.0],        # Mayoría de datos + baseline
        'estimator__colsample_bytree': [0.8, 1.0],       # Adaptado a 7 features PCA
        'estimator__reg_alpha': [0, 0.01, 0.1],          # L1: incluye baseline Martini
        'estimator__reg_lambda': [0, 0.01, 0.1]          # L2: incluye baseline Martini
    }
    
    # MultiOutput para 2 targets simultáneos
    modelo = MultiOutputRegressor(xgb_base, n_jobs=-1)
    
    # Validación temporal para datos de panel
    cv = TimeSeriesSplit(n_splits=5)
    
    # Grid search con métricas de Martini
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring=['r2', 'neg_mean_absolute_error'],
        refit='r2',                    # Optimizar por R² como Martini
        n_jobs=-1,
        verbose=2
    )
    
    return grid_search

# Justificación de cada parámetro:
# 1. objective='reg:squarederror': FIES son valores continuos 0-100, no probabilidades
# 2. n_estimators=[50,100,150]: Datos limitados requieren menos árboles
# 3. max_depth=[3,4,5,6]: PCA redujo complejidad, incluye rango Martini
# 4. learning_rate=[0.01-0.2]: Incluye rango Martini + aprendizaje gradual
# 5. subsample=[0.8,0.9,1.0]: Usar mayoría de datos limitados
# 6. colsample_bytree=[0.8,1.0]: Solo 7 features PCA disponibles
# 7. reg_alpha/lambda: Prevenir overfitting + comparar con Martini (0)
# 8. TimeSeriesSplit: Respeta estructura temporal de panel
# 9. MultiOutputRegressor: Predicción simultánea de 2 targets FIES
'''
    
    archivo_codigo = "d:/Tesis maestria/Tesis codigo/analisis_pca/scripts/configuracion_xgboost_justificada.py"
    with open(archivo_codigo, 'w', encoding='utf-8') as f:
        f.write(codigo)
    
    print(f"Código guardado: configuracion_xgboost_justificada.py")
    
    return codigo

def main():
    """Función principal de análisis"""
    print("ANÁLISIS Y JUSTIFICACIÓN DE PARÁMETROS XGBOOST")
    
    # 1. Analizar contexto de datos
    contexto = analizar_contexto_datos()
    
    # 2. Comparar configuraciones
    configuraciones = comparar_configuraciones()
    
    # 3. Justificar parámetros óptimos
    justificaciones = justificar_parametros_optimos()
    
    # 4. Generar configuración final
    config_final = generar_configuracion_final()
    
    # 5. Comparar con literatura
    comparar_con_literatura()
    
    # 6. Generar código
    codigo = generar_codigo_implementacion()
    
    print(f"\n" + "=" * 70)
    print("RESUMEN DE RECOMENDACIONES")
    print("=" * 70)
    print("1. Usar reg:squarederror (no reg:logistic como Martini)")
    print("2. Grid search expandido adaptado a datos PCA")
    print("3. Regularización explícita para datos limitados")
    print("4. Validación temporal con TimeSeriesSplit")
    print("5. MultiOutputRegressor para 2 targets simultáneos")
    print("6. Configuración justificada científicamente")
    
    return contexto, configuraciones, justificaciones, config_final

if __name__ == "__main__":
    contexto, configuraciones, justificaciones, config_final = main()

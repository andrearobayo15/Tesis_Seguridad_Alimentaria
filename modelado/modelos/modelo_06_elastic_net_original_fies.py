#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO 6: ELASTIC NET CON VARIABLES ORIGINALES PARA PREDICCIÓN FIES
===================================================================

Implementa Elastic Net usando variables originales (sin PCA) para predecir
específicamente las variables FIES (FIES_moderado_grave, FIES_grave).

Este modelo corrige la implementación anterior que predecía todas las variables
del dataset, para permitir comparación justa con el modelo PCA.

Características:
- Variables originales completas (climáticas, socioeconómicas, geográficas)
- Predicción específica de 2 variables FIES
- Validación cruzada temporal (TimeSeriesSplit)
- Grid search para optimización de hiperparámetros
- Escalado de features y targets

Autor: Sistema de Análisis ML
Fecha: 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports de scikit-learn
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def cargar_datos():
    """Carga y prepara los datos con variables originales"""
    
    print("1. Cargando datos originales...")
    
    # Cargar base master con variables originales
    data_path = Path("d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv")
    df = pd.read_csv(data_path)
    
    print(f"   Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    
    # Filtrar solo datos con FIES (2023-2024)
    df_fies = df[df['FIES_moderado_grave'].notna()].copy()
    print(f"   Registros con FIES: {len(df_fies)}")
    
    return df_fies

def preparar_features_originales(df):
    """Prepara features usando variables originales (sin PCA)"""
    
    print("2. Preparando features con variables originales...")
    
    # Variables a excluir (identificadores y targets)
    variables_excluir = [
        'departamento', 'año', 'mes', 'fecha', 'clave',
        'FIES_moderado_grave', 'FIES_grave'  # Variables objetivo
    ]
    
    # Seleccionar variables numéricas para features
    variables_numericas = [col for col in df.columns 
                          if col not in variables_excluir and 
                          df[col].dtype in ['int64', 'float64']]
    
    print(f"   Variables numéricas seleccionadas: {len(variables_numericas)}")
    
    # Usar datos ya imputados (no necesita imputación adicional)
    X_numeric_imputed = df[variables_numericas].copy()
    
    # Agregar variables categóricas codificadas
    # Departamento (one-hot encoding)
    dept_dummies = pd.get_dummies(df['departamento'], prefix='dept')
    
    # Mes (encoding cíclico)
    meses_map = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    df['mes_num'] = df['mes'].map(meses_map)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes_num'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes_num'] / 12)
    
    # Año normalizado
    df['año_norm'] = (df['año'] - df['año'].min()) / (df['año'].max() - df['año'].min())
    
    # Combinar todas las features
    X_final = pd.concat([
        X_numeric_imputed,
        dept_dummies,
        df[['mes_sin', 'mes_cos', 'año_norm']]
    ], axis=1)
    
    # Variables objetivo (FIES)
    y = df[['FIES_moderado_grave', 'FIES_grave']].copy()
    
    print(f"   Estructura final:")
    print(f"     X: {X_final.shape} (features)")
    print(f"     y: {y.shape} (targets)")
    print(f"     Variables numéricas: {len(variables_numericas)}")
    print(f"     Variables departamento: {len(dept_dummies.columns)}")
    print(f"     Variables temporales: 3")
    
    return X_final, y, variables_numericas

def entrenar_elastic_net_original(X, y):
    """Entrena modelo Elastic Net con variables originales"""
    
    print("3. Entrenando Elastic Net con variables originales...")
    
    # Escalado de features y targets
    print("   Escalando datos...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Configurar modelo con MultiOutputRegressor
    elastic_net = ElasticNet(random_state=42, max_iter=2000)
    modelo = MultiOutputRegressor(elastic_net)
    
    # Grid search para hiperparámetros
    print("   Optimizando hiperparámetros...")
    param_grid = {
        'estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'estimator__max_iter': [1000, 2000, 3000]
    }
    
    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        modelo, 
        param_grid, 
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y_scaled)
    
    print(f"   Optimización completada:")
    print(f"     Mejor score: {grid_search.best_score_:.4f}")
    print(f"     Mejores parámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"       {param}: {value}")
    
    return grid_search.best_estimator_, scaler_X, scaler_y, grid_search.best_params_

def evaluar_modelo_cv(modelo, X, y, scaler_X, scaler_y):
    """Evalúa modelo con validación cruzada temporal"""
    
    print("4. Evaluando con validación cruzada temporal...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"   Fold {fold}/5...")
        
        # División train/validation
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Escalado
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Entrenamiento
        modelo.fit(X_train_scaled, y_train_scaled)
        
        # Predicción
        y_pred_scaled = modelo.predict(X_val_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        print(f"     RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Estadísticas finales
    metricas_cv = {
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores)
    }
    
    print(f"\n   MÉTRICAS VALIDACIÓN CRUZADA TEMPORAL ELASTIC NET ORIGINAL:")
    print(f"     RMSE: {metricas_cv['RMSE_mean']:.4f} ± {metricas_cv['RMSE_std']:.4f}")
    print(f"     MAE:  {metricas_cv['MAE_mean']:.4f} ± {metricas_cv['MAE_std']:.4f}")
    print(f"     R²:   {metricas_cv['R2_mean']:.4f} ± {metricas_cv['R2_std']:.4f}")
    
    return metricas_cv

def entrenar_modelo_final(X, y, mejores_params, scaler_X, scaler_y):
    """Entrena modelo final con todos los datos"""
    
    print("5. Entrenando modelo final...")
    
    # Escalado completo
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Modelo final con mejores parámetros
    elastic_net = ElasticNet(**{k.replace('estimator__', ''): v for k, v in mejores_params.items()})
    modelo_final = MultiOutputRegressor(elastic_net)
    
    # Entrenamiento
    modelo_final.fit(X_scaled, y_scaled)
    
    # Métricas de entrenamiento
    y_pred_scaled = modelo_final.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    rmse_train = np.sqrt(mean_squared_error(y, y_pred))
    mae_train = mean_absolute_error(y, y_pred)
    r2_train = r2_score(y, y_pred)
    
    print(f"   Métricas entrenamiento final (referencia):")
    print(f"     RMSE: {rmse_train:.4f}")
    print(f"     MAE: {mae_train:.4f}")
    print(f"     R²: {r2_train:.4f}")
    
    metricas_train = {
        'RMSE': rmse_train,
        'MAE': mae_train,
        'R2': r2_train
    }
    
    return modelo_final, metricas_train

def generar_predicciones_2025(modelo, scaler_X, scaler_y, variables_numericas):
    """Genera predicciones para 2025"""
    
    print("6. Generando predicciones FIES 2025...")
    
    # Cargar datos completos para estructura 2025
    data_path = Path("d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv")
    df_completo = pd.read_csv(data_path)
    
    # Filtrar solo 2025
    df_2025 = df_completo[df_completo['año'] == 2025].copy()
    print(f"   Registros 2025: {len(df_2025)}")
    
    # Preparar features igual que en entrenamiento
    # Verificar qué variables están disponibles en 2025
    vars_disponibles = [var for var in variables_numericas if var in df_2025.columns]
    vars_faltantes = [var for var in variables_numericas if var not in df_2025.columns]
    
    print(f"   Variables disponibles en 2025: {len(vars_disponibles)}")
    print(f"   Variables faltantes en 2025: {len(vars_faltantes)}")
    
    # Crear DataFrame con variables disponibles (datos ya imputados)
    X_numeric = df_2025[vars_disponibles].copy()
    
    # Agregar variables faltantes con valor 0 (para variables que no existen en 2025)
    for var in vars_faltantes:
        X_numeric[var] = 0
    
    # Reordenar columnas para coincidir con entrenamiento
    X_numeric_final = X_numeric[variables_numericas]
    
    # Rellenar cualquier NaN restante con 0 (por seguridad)
    X_numeric_final = X_numeric_final.fillna(0)
    
    print(f"   Variables con datos: {len(vars_disponibles)}")
    print(f"   Variables agregadas como 0: {len(vars_faltantes)}")
    
    X_numeric_imputed = X_numeric_final
    
    # Variables categóricas
    dept_dummies = pd.get_dummies(df_2025['departamento'], prefix='dept')
    
    # Mes cíclico
    meses_map = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    df_2025['mes_num'] = df_2025['mes'].map(meses_map)
    df_2025['mes_sin'] = np.sin(2 * np.pi * df_2025['mes_num'] / 12)
    df_2025['mes_cos'] = np.cos(2 * np.pi * df_2025['mes_num'] / 12)
    df_2025['año_norm'] = 1.0  # 2025 normalizado
    
    # Combinar features
    X_2025 = pd.concat([
        X_numeric_imputed,
        dept_dummies,
        df_2025[['mes_sin', 'mes_cos', 'año_norm']]
    ], axis=1)
    
    # Asegurar mismas columnas que entrenamiento
    # (Agregar columnas faltantes con 0)
    columnas_entrenamiento = scaler_X.feature_names_in_
    for col in columnas_entrenamiento:
        if col not in X_2025.columns:
            X_2025[col] = 0
    
    X_2025 = X_2025[columnas_entrenamiento]
    
    # Predicción
    X_2025_scaled = scaler_X.transform(X_2025)
    y_pred_scaled = modelo.predict(X_2025_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Crear DataFrame de predicciones
    predicciones_df = pd.DataFrame({
        'departamento': df_2025['departamento'],
        'mes': df_2025['mes'],
        'FIES_moderado_grave': y_pred[:, 0],
        'FIES_grave': y_pred[:, 1]
    })
    
    print(f"   Predicciones generadas: {len(predicciones_df)} registros")
    print(f"   Variables predichas: FIES_moderado_grave, FIES_grave")
    print(f"   Features utilizados: {len(columnas_entrenamiento)}")
    print(f"   Rangos predichos:")
    print(f"     FIES_moderado_grave: {y_pred[:, 0].min():.2f} - {y_pred[:, 0].max():.2f}")
    print(f"     FIES_grave: {y_pred[:, 1].min():.2f} - {y_pred[:, 1].max():.2f}")
    
    return predicciones_df

def guardar_resultados(modelo, scaler_X, scaler_y, mejores_params, metricas_cv, metricas_train, predicciones, variables_numericas):
    """Guarda todos los resultados"""
    
    print("7. Guardando resultados Elastic Net Original...")
    
    # Crear directorios
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    base_path.mkdir(exist_ok=True)
    (base_path / "modelos").mkdir(exist_ok=True)
    (base_path / "predicciones").mkdir(exist_ok=True)
    (base_path / "metricas").mkdir(exist_ok=True)
    
    # Guardar modelo
    modelo_path = base_path / "modelos" / "elastic_net_original_fies_modelo.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump({
            'modelo': modelo,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'mejores_params': mejores_params,
            'variables_numericas': variables_numericas
        }, f)
    
    # Guardar predicciones
    pred_path = base_path / "predicciones" / "elastic_net_original_fies_predicciones_2025.csv"
    predicciones.to_csv(pred_path, index=False)
    
    # Guardar métricas
    metricas_completas = {
        "modelo": "ElasticNet_Original_FIES",
        "variables_originales": len(variables_numericas),
        "hiperparametros_optimizados": mejores_params,
        "entrenamiento": metricas_train,
        "validacion_cruzada": metricas_cv,
        "resumen": {
            "rmse_entrenamiento": metricas_train['RMSE'],
            "r2_entrenamiento": metricas_train['R2'],
            "rmse_cv": metricas_cv['RMSE_mean'],
            "r2_cv": metricas_cv['R2_mean'],
            "variables_predichas": 2,
            "registros_entrenamiento": len(predicciones) * 3,  # Aproximado 2023-2024
            "registros_prediccion": len(predicciones),
            "features_utilizados": len(scaler_X.feature_names_in_)
        },
        "fecha_ejecucion": datetime.now().isoformat()
    }
    
    metricas_path = base_path / "metricas" / "elastic_net_original_fies_metricas.json"
    with open(metricas_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
    
    print(f"   Modelo Elastic Net Original guardado: {modelo_path}")
    print(f"   Predicciones: {pred_path}")
    print(f"   Métricas: {metricas_path}")

def main():
    """Función principal"""
    
    print("MODELO 6: ELASTIC NET CON VARIABLES ORIGINALES PARA PREDICCIÓN FIES")
    print("=" * 75)
    print("INICIANDO PIPELINE COMPLETO - MODELO ELASTIC NET ORIGINAL FIES")
    print("=" * 75)
    
    # Pipeline completo
    df = cargar_datos()
    X, y, variables_numericas = preparar_features_originales(df)
    modelo, scaler_X, scaler_y, mejores_params = entrenar_elastic_net_original(X, y)
    metricas_cv = evaluar_modelo_cv(modelo, X, y, scaler_X, scaler_y)
    modelo_final, metricas_train = entrenar_modelo_final(X, y, mejores_params, scaler_X, scaler_y)
    predicciones = generar_predicciones_2025(modelo_final, scaler_X, scaler_y, variables_numericas)
    guardar_resultados(modelo_final, scaler_X, scaler_y, mejores_params, metricas_cv, metricas_train, predicciones, variables_numericas)
    
    print("\n" + "=" * 75)
    print("MODELO ELASTIC NET ORIGINAL FIES COMPLETADO EXITOSAMENTE")
    print("=" * 75)
    print("RESUMEN FINAL:")
    print(f"- Modelo: Elastic Net con variables originales")
    print(f"- Variables predichas: FIES_moderado_grave, FIES_grave")
    print(f"- Mejores parámetros: {mejores_params}")
    print(f"- RMSE entrenamiento: {metricas_train['RMSE']:.4f}")
    print(f"- R² entrenamiento: {metricas_train['R2']:.4f}")
    print(f"- RMSE validación cruzada: {metricas_cv['RMSE_mean']:.4f}")
    print(f"- R² validación cruzada: {metricas_cv['R2_mean']:.4f}")
    print(f"- Predicciones 2025: {len(predicciones)} registros")
    print(f"- Archivos generados: modelo Elastic Net original, predicciones, métricas")

if __name__ == "__main__":
    main()

"""
MODELO 2: XGBOOST CON COMPONENTES PCA PARA PREDICCIÓN FIES
==========================================================
Implementación de XGBoost usando 7 componentes principales para predecir
variables FIES_moderado_grave y FIES_grave con parámetros justificados
según análisis de Martini et al.

Autor: Análisis PCA - Tesis Maestría
Fecha: 2025-08-26
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from datetime import datetime

class ModeloXGBoostPCA:
    """Modelo XGBoost usando componentes PCA para predicción FIES"""
    
    def __init__(self):
        """Inicializar modelo XGBoost PCA"""
        self.df = None
        self.X_train = None
        self.y_train = None
        self.modelo = None
        self.predicciones_2025 = None
        self.metricas_cv = None
        self.metricas_entrenamiento = None
        
    def cargar_datos_pca(self):
        """Cargar datos con componentes PCA y variables objetivo FIES"""
        print("1. Cargando datos PCA...")
        
        # Cargar base con componentes PCA
        pca_path = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/base_pca_con_objetivos.csv"
        self.df = pd.read_csv(pca_path)
        
        print(f"   Datos cargados: {len(self.df)} registros, {len(self.df.columns)} columnas")
        print(f"   Componentes PCA: PC1-PC7")
        print(f"   Variables objetivo: FIES_moderado_grave, FIES_grave")
        print(f"   Período: 2022-2025")
        
        return self
    
    def preparar_datos_entrenamiento(self):
        """Preparar datos de entrenamiento usando componentes PCA"""
        print("\n2. Preparando datos de entrenamiento...")
        
        # Filtrar datos para entrenamiento (2022-2024) y predicción (2025)
        datos_entrenamiento = self.df[self.df['año'].isin([2022, 2023, 2024])].copy()
        
        # Features: 7 componentes PCA + encodings categóricos
        componentes_pca = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
        
        # Variables objetivo FIES
        variables_objetivo = ['FIES_moderado_grave', 'FIES_grave']
        
        # Crear encodings categóricos
        # Departamento (one-hot encoding)
        dept_dummies = pd.get_dummies(datos_entrenamiento['departamento'], prefix='dept')
        
        # Convertir mes a numérico si es string
        if datos_entrenamiento['mes'].dtype == 'object':
            meses_map = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
            }
            datos_entrenamiento['mes_num'] = datos_entrenamiento['mes'].map(meses_map)
        else:
            datos_entrenamiento['mes_num'] = datos_entrenamiento['mes']
        
        # Mes (encoding cíclico)
        datos_entrenamiento['mes_sin'] = np.sin(2 * np.pi * datos_entrenamiento['mes_num'] / 12)
        datos_entrenamiento['mes_cos'] = np.cos(2 * np.pi * datos_entrenamiento['mes_num'] / 12)
        
        # Año (normalizado)
        datos_entrenamiento['año_norm'] = (datos_entrenamiento['año'] - 2022) / 2
        
        # Combinar features
        self.X_train = pd.concat([
            datos_entrenamiento[componentes_pca],  # 7 componentes PCA
            dept_dummies,                          # 32 departamentos
            datos_entrenamiento[['mes_sin', 'mes_cos', 'año_norm']]  # 3 temporales
        ], axis=1)
        
        # Variables objetivo
        self.y_train = datos_entrenamiento[variables_objetivo].copy()
        
        # Manejar valores faltantes
        self.X_train = self.X_train.fillna(0)  # PCA no debería tener NaN
        self.y_train = self.y_train.fillna(self.y_train.median())
        
        print(f"   Estructura de entrenamiento:")
        print(f"     X_train: {self.X_train.shape} (7 PCA + {len(dept_dummies.columns)} dept + 3 temporal)")
        print(f"     y_train: {self.y_train.shape} (FIES_moderado_grave, FIES_grave)")
        print(f"     Registros: {len(datos_entrenamiento)} (2022-2024)")
        
        return self
    
    def configurar_modelo(self):
        """Configurar XGBoost con parámetros justificados para datos PCA"""
        print("\n3. Configurando modelo XGBoost para PCA...")
        
        # Configuración optimizada para componentes PCA (dimensionalidad reducida)
        xgb_params = {
            'objective': 'reg:squarederror',  # Para valores continuos FIES
            'n_estimators': 150,              # Más estimadores para PCA (menos overfitting)
            'max_depth': 5,                   # Rango Martini [3,4,5] - valor alto
            'learning_rate': 0.05,            # Rango Martini [0.05,0.1,0.2] - conservador
            'subsample': 0.9,                 # Usar más datos (menos features)
            'colsample_bytree': 1.0,          # Usar todas las features PCA
            'reg_alpha': 0.001,               # Regularización muy ligera
            'reg_lambda': 0.001,              # Regularización muy ligera
            'random_state': 42,               # Reproducibilidad
            'n_jobs': -1,                     # Usar todos los cores
            'verbosity': 1                    # Mostrar progreso
        }
        
        # Crear modelo base XGBoost
        xgb_base = xgb.XGBRegressor(**xgb_params)
        
        # Wrapper para múltiples outputs (FIES_moderado_grave, FIES_grave)
        self.modelo = MultiOutputRegressor(xgb_base, n_jobs=-1)
        
        print("   Configuración XGBoost PCA:")
        for param, valor in xgb_params.items():
            print(f"     {param}: {valor}")
        
        print("   Justificación para PCA:")
        print("     - Menos regularización (dimensionalidad reducida)")
        print("     - Más estimadores (menor riesgo overfitting)")
        print("     - colsample_bytree=1.0 (usar todos los componentes)")
        
        return self
    
    def entrenar_modelo(self):
        """Entrenar modelo XGBoost PCA con validación cruzada temporal"""
        print("\n4. Entrenando modelo XGBoost PCA...")
        
        # VALIDACIÓN CRUZADA TEMPORAL
        print("   Implementando validación cruzada temporal...")
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=5)  # 5 folds para más robustez
        cv_scores = {'RMSE': [], 'MAE': [], 'R2': []}
        
        fold = 1
        for train_idx, val_idx in tscv.split(self.X_train):
            print(f"   Fold {fold}/5...")
            
            # Split temporal
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # Entrenar en fold
            self.modelo.fit(X_fold_train, y_fold_train)
            
            # Predecir en validación
            y_pred_val = self.modelo.predict(X_fold_val)
            
            # Métricas por fold (promedio de FIES_moderado_grave y FIES_grave)
            fold_rmse = []
            fold_mae = []
            fold_r2 = []
            
            for i in range(len(self.y_train.columns)):
                rmse = np.sqrt(mean_squared_error(y_fold_val.iloc[:, i], y_pred_val[:, i]))
                mae = mean_absolute_error(y_fold_val.iloc[:, i], y_pred_val[:, i])
                r2 = r2_score(y_fold_val.iloc[:, i], y_pred_val[:, i])
                
                fold_rmse.append(rmse)
                fold_mae.append(mae)
                fold_r2.append(r2)
            
            # Promedios del fold
            cv_scores['RMSE'].append(np.mean(fold_rmse))
            cv_scores['MAE'].append(np.mean(fold_mae))
            cv_scores['R2'].append(np.mean(fold_r2))
            
            print(f"     RMSE: {np.mean(fold_rmse):.4f}, MAE: {np.mean(fold_mae):.4f}, R²: {np.mean(fold_r2):.4f}")
            fold += 1
        
        # Métricas de validación cruzada
        self.metricas_cv = {
            'RMSE_mean': np.mean(cv_scores['RMSE']),
            'RMSE_std': np.std(cv_scores['RMSE']),
            'MAE_mean': np.mean(cv_scores['MAE']),
            'MAE_std': np.std(cv_scores['MAE']),
            'R2_mean': np.mean(cv_scores['R2']),
            'R2_std': np.std(cv_scores['R2'])
        }
        
        print(f"\n   MÉTRICAS VALIDACIÓN CRUZADA TEMPORAL PCA:")
        print(f"     RMSE: {self.metricas_cv['RMSE_mean']:.4f} ± {self.metricas_cv['RMSE_std']:.4f}")
        print(f"     MAE:  {self.metricas_cv['MAE_mean']:.4f} ± {self.metricas_cv['MAE_std']:.4f}")
        print(f"     R²:   {self.metricas_cv['R2_mean']:.4f} ± {self.metricas_cv['R2_std']:.4f}")
        
        # Entrenar modelo final con todos los datos
        print("   Entrenando modelo final con todos los datos...")
        self.modelo.fit(self.X_train, self.y_train)
        
        # Métricas de entrenamiento (referencia)
        y_pred_train = self.modelo.predict(self.X_train)
        rmse_train = np.mean([np.sqrt(mean_squared_error(self.y_train.iloc[:, i], y_pred_train[:, i])) 
                             for i in range(len(self.y_train.columns))])
        mae_train = np.mean([mean_absolute_error(self.y_train.iloc[:, i], y_pred_train[:, i]) 
                            for i in range(len(self.y_train.columns))])
        r2_train = np.mean([r2_score(self.y_train.iloc[:, i], y_pred_train[:, i]) 
                           for i in range(len(self.y_train.columns))])
        
        # Guardar métricas de entrenamiento
        self.metricas_entrenamiento = {
            'RMSE': rmse_train,
            'MAE': mae_train,
            'R2': r2_train
        }
        
        print(f"   Métricas entrenamiento final (referencia):")
        print(f"     RMSE: {rmse_train:.4f}")
        print(f"     MAE: {mae_train:.4f}")
        print(f"     R²: {r2_train:.4f}")
        
        return self
    
    def predecir_2025(self):
        """Generar predicciones FIES para 2025 usando componentes PCA"""
        print("\n5. Generando predicciones FIES 2025 con PCA...")
        
        # Datos 2025
        datos_2025 = self.df[self.df['año'] == 2025].copy()
        
        # Features: componentes PCA + encodings categóricos
        componentes_pca = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
        
        # Encodings categóricos para 2025
        dept_dummies = pd.get_dummies(datos_2025['departamento'], prefix='dept')
        
        # Asegurar que tenemos todas las columnas de departamento
        for col in self.X_train.columns:
            if col.startswith('dept_') and col not in dept_dummies.columns:
                dept_dummies[col] = 0
        
        # Reordenar columnas para coincidir con entrenamiento
        dept_cols = [col for col in self.X_train.columns if col.startswith('dept_')]
        dept_dummies = dept_dummies.reindex(columns=dept_cols, fill_value=0)
        
        # Convertir mes a numérico para 2025
        if datos_2025['mes'].dtype == 'object':
            meses_map = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
            }
            datos_2025['mes_num'] = datos_2025['mes'].map(meses_map)
        else:
            datos_2025['mes_num'] = datos_2025['mes']
        
        # Mes y año
        datos_2025['mes_sin'] = np.sin(2 * np.pi * datos_2025['mes_num'] / 12)
        datos_2025['mes_cos'] = np.cos(2 * np.pi * datos_2025['mes_num'] / 12)
        datos_2025['año_norm'] = (datos_2025['año'] - 2022) / 2
        
        # Combinar features para predicción
        X_pred = pd.concat([
            datos_2025[componentes_pca],
            dept_dummies,
            datos_2025[['mes_sin', 'mes_cos', 'año_norm']]
        ], axis=1)
        
        # Asegurar mismo orden de columnas que entrenamiento
        X_pred = X_pred.reindex(columns=self.X_train.columns, fill_value=0)
        
        # Generar predicciones
        predicciones = self.modelo.predict(X_pred)
        
        # Crear DataFrame con predicciones
        self.predicciones_2025 = pd.DataFrame({
            'departamento': datos_2025['departamento'],
            'año': datos_2025['año'],
            'mes': datos_2025['mes'],
            'fecha': datos_2025['fecha'],
            'FIES_moderado_grave': predicciones[:, 0],
            'FIES_grave': predicciones[:, 1]
        })
        
        print(f"   Predicciones generadas: {len(self.predicciones_2025)} registros")
        print(f"   Variables predichas: FIES_moderado_grave, FIES_grave")
        print(f"   Componentes PCA utilizados: 7")
        print(f"   Features totales: {len(X_pred.columns)}")
        
        # Estadísticas de predicciones
        print(f"   Rangos predichos:")
        print(f"     FIES_moderado_grave: {self.predicciones_2025['FIES_moderado_grave'].min():.2f} - {self.predicciones_2025['FIES_moderado_grave'].max():.2f}")
        print(f"     FIES_grave: {self.predicciones_2025['FIES_grave'].min():.2f} - {self.predicciones_2025['FIES_grave'].max():.2f}")
        
        return self
    
    def guardar_resultados(self):
        """Guardar modelo, predicciones y métricas PCA"""
        print("\n6. Guardando resultados PCA...")
        
        # Crear directorios si no existen
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas', exist_ok=True)
        
        # 1. Guardar modelo
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos/xgboost_pca_modelo.pkl', 'wb') as f:
            pickle.dump(self.modelo, f)
        
        # 2. Guardar predicciones 2025
        self.predicciones_2025.to_csv(
            'd:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/xgboost_pca_predicciones_2025.csv', 
            index=False
        )
        
        # 3. Guardar métricas
        metricas_completas = {
            'modelo': 'XGBoost_PCA',
            'componentes_pca': 7,
            'entrenamiento': self.metricas_entrenamiento,
            'validacion_cruzada': self.metricas_cv,
            'resumen': {
                'rmse_entrenamiento': self.metricas_entrenamiento['RMSE'],
                'r2_entrenamiento': self.metricas_entrenamiento['R2'],
                'rmse_cv': self.metricas_cv['RMSE_mean'],
                'r2_cv': self.metricas_cv['R2_mean'],
                'variables_predichas': 2,  # FIES_moderado_grave, FIES_grave
                'registros_entrenamiento': len(self.X_train),
                'registros_prediccion': len(self.predicciones_2025),
                'features_utilizados': len(self.X_train.columns)
            },
            'fecha_ejecucion': datetime.now().isoformat()
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/xgboost_pca_metricas.json', 'w') as f:
            json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
        
        print("   Modelo PCA guardado: modelos/xgboost_pca_modelo.pkl")
        print("   Predicciones: predicciones/xgboost_pca_predicciones_2025.csv")
        print("   Métricas: metricas/xgboost_pca_metricas.json")
        
        return self
    
    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo del modelo XGBoost PCA"""
        return (self.cargar_datos_pca()
         .preparar_datos_entrenamiento()
         .configurar_modelo()
         .entrenar_modelo()
         .predecir_2025()
         .guardar_resultados())

def main():
    """Función principal para ejecutar modelo XGBoost PCA"""
    
    print("MODELO 2: XGBOOST CON COMPONENTES PCA PARA PREDICCIÓN FIES")
    print("=" * 60)
    print("INICIANDO PIPELINE COMPLETO - MODELO XGBOOST PCA")
    print("=" * 60)
    
    # Ejecutar modelo
    modelo_xgboost_pca = ModeloXGBoostPCA()
    modelo_xgboost_pca.ejecutar_pipeline_completo()
    
    print("\n" + "=" * 60)
    print("MODELO XGBOOST PCA COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    # Resumen final
    rmse_train = modelo_xgboost_pca.metricas_entrenamiento['RMSE']
    r2_train = modelo_xgboost_pca.metricas_entrenamiento['R2']
    rmse_cv = modelo_xgboost_pca.metricas_cv['RMSE_mean']
    r2_cv = modelo_xgboost_pca.metricas_cv['R2_mean']
    
    print(f"RESUMEN FINAL:")
    print(f"- Modelo: XGBoost con 7 componentes PCA")
    print(f"- Variables predichas: FIES_moderado_grave, FIES_grave")
    print(f"- RMSE entrenamiento: {rmse_train:.4f}")
    print(f"- R² entrenamiento: {r2_train:.4f}")
    print(f"- RMSE validación cruzada: {rmse_cv:.4f}")
    print(f"- R² validación cruzada: {r2_cv:.4f}")
    print(f"- Predicciones 2025: {len(modelo_xgboost_pca.predicciones_2025)} registros")
    print(f"- Archivos generados: modelo PCA, predicciones, métricas")
    
    return modelo_xgboost_pca

if __name__ == "__main__":
    modelo = main()

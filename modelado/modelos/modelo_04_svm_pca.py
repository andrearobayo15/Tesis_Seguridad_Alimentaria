"""
MODELO 4: SVM CON COMPONENTES PCA PARA PREDICCIÓN FIES
======================================================
Implementación de Support Vector Machine usando 7 componentes principales 
para predecir variables FIES_moderado_grave y FIES_grave con parámetros
optimizados para datos PCA.

Autor: Análisis PCA - Tesis Maestría
Fecha: 2025-08-26
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModeloSVMPCA:
    """Modelo SVM usando componentes PCA para predicción FIES"""
    
    def __init__(self):
        """Inicializar modelo SVM PCA"""
        self.df = None
        self.X_train = None
        self.y_train = None
        self.scaler_X = None
        self.scaler_y = None
        self.modelo = None
        self.predicciones_2025 = None
        self.metricas_cv = None
        self.metricas_entrenamiento = None
        self.mejor_params = None
        
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
        
        # Filtrar datos para entrenamiento (2022-2024)
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
        self.X_train = self.X_train.fillna(0)
        self.y_train = self.y_train.fillna(self.y_train.median())
        
        print(f"   Estructura de entrenamiento:")
        print(f"     X_train: {self.X_train.shape} (7 PCA + {len(dept_dummies.columns)} dept + 3 temporal)")
        print(f"     y_train: {self.y_train.shape} (FIES_moderado_grave, FIES_grave)")
        print(f"     Registros: {len(datos_entrenamiento)} (2022-2024)")
        
        return self
    
    def escalar_datos(self):
        """Escalar datos para SVM (crítico para rendimiento)"""
        print("\n3. Escalando datos para SVM...")
        
        # Escalar features (X)
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.X_train.columns, index=self.X_train.index)
        
        # Escalar targets (y) para mejor convergencia SVM
        self.scaler_y = StandardScaler()
        y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_train = pd.DataFrame(y_train_scaled, columns=self.y_train.columns, index=self.y_train.index)
        
        print("   Escalado completado:")
        print(f"     X_train escalado: media ~ 0, std ~ 1")
        print(f"     y_train escalado: media ~ 0, std ~ 1")
        print("   Escaladores guardados para predicción")
        
        return self
    
    def optimizar_hiperparametros(self):
        """Optimizar hiperparámetros SVM usando GridSearch con validación temporal"""
        print("\n4. Optimizando hiperparámetros SVM...")
        
        # Parámetros a optimizar (reducidos para eficiencia)
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],           # Regularización
            'estimator__epsilon': [0.01, 0.1, 0.2],     # Tolerancia
            'estimator__kernel': ['rbf', 'linear'],      # Kernel
            'estimator__gamma': ['scale', 'auto']        # Para kernel RBF
        }
        
        # Modelo base SVM
        svm_base = SVR()
        modelo_multi = MultiOutputRegressor(svm_base, n_jobs=1)  # SVM no paralelizable internamente
        
        # GridSearch con validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)  # Reducido para eficiencia
        
        print("   Ejecutando GridSearch (esto puede tomar varios minutos)...")
        print(f"   Combinaciones a probar: {len(param_grid['estimator__C']) * len(param_grid['estimator__epsilon']) * len(param_grid['estimator__kernel']) * len(param_grid['estimator__gamma'])}")
        
        grid_search = GridSearchCV(
            modelo_multi,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Ejecutar optimización
        grid_search.fit(self.X_train, self.y_train)
        
        # Mejores parámetros
        self.mejor_params = grid_search.best_params_
        self.modelo = grid_search.best_estimator_
        
        print(f"   Optimización completada:")
        print(f"     Mejor score: {-grid_search.best_score_:.4f}")
        print(f"     Mejores parámetros:")
        for param, valor in self.mejor_params.items():
            print(f"       {param}: {valor}")
        
        return self
    
    def entrenar_modelo_final(self):
        """Entrenar modelo final y evaluar con validación cruzada temporal"""
        print("\n5. Entrenando modelo final SVM PCA...")
        
        # VALIDACIÓN CRUZADA TEMPORAL con mejores parámetros
        print("   Evaluando con validación cruzada temporal...")
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {'RMSE': [], 'MAE': [], 'R2': []}
        
        fold = 1
        for train_idx, val_idx in tscv.split(self.X_train):
            print(f"   Fold {fold}/5...")
            
            # Split temporal
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # Crear modelo con mejores parámetros
            svm_fold = SVR(
                C=self.mejor_params['estimator__C'],
                epsilon=self.mejor_params['estimator__epsilon'],
                kernel=self.mejor_params['estimator__kernel'],
                gamma=self.mejor_params['estimator__gamma']
            )
            modelo_fold = MultiOutputRegressor(svm_fold, n_jobs=-1)
            
            # Entrenar en fold
            modelo_fold.fit(X_fold_train, y_fold_train)
            
            # Predecir en validación
            y_pred_val = modelo_fold.predict(X_fold_val)
            
            # Desescalar predicciones para métricas reales
            y_pred_val_real = self.scaler_y.inverse_transform(y_pred_val)
            y_val_real = self.scaler_y.inverse_transform(y_fold_val)
            
            # Métricas por fold (promedio de FIES_moderado_grave y FIES_grave)
            fold_rmse = []
            fold_mae = []
            fold_r2 = []
            
            for i in range(len(self.y_train.columns)):
                rmse = np.sqrt(mean_squared_error(y_val_real[:, i], y_pred_val_real[:, i]))
                mae = mean_absolute_error(y_val_real[:, i], y_pred_val_real[:, i])
                r2 = r2_score(y_val_real[:, i], y_pred_val_real[:, i])
                
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
        
        print(f"\n   MÉTRICAS VALIDACIÓN CRUZADA TEMPORAL SVM PCA:")
        print(f"     RMSE: {self.metricas_cv['RMSE_mean']:.4f} ± {self.metricas_cv['RMSE_std']:.4f}")
        print(f"     MAE:  {self.metricas_cv['MAE_mean']:.4f} ± {self.metricas_cv['MAE_std']:.4f}")
        print(f"     R²:   {self.metricas_cv['R2_mean']:.4f} ± {self.metricas_cv['R2_std']:.4f}")
        
        # Entrenar modelo final con todos los datos (ya está entrenado en self.modelo)
        print("   Modelo final ya entrenado con mejores parámetros")
        
        # Métricas de entrenamiento (referencia, en escala original)
        y_pred_train_scaled = self.modelo.predict(self.X_train)
        y_pred_train = self.scaler_y.inverse_transform(y_pred_train_scaled)
        y_train_original = self.scaler_y.inverse_transform(self.y_train)
        
        rmse_train = np.mean([np.sqrt(mean_squared_error(y_train_original[:, i], y_pred_train[:, i])) 
                             for i in range(len(self.y_train.columns))])
        mae_train = np.mean([mean_absolute_error(y_train_original[:, i], y_pred_train[:, i]) 
                            for i in range(len(self.y_train.columns))])
        r2_train = np.mean([r2_score(y_train_original[:, i], y_pred_train[:, i]) 
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
        print("\n6. Generando predicciones FIES 2025 con SVM PCA...")
        
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
        
        # Escalar features para predicción
        X_pred_scaled = self.scaler_X.transform(X_pred)
        
        # Generar predicciones (escaladas)
        predicciones_scaled = self.modelo.predict(X_pred_scaled)
        
        # Desescalar predicciones a escala original
        predicciones = self.scaler_y.inverse_transform(predicciones_scaled)
        
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
        """Guardar modelo, predicciones y métricas SVM PCA"""
        print("\n7. Guardando resultados SVM PCA...")
        
        # Crear directorios si no existen
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas', exist_ok=True)
        
        # 1. Guardar modelo y escaladores
        modelo_completo = {
            'modelo': self.modelo,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'mejores_parametros': self.mejor_params
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos/svm_pca_modelo.pkl', 'wb') as f:
            pickle.dump(modelo_completo, f)
        
        # 2. Guardar predicciones 2025
        self.predicciones_2025.to_csv(
            'd:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/svm_pca_predicciones_2025.csv', 
            index=False
        )
        
        # 3. Guardar métricas
        metricas_completas = {
            'modelo': 'SVM_PCA',
            'componentes_pca': 7,
            'hiperparametros_optimizados': self.mejor_params,
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
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/svm_pca_metricas.json', 'w') as f:
            json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
        
        print("   Modelo SVM PCA guardado: modelos/svm_pca_modelo.pkl")
        print("   Predicciones: predicciones/svm_pca_predicciones_2025.csv")
        print("   Métricas: metricas/svm_pca_metricas.json")
        
        return self
    
    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo del modelo SVM PCA"""
        return (self.cargar_datos_pca()
         .preparar_datos_entrenamiento()
         .escalar_datos()
         .optimizar_hiperparametros()
         .entrenar_modelo_final()
         .predecir_2025()
         .guardar_resultados())

def main():
    """Función principal para ejecutar modelo SVM PCA"""
    
    print("MODELO 4: SVM CON COMPONENTES PCA PARA PREDICCIÓN FIES")
    print("=" * 55)
    print("INICIANDO PIPELINE COMPLETO - MODELO SVM PCA")
    print("=" * 55)
    
    # Ejecutar modelo
    modelo_svm_pca = ModeloSVMPCA()
    modelo_svm_pca.ejecutar_pipeline_completo()
    
    print("\n" + "=" * 55)
    print("MODELO SVM PCA COMPLETADO EXITOSAMENTE")
    print("=" * 55)
    
    # Resumen final
    rmse_train = modelo_svm_pca.metricas_entrenamiento['RMSE']
    r2_train = modelo_svm_pca.metricas_entrenamiento['R2']
    rmse_cv = modelo_svm_pca.metricas_cv['RMSE_mean']
    r2_cv = modelo_svm_pca.metricas_cv['R2_mean']
    
    print(f"RESUMEN FINAL:")
    print(f"- Modelo: SVM con 7 componentes PCA")
    print(f"- Variables predichas: FIES_moderado_grave, FIES_grave")
    print(f"- Mejores parámetros: {modelo_svm_pca.mejor_params}")
    print(f"- RMSE entrenamiento: {rmse_train:.4f}")
    print(f"- R² entrenamiento: {r2_train:.4f}")
    print(f"- RMSE validación cruzada: {rmse_cv:.4f}")
    print(f"- R² validación cruzada: {r2_cv:.4f}")
    print(f"- Predicciones 2025: {len(modelo_svm_pca.predicciones_2025)} registros")
    print(f"- Archivos generados: modelo SVM PCA, predicciones, métricas")
    
    return modelo_svm_pca

if __name__ == "__main__":
    modelo = main()

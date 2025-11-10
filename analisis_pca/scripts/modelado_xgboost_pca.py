"""
Modelado XGBoost con Componentes PCA
Basado en configuraciones de Martini et al. (2022)
"""

import pandas as pd
import numpy as np
import yaml
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class XGBoostPCAModeler:
    def __init__(self, config_path):
        """Inicializar modelador con configuración"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.results = {}
        self.predictions = {}
    
    def cargar_datos(self, archivo_datos):
        """Cargar datos PCA"""
        print("Cargando datos PCA...")
        df = pd.read_csv(archivo_datos)
        
        # Separar por año
        train_data = df[df['año'].isin([2022, 2023, 2024])].copy()
        test_data = df[df['año'] == 2025].copy()
        
        print(f"Datos entrenamiento: {len(train_data):,} registros")
        print(f"Datos predicción: {len(test_data):,} registros")
        
        return train_data, test_data
    
    def preparar_features_targets(self, df):
        """Preparar features y targets"""
        features = self.config['variables']['features']
        targets = self.config['variables']['targets']
        
        X = df[features].copy()
        y = {}
        
        for target in targets:
            y[target] = df[target].copy()
        
        return X, y
    
    def entrenar_modelo_target(self, X_train, y_train, target_name):
        """Entrenar modelo para un target específico"""
        print(f"\nEntrenando modelo para {target_name}...")
        
        # Filtrar datos completos
        mask = ~y_train.isnull()
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]
        
        print(f"Registros con datos completos: {len(X_train_clean):,}")
        
        if len(X_train_clean) < 100:
            print(f"Advertencia: Pocos datos para {target_name}")
            return None, None
        
        # Configurar XGBoost
        xgb_params = {
            'objective': self.config['objetivo'],
            'random_state': self.config['configuracion_entrenamiento']['random_state'],
            'n_jobs': self.config['configuracion_entrenamiento']['n_jobs'],
            'verbosity': 0
        }
        
        estimator = xgb.XGBRegressor(**xgb_params)
        
        # Grid Search con validación temporal
        cv_folds = self.config['configuracion_entrenamiento']['cv_folds']
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.config['hiperparametros_grid_search'],
            cv=tscv,
            scoring=['neg_mean_absolute_error', 'r2'],
            refit='r2',
            n_jobs=self.config['configuracion_entrenamiento']['n_jobs'],
            verbose=1
        )
        
        # Entrenar
        grid_search.fit(X_train_clean, y_train_clean)
        
        # Mejor modelo
        best_model = grid_search.best_estimator_
        
        # Resultados
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        print(f"Mejores parámetros: {results['best_params']}")
        print(f"Mejor score R2: {results['best_score']:.4f}")
        
        return best_model, results
    
    def evaluar_modelo(self, model, X_test, y_test, target_name):
        """Evaluar modelo en datos de test"""
        if model is None:
            return None
        
        # Filtrar datos completos
        mask = ~y_test.isnull()
        X_test_clean = X_test[mask]
        y_test_clean = y_test[mask]
        
        if len(X_test_clean) == 0:
            print(f"No hay datos de test para {target_name}")
            return None
        
        # Predicciones
        y_pred = model.predict(X_test_clean)
        
        # Métricas
        r2 = r2_score(y_test_clean, y_pred)
        mae = mean_absolute_error(y_test_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
        
        metricas = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'n_test': len(X_test_clean)
        }
        
        print(f"\nMétricas {target_name}:")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        return metricas, y_pred, y_test_clean
    
    def generar_predicciones_2025(self, model, X_2025, target_name):
        """Generar predicciones para 2025"""
        if model is None:
            return None
        
        predicciones = model.predict(X_2025)
        
        return predicciones
    
    def entrenar_todos_modelos(self, archivo_datos):
        """Entrenar modelos para todos los targets"""
        print("=" * 60)
        print("ENTRENAMIENTO MODELOS XGBOOST PCA")
        print("=" * 60)
        
        # Cargar datos
        train_data, test_data = self.cargar_datos(archivo_datos)
        
        # Preparar features y targets
        X_train, y_train = self.preparar_features_targets(train_data)
        X_test, y_test = self.preparar_features_targets(test_data)
        
        # Entrenar para cada target
        for target in self.config['variables']['targets']:
            print(f"\n{'='*40}")
            print(f"TARGET: {target}")
            print(f"{'='*40}")
            
            # Entrenar modelo
            model, results = self.entrenar_modelo_target(X_train, y_train[target], target)
            
            if model is not None:
                # Guardar modelo y resultados
                self.models[target] = model
                self.results[target] = results
                
                # Evaluar en datos de validación (si existen)
                if target in y_test and not y_test[target].isnull().all():
                    eval_results = self.evaluar_modelo(model, X_test, y_test[target], target)
                    if eval_results:
                        self.results[target]['evaluacion'] = eval_results[0]
                
                # Generar predicciones 2025
                pred_2025 = self.generar_predicciones_2025(model, X_test, target)
                if pred_2025 is not None:
                    self.predictions[target] = pred_2025
        
        return self.models, self.results, self.predictions
    
    def guardar_resultados(self, directorio_salida):
        """Guardar modelos y resultados"""
        import os
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Guardar modelos
        for target, model in self.models.items():
            archivo_modelo = f"{directorio_salida}/modelo_xgboost_{target}.pkl"
            with open(archivo_modelo, 'wb') as f:
                pickle.dump(model, f)
            print(f"Modelo guardado: {archivo_modelo}")
        
        # Guardar resultados
        archivo_resultados = f"{directorio_salida}/resultados_xgboost_pca.pkl"
        with open(archivo_resultados, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Guardar predicciones
        archivo_predicciones = f"{directorio_salida}/predicciones_2025.pkl"
        with open(archivo_predicciones, 'wb') as f:
            pickle.dump(self.predictions, f)
        
        print(f"Resultados guardados en: {directorio_salida}")

def main():
    """Función principal"""
    # Configuración
    config_path = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/configuracion_xgboost_pca.yml"
    datos_path = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/base_pca_con_objetivos.csv"
    salida_path = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/modelos"
    
    # Crear modelador
    modelador = XGBoostPCAModeler(config_path)
    
    # Entrenar modelos
    models, results, predictions = modelador.entrenar_todos_modelos(datos_path)
    
    # Guardar resultados
    modelador.guardar_resultados(salida_path)
    
    print("\n" + "=" * 60)
    print("MODELADO COMPLETADO")
    print("=" * 60)
    print(f"Modelos entrenados: {len(models)}")
    print(f"Predicciones generadas: {len(predictions)}")

if __name__ == "__main__":
    main()

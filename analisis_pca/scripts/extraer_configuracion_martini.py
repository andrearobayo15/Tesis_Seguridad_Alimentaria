"""
Extracción de Configuraciones XGBoost de Martini et al.
Análisis de hiperparámetros y configuraciones del paper de referencia
"""

import pandas as pd
import yaml
import os

def extraer_configuracion_martini():
    """Extraer configuraciones de XGBoost del estudio de Martini"""
    print("=" * 60)
    print("CONFIGURACIONES XGBOOST DE MARTINI ET AL.")
    print("=" * 60)
    
    # Configuración extraída del archivo selected.yml
    configuracion_martini = {
        'modelo': 'XGBoost',
        'objetivo': 'reg:logistic',
        'hiperparametros_grid_search': {
            'max_depth': [4, 5, 6],
            'n_estimators': [100, 150, 200], 
            'learning_rate': [0.05, 0.1, 0.3]
        },
        'configuracion_entrenamiento': {
            'n_bootstrap': 100,
            'cv_folds': 4,
            'scoring': ['neg_mean_absolute_error', 'r2'],
            'min_diff': True,  # Criterio para seleccionar mejor modelo
            'n_jobs': 2,
            'verbosity': 0
        },
        'validacion_temporal': {
            'number_of_splits_validation': 4,
            'number_of_splits_test': 1,
            'months_aggregated_validation': 1,
            'months_aggregated_test': 2
        }
    }
    
    print("MODELO UTILIZADO:")
    print(f"  - Algoritmo: {configuracion_martini['modelo']}")
    print(f"  - Objetivo: {configuracion_martini['objetivo']}")
    
    print(f"\nHIPERPARAMETROS GRID SEARCH:")
    for param, valores in configuracion_martini['hiperparametros_grid_search'].items():
        print(f"  - {param}: {valores}")
    
    print(f"\nCONFIGURACION ENTRENAMIENTO:")
    for param, valor in configuracion_martini['configuracion_entrenamiento'].items():
        print(f"  - {param}: {valor}")
    
    print(f"\nVALIDACION TEMPORAL:")
    for param, valor in configuracion_martini['validacion_temporal'].items():
        print(f"  - {param}: {valor}")
    
    return configuracion_martini

def adaptar_configuracion_para_pca():
    """Adaptar configuración de Martini para nuestros datos PCA"""
    print(f"\n" + "=" * 60)
    print("ADAPTACION PARA DATOS PCA")
    print("=" * 60)
    
    # Configuración base de Martini
    config_martini = extraer_configuracion_martini()
    
    # Adaptación para nuestro caso
    config_pca = {
        'modelo': 'XGBoost',
        'objetivo': 'reg:squarederror',  # Cambio a regresión estándar
        'hiperparametros_grid_search': {
            'max_depth': [3, 4, 5, 6],  # Agregamos profundidad menor
            'n_estimators': [50, 100, 150, 200],  # Agregamos menos estimadores
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Agregamos learning rate menor
            'subsample': [0.8, 0.9, 1.0],  # Agregamos subsample
            'colsample_bytree': [0.8, 0.9, 1.0]  # Agregamos colsample
        },
        'configuracion_entrenamiento': {
            'n_bootstrap': 50,  # Reducimos bootstrap para eficiencia
            'cv_folds': 5,  # Aumentamos CV folds
            'scoring': ['neg_mean_absolute_error', 'r2'],
            'min_diff': True,
            'n_jobs': -1,  # Usar todos los cores
            'verbosity': 1,  # Más verbose para debugging
            'random_state': 42
        },
        'validacion_temporal': {
            'test_size': 0.25,  # 2025 como test (25% de datos)
            'validation_size': 0.2,  # 20% para validación
            'shuffle': False  # Mantener orden temporal
        },
        'variables': {
            'features': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'],
            'targets': ['FIES_moderado_grave', 'FIES_grave'],
            'identificadores': ['departamento', 'año', 'mes', 'fecha', 'clave']
        }
    }
    
    print("ADAPTACIONES REALIZADAS:")
    print("  - Objetivo: reg:squarederror (regresión estándar)")
    print("  - Hiperparámetros: Expandido grid search")
    print("  - Bootstrap: Reducido a 50 para eficiencia")
    print("  - CV: Aumentado a 5 folds")
    print("  - Features: 7 componentes PCA")
    print("  - Targets: FIES_moderado_grave y FIES_grave")
    
    return config_pca

def generar_configuracion_yaml(config_pca):
    """Generar archivo YAML con configuración adaptada"""
    print(f"\n" + "=" * 50)
    print("GENERANDO ARCHIVO CONFIGURACION")
    print("=" * 50)
    
    archivo_config = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/configuracion_xgboost_pca.yml"
    
    with open(archivo_config, 'w', encoding='utf-8') as f:
        yaml.dump(config_pca, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Configuración guardada: configuracion_xgboost_pca.yml")
    
    return archivo_config

def crear_script_modelado_xgboost():
    """Crear script de modelado XGBoost basado en configuración de Martini"""
    print(f"\n" + "=" * 50)
    print("CREANDO SCRIPT MODELADO XGBOOST")
    print("=" * 50)
    
    script_content = '''"""
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
        print(f"\\nEntrenando modelo para {target_name}...")
        
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
        
        print(f"\\nMétricas {target_name}:")
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
            print(f"\\n{'='*40}")
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
    
    print("\\n" + "=" * 60)
    print("MODELADO COMPLETADO")
    print("=" * 60)
    print(f"Modelos entrenados: {len(models)}")
    print(f"Predicciones generadas: {len(predictions)}")

if __name__ == "__main__":
    main()
'''
    
    archivo_script = "d:/Tesis maestria/Tesis codigo/analisis_pca/scripts/modelado_xgboost_pca.py"
    with open(archivo_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"Script creado: modelado_xgboost_pca.py")
    
    return archivo_script

def main():
    """Función principal"""
    print("EXTRAYENDO CONFIGURACIONES DE MARTINI ET AL.")
    
    # 1. Extraer configuración original
    config_martini = extraer_configuracion_martini()
    
    # 2. Adaptar para PCA
    config_pca = adaptar_configuracion_para_pca()
    
    # 3. Generar YAML
    archivo_config = generar_configuracion_yaml(config_pca)
    
    # 4. Crear script de modelado
    archivo_script = crear_script_modelado_xgboost()
    
    print(f"\n" + "=" * 60)
    print("EXTRACCION COMPLETADA")
    print("=" * 60)
    print("Archivos generados:")
    print(f"  - {archivo_config}")
    print(f"  - {archivo_script}")
    
    return config_martini, config_pca

if __name__ == "__main__":
    config_martini, config_pca = main()

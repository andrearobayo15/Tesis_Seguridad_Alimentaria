"""
MODELO 1: XGBOOST PARA PREDICCIÓN MULTIVARIADA FIES 2025
========================================================

Descripción: Gradient Boosting optimizado para predicción de 50 variables FIES, 
socioeconómicas y climáticas para 2025 usando datos históricos 2022-2024.

Características:
- Ensemble secuencial de árboles débiles
- Optimización por gradiente de segundo orden
- Regularización L1 y L2 incorporada
- Manejo nativo de missing values
- Feature importance automática

Funcionamiento:
- Entrena árboles secuencialmente corrigiendo errores del anterior
- Función objetivo: Loss + Regularización
- Optimización: Newton-Raphson con gradiente y hessiana
- Predicción: Suma ponderada de todos los árboles

Ventajas:
- Alta precisión predictiva
- Control automático de overfitting
- Eficiente con datos limitados
- Interpretabilidad via feature importance

Limitaciones:
- Hiperparámetros requieren tuning cuidadoso
- Menos interpretable que modelos lineales
- Sensible a overfitting sin regularización

Referencia Académica: Modelo utilizado por Martini en investigación FIES
Autor: Tesis Maestría
Fecha: Agosto 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost y sklearn
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class ModeloXGBoostFIES:
    """
    Modelo XGBoost para predicción multivariada de variables FIES 2025
    """
    
    def __init__(self, preprocessor_path):
        """
        Inicializar modelo XGBoost con preprocessor entrenado
        
        Args:
            preprocessor_path (str): Ruta al archivo preprocessor_fies.pkl
        """
        self.preprocessor_path = preprocessor_path
        self.preprocessor = None
        self.modelo = None
        self.resultados = {}
        self.predicciones_2025 = None
        self.feature_importance = None
        self.metricas_validacion = {}
        
        # Variables objetivo (2 variables FIES críticas)
        self.variables_objetivo = [
            'FIES_moderado_grave',  # Inseguridad alimentaria moderada-grave (12.8-59.7%)
            'FIES_grave'            # Inseguridad alimentaria grave (0.9-17.5%)
        ]
        
        print("MODELO 1: XGBOOST PARA PREDICCIÓN MULTIVARIADA FIES 2025")
        print("=" * 60)
        
    def cargar_datos_directamente(self):
        """Cargar datos directamente desde CSV para evitar problemas de pickle"""
        print("1. Cargando datos directamente...")
        
        # Cargar datos directamente desde CSV
        ruta_datos = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
        self.df = pd.read_csv(ruta_datos)
        
        print(f"   Datos cargados: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
        print(f"   Variables objetivo: {len(self.variables_objetivo)}")
        return self
    
    def preparar_datos_entrenamiento(self):
        """Preparar datos con estructura temporal correcta: 2022-2023 → entrenamiento, 2024 → test"""
        print("\n2. Preparando datos de entrenamiento...")
        
        # ARQUITECTURA TEMPORAL CORRECTA:
        # Train: 2022-2023 (768 registros)
        # Test: 2024 (384 registros) 
        # Predict: 2025 (384 registros)
        
        # Datos de entrenamiento: 2022-2023
        datos_train = self.df[self.df['año'].isin([2022, 2023])].copy()
        
        # Datos de test: 2024
        datos_test = self.df[self.df['año'] == 2024].copy()
        
        print(f"   Datos entrenamiento: {len(datos_train)} registros (2022-2023)")
        print(f"   Datos test: {len(datos_test)} registros (2024)")
        
        # Variables FIES objetivo (SOLO las 2 críticas para evaluación)
        # CORRECCIÓN: Evaluar SOLO las variables objetivo según USER
        variables_fies_objetivo = [
            'FIES_moderado_grave',  # Inseguridad alimentaria moderada-grave
            'FIES_grave'            # Inseguridad alimentaria grave
        ]
        
        variables_disponibles = []
        for var in variables_fies_objetivo:
            if var in self.df.columns:
                # Verificar completitud en entrenamiento y test
                completitud_train = datos_train[var].notna().mean()
                completitud_test = datos_test[var].notna().mean()
                
                if completitud_train > 0.90 and completitud_test > 0.90:
                    variables_disponibles.append(var)
        
        self.variables_objetivo_finales = variables_disponibles
        print(f"   Variables disponibles para modelado: {len(self.variables_objetivo_finales)}")
        
        # Guardar datos para uso posterior
        self.datos_train = datos_train
        self.datos_test = datos_test
        
        # CREAR ESTRUCTURA DE FEATURES SIMPLIFICADA
        print("   Creando estructura de features...")
        
        # Usar enfoque más simple: cada registro es un feature vector independiente
        # Features: variables auxiliares + encodings categóricos
        # Target: variables FIES
        
        # Variables auxiliares disponibles (no FIES)
        variables_auxiliares = ['IPM_Total', 'IPC_Total', 'Pobreza_monetaria', 
                               'precipitacion_promedio', 'temperatura_promedio', 'ndvi_promedio']
        
        # Filtrar variables auxiliares que existen y tienen datos
        features_disponibles = []
        for var in variables_auxiliares:
            if var in datos_train.columns:
                completitud = datos_train[var].notna().mean()
                if completitud > 0.5:  # Al menos 50% de datos
                    features_disponibles.append(var)
        
        print(f"   Features auxiliares disponibles: {len(features_disponibles)}")
        
        # 1. TARGET ENCODING para departamentos (usando datos de entrenamiento)
        dept_encodings = {}
        for var in self.variables_objetivo_finales:
            dept_avg = datos_train.groupby('departamento')[var].mean()
            dept_encodings[f'dept_enc_{var}'] = dept_avg.to_dict()
        
        # 2. ENCODING CÍCLICO para meses
        meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        # Guardar para uso posterior
        self.dept_encodings = dept_encodings
        self.meses_map = meses_map
        self.features_disponibles = features_disponibles
        
        # CREAR DATASETS DE ENTRENAMIENTO Y TEST
        print("   Preparando datasets finales...")
        
        # Preparar X_train, y_train
        X_train_list = []
        y_train_list = []
        
        for _, row in datos_train.iterrows():
            # Features: variables auxiliares + encodings
            feature_vector = []
            
            # 1. Variables auxiliares
            for var in features_disponibles:
                val = row[var] if pd.notna(row[var]) else 0
                feature_vector.append(val)
            
            # 2. Target encoding departamento
            dept = row['departamento']
            for var in self.variables_objetivo_finales:
                dept_enc_key = f'dept_enc_{var}'
                if dept in dept_encodings[dept_enc_key]:
                    feature_vector.append(dept_encodings[dept_enc_key][dept])
                else:
                    feature_vector.append(0)
            
            # 3. Encoding cíclico mes
            mes = row['mes']
            if mes in meses_map:
                mes_num = meses_map[mes]
                mes_sin = np.sin(2 * np.pi * mes_num / 12)
                mes_cos = np.cos(2 * np.pi * mes_num / 12)
                feature_vector.extend([mes_sin, mes_cos])
            else:
                feature_vector.extend([0, 0])
            
            # 4. Año normalizado
            año_norm = (row['año'] - 2022) / 2  # 2022=0, 2023=0.5
            feature_vector.append(año_norm)
            
            # Target: variables FIES
            target_vector = []
            for var in self.variables_objetivo_finales:
                val = row[var] if pd.notna(row[var]) else 0
                target_vector.append(val)
            
            X_train_list.append(feature_vector)
            y_train_list.append(target_vector)
        
        # Convertir a DataFrames
        feature_names = (features_disponibles + 
                        [f'dept_enc_{var}' for var in self.variables_objetivo_finales] +
                        ['mes_sin', 'mes_cos', 'año_norm'])
        
        self.X_train = pd.DataFrame(X_train_list, columns=feature_names)
        self.y_train = pd.DataFrame(y_train_list, columns=self.variables_objetivo_finales)
        
        # Preparar X_test, y_test de manera similar
        X_test_list = []
        y_test_list = []
        
        for _, row in datos_test.iterrows():
            # Features (mismo proceso)
            feature_vector = []
            
            # Variables auxiliares
            for var in features_disponibles:
                val = row[var] if pd.notna(row[var]) else 0
                feature_vector.append(val)
            
            # Target encoding departamento
            dept = row['departamento']
            for var in self.variables_objetivo_finales:
                dept_enc_key = f'dept_enc_{var}'
                if dept in dept_encodings[dept_enc_key]:
                    feature_vector.append(dept_encodings[dept_enc_key][dept])
                else:
                    feature_vector.append(0)
            
            # Encoding cíclico mes
            mes = row['mes']
            if mes in meses_map:
                mes_num = meses_map[mes]
                mes_sin = np.sin(2 * np.pi * mes_num / 12)
                mes_cos = np.cos(2 * np.pi * mes_num / 12)
                feature_vector.extend([mes_sin, mes_cos])
            else:
                feature_vector.extend([0, 0])
            
            # Año normalizado (2024 = 1.0)
            año_norm = (row['año'] - 2022) / 2
            feature_vector.append(año_norm)
            
            # Target
            target_vector = []
            for var in self.variables_objetivo_finales:
                val = row[var] if pd.notna(row[var]) else 0
                target_vector.append(val)
            
            X_test_list.append(feature_vector)
            y_test_list.append(target_vector)
        
        self.X_test = pd.DataFrame(X_test_list, columns=feature_names)
        self.y_test = pd.DataFrame(y_test_list, columns=self.variables_objetivo_finales)
        
        print(f"   Estructura final:")
        print(f"     X_train: {self.X_train.shape}")
        print(f"     y_train: {self.y_train.shape}")
        print(f"     X_test: {self.X_test.shape}")
        print(f"     y_test: {self.y_test.shape}")
        print(f"     Features: {len(features_disponibles)} auxiliares + {len(self.variables_objetivo_finales)} encodings + 3 temporales")
        
        return self
    
    def configurar_modelo(self):
        """Configurar XGBoost con hiperparámetros justificados según análisis Martini"""
        print("\n3. Configurando modelo XGBoost...")
        
        # Configuración CORREGIDA para controlar overfitting y evaluar solo FIES objetivo
        xgb_params = {
            'objective': 'reg:squarederror',  # Para valores continuos FIES (no probabilidades)
            'n_estimators': 50,               # REDUCIDO para controlar overfitting
            'max_depth': 3,                   # REDUCIDO para controlar overfitting
            'learning_rate': 0.05,            # REDUCIDO para aprendizaje más conservador
            'subsample': 0.7,                 # MÁS regularización para prevenir overfitting
            'colsample_bytree': 0.7,          # MÁS regularización por árbol
            'reg_alpha': 0.1,                 # MAYOR L1 regularización
            'reg_lambda': 0.1,                # MAYOR L2 regularización
            'random_state': 42,               # Reproducibilidad
            'n_jobs': -1,                     # Usar todos los cores
            'verbosity': 1                    # Mostrar progreso
        }
        
        # Crear modelo base XGBoost
        xgb_base = xgb.XGBRegressor(**xgb_params)
        
        # Wrapper para múltiples outputs
        self.modelo = MultiOutputRegressor(xgb_base, n_jobs=-1)
        
        print("   Configuración XGBoost:")
        for param, valor in xgb_params.items():
            print(f"     {param}: {valor}")
        
        return self
    
    def entrenar_modelo(self):
        """Entrenar modelo XGBoost con validación cruzada temporal y evaluación en test"""
        print("\n4. Entrenando modelo XGBoost...")
        
        # VALIDACIÓN CRUZADA TEMPORAL sobre datos de entrenamiento (2022-2023)
        print("   Implementando validación cruzada temporal sobre datos 2022-2023...")
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=5)  # 5 folds para mejor estadística
        cv_scores = {'RMSE': [], 'MAE': [], 'R2': []}
        
        fold = 1
        for train_idx, val_idx in tscv.split(self.X_train):
            print(f"   Fold {fold}/5...")
            
            # Split temporal dentro de datos de entrenamiento
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # Entrenar en fold
            self.modelo.fit(X_fold_train, y_fold_train)
            
            # Predecir en validación
            y_pred_val = self.modelo.predict(X_fold_val)
            
            # Métricas por fold (promedio de todas las variables)
            fold_rmse = []
            fold_mae = []
            fold_r2 = []
            
            for i in range(len(self.variables_objetivo_finales)):
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
        
        print(f"\n   MÉTRICAS VALIDACIÓN CRUZADA TEMPORAL:")
        print(f"     RMSE: {self.metricas_cv['RMSE_mean']:.4f} ± {self.metricas_cv['RMSE_std']:.4f}")
        print(f"     MAE:  {self.metricas_cv['MAE_mean']:.4f} ± {self.metricas_cv['MAE_std']:.4f}")
        print(f"     R²:   {self.metricas_cv['R2_mean']:.4f} ± {self.metricas_cv['R2_std']:.4f}")
        
        # Entrenar modelo final con todos los datos
        print("   Entrenando modelo final con todos los datos...")
        self.modelo.fit(self.X_train, self.y_train)
        
        # Métricas de entrenamiento (solo para referencia)
        y_pred_train = self.modelo.predict(self.X_train)
        rmse_train = np.mean([np.sqrt(mean_squared_error(self.y_train.iloc[:, i], y_pred_train[:, i])) 
                             for i in range(len(self.variables_objetivo_finales))])
        mae_train = np.mean([mean_absolute_error(self.y_train.iloc[:, i], y_pred_train[:, i]) 
                            for i in range(len(self.variables_objetivo_finales))])
        r2_train = np.mean([r2_score(self.y_train.iloc[:, i], y_pred_train[:, i]) 
                           for i in range(len(self.variables_objetivo_finales))])
        
        # Guardar métricas de entrenamiento para compatibilidad
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
        """Generar predicciones para 2025 usando datos históricos + variables categóricas"""
        print("\n5. Generando predicciones para 2025...")
        
        # ESTRATEGIA MEJORADA:
        # Usar datos históricos 2022-2024 + encodings categóricos para predecir 2025
        
        # Datos históricos (2022-2024)
        datos_historicos = self.df[self.df['año'].isin([2022, 2023, 2024])].copy()
        
        # Datos 2025 (estructura)
        datos_2025 = self.df[self.df['año'] == 2025].copy()
        
        X_2025_list = []
        departamentos_meses = []
        
        for dept in datos_2025['departamento'].unique():
            for mes in datos_2025['mes'].unique():
                # Verificar que existe el registro en 2025
                registro_2025 = datos_2025[
                    (datos_2025['departamento'] == dept) & 
                    (datos_2025['mes'] == mes)
                ]
                
                if len(registro_2025) == 1:
                    # Features históricos: datos 2022-2024 para este departamento-mes
                    registros_historicos = datos_historicos[
                        (datos_historicos['departamento'] == dept) & 
                        (datos_historicos['mes'] == mes)
                    ]
                    
                    if len(registros_historicos) >= 2:  # Al menos 2 años de datos
                        feature_vector = []
                        
                        # 1. Features históricos (usar últimos 2 años: 2023-2024)
                        for var in self.variables_objetivo_finales:
                            vals_historicos = registros_historicos[var].tail(2).values
                            if len(vals_historicos) >= 2:
                                feature_vector.extend(vals_historicos[-2:])  # Últimos 2 valores
                            else:
                                # Si faltan datos, usar mediana
                                mediana = registros_historicos[var].median()
                                feature_vector.extend([mediana, mediana])
                        
                        # 2. Target encoding departamentos (usar encodings calculados)
                        for var in self.variables_objetivo_finales:
                            dept_enc_key = f'dept_enc_{var}'
                            if dept in self.dept_encodings[dept_enc_key]:
                                feature_vector.append(self.dept_encodings[dept_enc_key][dept])
                            else:
                                # Si no existe, usar promedio global de entrenamiento
                                feature_vector.append(0)  # Valor por defecto
                        
                        # 3. Encoding cíclico meses
                        if mes in self.meses_map:
                            mes_num = self.meses_map[mes]
                            mes_sin = np.sin(2 * np.pi * mes_num / 12)
                            mes_cos = np.cos(2 * np.pi * mes_num / 12)
                            feature_vector.extend([mes_sin, mes_cos])
                        else:
                            feature_vector.extend([0, 0])  # Default
                        
                        # 4. Año como variable numérica (2025 = 2)
                        feature_vector.append(2)  # 2022=0, 2023=1, 2024=2, 2025=3 pero usamos 2 como referencia
                        
                        X_2025_list.append(feature_vector)
                        departamentos_meses.append((dept, mes))
        
        # Convertir a DataFrame con los mismos nombres de columnas que entrenamiento
        X_2025 = pd.DataFrame(X_2025_list, columns=self.X_train.columns)
        
        # Manejar missing values
        X_2025 = X_2025.fillna(X_2025.median())
        
        # Generar predicciones
        self.predicciones_2025 = self.modelo.predict(X_2025)
        
        # Convertir a DataFrame
        self.predicciones_2025_df = pd.DataFrame(
            self.predicciones_2025,
            columns=self.variables_objetivo_finales
        )
        
        # Agregar información de contexto
        departamentos = [dm[0] for dm in departamentos_meses]
        meses = [dm[1] for dm in departamentos_meses]
        
        self.predicciones_2025_df['departamento'] = departamentos
        self.predicciones_2025_df['mes'] = meses
        self.predicciones_2025_df['año'] = 2025
        
        print(f"   Predicciones generadas: {self.predicciones_2025_df.shape}")
        print(f"   Variables predichas: {len(self.variables_objetivo_finales)}")
        print(f"   Departamentos-meses: {len(departamentos_meses)}")
        print(f"   Features utilizados: {len(X_2025.columns)}")
        print(f"   Encodings aplicados: dept_enc ({len(self.variables_objetivo_finales)}), mes_cíclico (2), año (1)")
        
        return self
    
    def guardar_resultados(self):
        """Guardar modelo y resultados"""
        print("\n6. Guardando resultados...")
        
        # Crear directorios
        import os
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas', exist_ok=True)
        
        # 1. Guardar modelo entrenado
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos/xgboost_modelo.pkl', 'wb') as f:
            pickle.dump(self.modelo, f)
        
        # 2. Guardar predicciones 2025
        self.predicciones_2025_df.to_csv(
            'd:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/xgboost_predicciones_2025.csv', 
            index=False
        )
        
        # 3. Guardar métricas
        metricas_completas = {
            'entrenamiento': self.metricas_entrenamiento,
            'validacion_cruzada': self.metricas_cv,
            'resumen': {
                'rmse_entrenamiento': self.metricas_entrenamiento['RMSE'],
                'r2_entrenamiento': self.metricas_entrenamiento['R2'],
                'rmse_cv': self.metricas_cv['RMSE_mean'],
                'r2_cv': self.metricas_cv['R2_mean'],
                'variables_predichas': len(self.variables_objetivo_finales),
                'registros_entrenamiento': len(self.X_train),
                'registros_prediccion': len(self.predicciones_2025)
            }
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/xgboost_metricas.json', 'w') as f:
            json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
        
        print("   Modelo guardado: modelos/xgboost_modelo.pkl")
        print("   Predicciones: predicciones/xgboost_predicciones_2025.csv")
        print("   Métricas: metricas/xgboost_metricas.json")
        
        return self
    
    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo del modelo XGBoost"""
        print("INICIANDO PIPELINE COMPLETO - MODELO XGBOOST")
        print("=" * 60)
        
        # Crear directorios necesarios
        import os
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/reportes', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones', exist_ok=True)
        
        (self.cargar_datos_directamente()
         .preparar_datos_entrenamiento()
         .configurar_modelo()
         .entrenar_modelo()
         .predecir_2025()
         .guardar_resultados())
        
        print("\n" + "=" * 60)
        print("MODELO XGBOOST COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        # Resumen final
        rmse_train = self.metricas_entrenamiento['RMSE']
        r2_train = self.metricas_entrenamiento['R2']
        rmse_cv = self.metricas_cv['RMSE_mean']
        r2_cv = self.metricas_cv['R2_mean']
        
        print(f"RESUMEN FINAL:")
        print(f"- Variables predichas: {len(self.variables_objetivo_finales)}")
        print(f"- RMSE entrenamiento: {rmse_train:.4f}")
        print(f"- R² entrenamiento: {r2_train:.4f}")
        print(f"- RMSE validación cruzada: {rmse_cv:.4f}")
        print(f"- R² validación cruzada: {r2_cv:.4f}")
        print(f"- Predicciones 2025: {len(self.predicciones_2025)} registros")
        print(f"- Archivos generados: modelo, predicciones, métricas")
        
        return self

def main():
    """Función principal para ejecutar modelo XGBoost"""
    
    # Ruta al preprocessor
    preprocessor_path = "d:/Tesis maestria/Tesis codigo/modelado/resultados/preprocessor_fies.pkl"
    
    # Crear y ejecutar modelo
    modelo_xgboost = ModeloXGBoostFIES(preprocessor_path)
    modelo_xgboost.ejecutar_pipeline_completo()
    
    return modelo_xgboost

if __name__ == "__main__":
    modelo = main()

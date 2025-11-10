"""
MODELO 4: ELASTIC NET PARA PREDICCIÓN MULTIVARIADA FIES 2025

Descripción:
- Modelo Elastic Net para predicción de 50 variables socioeconómicas, FIES y climáticas
- Estructura temporal: 2022-2023 → 2024 (validación), 2022-2024 → 2025 (predicción)
- Variables categóricas: Target encoding + encoding cíclico
- Regularización: Combinación de Ridge (L2) y Lasso (L1) para selección de features
- Validación: Temporal con métricas RMSE, MAE, R²

Autor: Análisis de Tesis Maestría
Fecha: Agosto 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# Modelos y métricas
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración
import warnings
warnings.filterwarnings('ignore')

print("MODELO 4: ELASTIC NET PARA PREDICCIÓN MULTIVARIADA FIES 2025")
print("=" * 65)

class ModeloElasticNet:
    """Clase para modelo Elastic Net multivariado FIES 2025"""
    
    def __init__(self):
        """Inicializar el modelo Elastic Net"""
        self.df = None
        self.variables_objetivo = [
            # Variables IPM (9)
            'Analfabetismo', 'Bajo_logro_educativo', 'Barreras_acceso_salud',
            'Desempleo_larga_duracion', 'Inasistencia_escolar', 'Rezago_escolar',
            'Sin_aseguramiento_salud', 'Trabajo_informal', 'IPM_Total',
            
            # Variables ECV (27)
            'Vida_general', 'Salud', 'Seguridad', 'Trabajo_actividad', 'Tiempo_libre', 'Ingreso',
            'Propia_totalmente_pagada', 'Propia_la_estan_pagando', 'En_arriendo_o_subarriendo',
            'Con_permiso_sin_pago', 'Posesion_sin_titulo', 'Propiedad_colectiva',
            'Deficit_cuantitativo', 'Deficit_cualitativo', 'Deficit_habitacional',
            'No_alcanzan_gastos_minimos', 'Alcanzan_gastos_minimos', 'Cubren_mas_gastos_minimos',
            'Pobreza_monetaria', 'No_pobres', 'Energia', 'Gas_natural', 'Acueducto',
            'Alcantarillado', 'Recoleccion_basura', 'Telefono_fijo', 'Ningun_servicio',
            
            # Variables FIES (10)
            'FIES_preocupacion_alimentos', 'FIES_no_alimentos_saludables', 
            'FIES_poca_variedad_alimentos', 'FIES_saltar_comida', 'FIES_comio_menos',
            'FIES_sin_alimentos', 'FIES_hambre_sin_comer', 'FIES_no_comio_dia_entero',
            'FIES_moderado_grave', 'FIES_grave',
            
            # Variables climáticas (3)
            'precipitacion_promedio', 'temperatura_promedio', 'ndvi_promedio',
            
            # Variable IPC (1)
            'IPC_Total'
        ]
        
        self.variables_objetivo_finales = []
        self.X_train = None
        self.y_train = None
        self.modelo = None
        self.scaler_X = None
        self.scaler_y = None
        self.metricas_entrenamiento = {}
        self.predicciones_2025_df = None
        self.dept_encodings = {}
        self.meses_map = {}
    
    def cargar_datos_directamente(self):
        """Cargar datos directamente desde CSV"""
        print("1. Cargando datos directamente...")
        
        ruta_datos = 'd:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv'
        
        try:
            self.df = pd.read_csv(ruta_datos)
            print(f"   Datos cargados: {len(self.df)} registros, {len(self.df.columns)} columnas")
            print(f"   Variables objetivo: {len(self.variables_objetivo)}")
            
        except Exception as e:
            print(f"   ERROR al cargar datos: {e}")
            return None
        
        return self
    
    def preparar_datos_entrenamiento(self):
        """Preparar datos con estructura temporal correcta"""
        print("\n2. Preparando datos de entrenamiento...")
        
        # Datos de entrenamiento (features): 2022-2023
        datos_features = self.df[self.df['año'].isin([2022, 2023])].copy()
        
        # Datos objetivo (target): 2024
        datos_target = self.df[self.df['año'] == 2024].copy()
        
        print(f"   Registros features (2022-2023): {len(datos_features)}")
        print(f"   Registros target (2024): {len(datos_target)}")
        
        # Variables disponibles
        variables_disponibles = []
        for var in self.variables_objetivo:
            if var in self.df.columns:
                completitud_historica = datos_features[var].notna().mean()
                completitud_target = datos_target[var].notna().mean()
                
                if completitud_historica > 0.95 and completitud_target > 0.95:
                    variables_disponibles.append(var)
        
        self.variables_objetivo_finales = variables_disponibles
        print(f"   Variables disponibles para modelado: {len(self.variables_objetivo_finales)}")
        
        # CREAR ENCODINGS CATEGÓRICOS
        print("   Creando encodings categóricos...")
        
        # Target encoding para departamentos
        dept_encodings = {}
        for var in self.variables_objetivo_finales:
            dept_avg = datos_features.groupby('departamento')[var].mean()
            dept_encodings[f'dept_enc_{var}'] = dept_avg.to_dict()
        
        # Encoding cíclico para meses
        meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        self.dept_encodings = dept_encodings
        self.meses_map = meses_map
        
        # Crear estructura de datos temporal
        X_list = []
        y_list = []
        
        for dept in datos_target['departamento'].unique():
            for mes in datos_target['mes'].unique():
                target_row = datos_target[
                    (datos_target['departamento'] == dept) & 
                    (datos_target['mes'] == mes)
                ]
                
                if len(target_row) == 1:
                    feature_rows = datos_features[
                        (datos_features['departamento'] == dept) & 
                        (datos_features['mes'] == mes)
                    ]
                    
                    if len(feature_rows) == 2:
                        feature_vector = []
                        
                        # Features históricos
                        for var in self.variables_objetivo_finales:
                            vals_2022_2023 = feature_rows[var].values
                            feature_vector.extend(vals_2022_2023)
                        
                        # Target encoding departamentos
                        for var in self.variables_objetivo_finales:
                            dept_enc_key = f'dept_enc_{var}'
                            if dept in dept_encodings[dept_enc_key]:
                                feature_vector.append(dept_encodings[dept_enc_key][dept])
                            else:
                                feature_vector.append(datos_features[var].mean())
                        
                        # Encoding cíclico meses
                        if mes in meses_map:
                            mes_num = meses_map[mes]
                            mes_sin = np.sin(2 * np.pi * mes_num / 12)
                            mes_cos = np.cos(2 * np.pi * mes_num / 12)
                            feature_vector.extend([mes_sin, mes_cos])
                        else:
                            feature_vector.extend([0, 0])
                        
                        # Año
                        feature_vector.append(1)
                        
                        # Vector objetivo
                        target_vector = target_row[self.variables_objetivo_finales].values[0]
                        
                        X_list.append(feature_vector)
                        y_list.append(target_vector)
        
        # Convertir a DataFrames
        self.X_train = pd.DataFrame(X_list)
        self.y_train = pd.DataFrame(y_list, columns=self.variables_objetivo_finales)
        
        # Crear nombres de columnas
        feature_names = []
        for var in self.variables_objetivo_finales:
            feature_names.extend([f"{var}_2022", f"{var}_2023"])
        for var in self.variables_objetivo_finales:
            feature_names.append(f"dept_enc_{var}")
        feature_names.extend(['mes_sin', 'mes_cos', 'año_ref'])
        
        self.X_train.columns = feature_names
        
        # Manejar missing values
        self.X_train = self.X_train.fillna(self.X_train.median())
        self.y_train = self.y_train.fillna(self.y_train.median())
        
        print(f"   Estructura final:")
        print(f"     X_train: {self.X_train.shape}")
        print(f"     y_train: {self.y_train.shape}")
        
        return self
    
    def configurar_modelo(self):
        """Configurar modelo Elastic Net"""
        print("\n3. Configurando modelo Elastic Net...")
        
        elastic_config = {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'fit_intercept': True,
            'max_iter': 2000,
            'tol': 1e-4,
            'random_state': 42
        }
        
        print("   Configuración Elastic Net:")
        for key, value in elastic_config.items():
            print(f"     {key}: {value}")
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        base_elastic = ElasticNet(**elastic_config)
        self.modelo = MultiOutputRegressor(base_elastic, n_jobs=-1)
        
        print("   Escaladores configurados")
        print("   Regularización: L1 + L2 para selección de features")
        
        return self
    
    def entrenar_modelo(self):
        """Entrenar modelo Elastic Net"""
        print("\n4. Entrenando modelo Elastic Net...")
        
        X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        
        print("   Iniciando entrenamiento...")
        self.modelo.fit(X_train_scaled, y_train_scaled)
        print("   Entrenamiento completado")
        
        # Calcular métricas
        y_pred_train_scaled = self.modelo.predict(X_train_scaled)
        y_pred_train = self.scaler_y.inverse_transform(y_pred_train_scaled)
        
        self.metricas_entrenamiento = {}
        rmse_total = []
        mae_total = []
        r2_total = []
        
        for i, var in enumerate(self.variables_objetivo_finales):
            y_true_var = self.y_train.iloc[:, i]
            y_pred_var = y_pred_train[:, i]
            
            rmse = np.sqrt(mean_squared_error(y_true_var, y_pred_var))
            mae = mean_absolute_error(y_true_var, y_pred_var)
            r2 = r2_score(y_true_var, y_pred_var)
            
            self.metricas_entrenamiento[var] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            rmse_total.append(rmse)
            mae_total.append(mae)
            r2_total.append(r2)
        
        rmse_promedio = np.mean(rmse_total)
        mae_promedio = np.mean(mae_total)
        r2_promedio = np.mean(r2_total)
        
        print(f"   Métricas de entrenamiento:")
        print(f"     RMSE: {rmse_promedio:.4f}")
        print(f"     MAE: {mae_promedio:.4f}")
        print(f"     R²: {r2_promedio:.4f}")
        
        return self
    
    def predecir_2025(self):
        """Generar predicciones para 2025"""
        print("\n5. Generando predicciones para 2025...")
        
        datos_historicos = self.df[self.df['año'].isin([2022, 2023, 2024])].copy()
        datos_2025 = self.df[self.df['año'] == 2025].copy()
        
        X_2025_list = []
        departamentos_meses = []
        
        for dept in datos_2025['departamento'].unique():
            for mes in datos_2025['mes'].unique():
                registro_2025 = datos_2025[
                    (datos_2025['departamento'] == dept) & 
                    (datos_2025['mes'] == mes)
                ]
                
                if len(registro_2025) == 1:
                    registros_historicos = datos_historicos[
                        (datos_historicos['departamento'] == dept) & 
                        (datos_historicos['mes'] == mes)
                    ]
                    
                    if len(registros_historicos) >= 2:
                        feature_vector = []
                        
                        # Features históricos
                        for var in self.variables_objetivo_finales:
                            vals_historicos = registros_historicos[var].tail(2).values
                            if len(vals_historicos) >= 2:
                                feature_vector.extend(vals_historicos[-2:])
                            else:
                                mediana = registros_historicos[var].median()
                                feature_vector.extend([mediana, mediana])
                        
                        # Target encoding departamentos
                        for var in self.variables_objetivo_finales:
                            dept_enc_key = f'dept_enc_{var}'
                            if dept in self.dept_encodings[dept_enc_key]:
                                feature_vector.append(self.dept_encodings[dept_enc_key][dept])
                            else:
                                feature_vector.append(0)
                        
                        # Encoding cíclico meses
                        if mes in self.meses_map:
                            mes_num = self.meses_map[mes]
                            mes_sin = np.sin(2 * np.pi * mes_num / 12)
                            mes_cos = np.cos(2 * np.pi * mes_num / 12)
                            feature_vector.extend([mes_sin, mes_cos])
                        else:
                            feature_vector.extend([0, 0])
                        
                        # Año
                        feature_vector.append(2)
                        
                        X_2025_list.append(feature_vector)
                        departamentos_meses.append((dept, mes))
        
        X_2025 = pd.DataFrame(X_2025_list, columns=self.X_train.columns)
        X_2025 = X_2025.fillna(X_2025.median())
        
        X_2025_scaled = self.scaler_X.transform(X_2025)
        predicciones_2025_scaled = self.modelo.predict(X_2025_scaled)
        self.predicciones_2025 = self.scaler_y.inverse_transform(predicciones_2025_scaled)
        
        self.predicciones_2025_df = pd.DataFrame(
            self.predicciones_2025,
            columns=self.variables_objetivo_finales
        )
        
        departamentos = [dm[0] for dm in departamentos_meses]
        meses = [dm[1] for dm in departamentos_meses]
        
        self.predicciones_2025_df['departamento'] = departamentos
        self.predicciones_2025_df['mes'] = meses
        self.predicciones_2025_df['año'] = 2025
        
        print(f"   Predicciones generadas: {self.predicciones_2025_df.shape}")
        print(f"   Variables predichas: {len(self.variables_objetivo_finales)}")
        
        return self
    
    def guardar_resultados(self):
        """Guardar modelo y resultados"""
        print("\n6. Guardando resultados...")
        
        import os
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones', exist_ok=True)
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas', exist_ok=True)
        
        modelo_completo = {
            'modelo': self.modelo,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'dept_encodings': self.dept_encodings,
            'meses_map': self.meses_map,
            'variables_objetivo': self.variables_objetivo_finales
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos/elastic_net_modelo.pkl', 'wb') as f:
            pickle.dump(modelo_completo, f)
        
        self.predicciones_2025_df.to_csv(
            'd:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/elastic_net_predicciones_2025.csv', 
            index=False
        )
        
        metricas_completas = {
            'entrenamiento': self.metricas_entrenamiento,
            'resumen': {
                'rmse_entrenamiento': np.mean([m['RMSE'] for m in self.metricas_entrenamiento.values()]),
                'mae_entrenamiento': np.mean([m['MAE'] for m in self.metricas_entrenamiento.values()]),
                'r2_entrenamiento': np.mean([m['R2'] for m in self.metricas_entrenamiento.values()]),
                'variables_predichas': len(self.variables_objetivo_finales),
                'registros_entrenamiento': len(self.X_train),
                'registros_prediccion': len(self.predicciones_2025_df),
                'regularizacion': 'L1 + L2',
                'alpha': 0.1,
                'l1_ratio': 0.5
            }
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/elastic_net_metricas.json', 'w') as f:
            json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
        
        print("   Modelo guardado: modelos/elastic_net_modelo.pkl")
        print("   Predicciones: predicciones/elastic_net_predicciones_2025.csv")
        print("   Métricas: metricas/elastic_net_metricas.json")
        
        return self
    
    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo del modelo Elastic Net"""
        print("INICIANDO PIPELINE COMPLETO - MODELO ELASTIC NET")
        print("=" * 65)
        
        import os
        os.makedirs('d:/Tesis maestria/Tesis codigo/modelado/resultados/reportes', exist_ok=True)
        
        (self.cargar_datos_directamente()
         .preparar_datos_entrenamiento()
         .configurar_modelo()
         .entrenar_modelo()
         .predecir_2025()
         .guardar_resultados())
        
        print("\n" + "=" * 65)
        print("MODELO ELASTIC NET COMPLETADO EXITOSAMENTE")
        print("=" * 65)
        
        rmse_promedio = np.mean([m['RMSE'] for m in self.metricas_entrenamiento.values()])
        r2_promedio = np.mean([m['R2'] for m in self.metricas_entrenamiento.values()])
        
        print("RESUMEN FINAL:")
        print(f"- Variables predichas: {len(self.variables_objetivo_finales)}")
        print(f"- RMSE entrenamiento: {rmse_promedio:.4f}")
        print(f"- R² entrenamiento: {r2_promedio:.4f}")
        print(f"- Predicciones 2025: {len(self.predicciones_2025_df)} registros")
        print("- Regularización: L1 + L2 aplicada")
        
        return self

def main():
    """Función principal"""
    modelo_elastic = ModeloElasticNet()
    return modelo_elastic.ejecutar_pipeline_completo()

if __name__ == "__main__":
    modelo = main()

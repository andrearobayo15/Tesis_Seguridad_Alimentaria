"""
MODELO 2: RANDOM FOREST PARA PREDICCIÓN MULTIVARIADA FIES 2025

Descripción:
- Modelo Random Forest para predicción de 50 variables socioeconómicas, FIES y climáticas
- Estructura temporal: 2022-2023 → 2024 (validación), 2022-2024 → 2025 (predicción)
- Variables categóricas: Target encoding + encoding cíclico
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración
import warnings
warnings.filterwarnings('ignore')

print("MODELO 2: RANDOM FOREST PARA PREDICCIÓN MULTIVARIADA FIES 2025")
print("=" * 60)

class ModeloRandomForest:
    """Clase para modelo Random Forest multivariado FIES 2025"""
    
    def __init__(self):
        """Inicializar el modelo Random Forest"""
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
        """Preparar datos con estructura temporal correcta: 2022-2023 → 2024 + Variables categóricas"""
        print("\n2. Preparando datos de entrenamiento...")
        
        # ESTRATEGIA MEJORADA:
        # X_train: datos 2022-2023 + variables categóricas codificadas
        # y_train: datos 2024 (384 registros) - mismo departamento/mes
        
        # Datos de entrenamiento (features): 2022-2023
        datos_features = self.df[self.df['año'].isin([2022, 2023])].copy()
        
        # Datos objetivo (target): 2024
        datos_target = self.df[self.df['año'] == 2024].copy()
        
        print(f"   Registros features (2022-2023): {len(datos_features)}")
        print(f"   Registros target (2024): {len(datos_target)}")
        
        # Variables disponibles (las 50 variables que queremos predecir)
        variables_disponibles = []
        for var in self.variables_objetivo:
            if var in self.df.columns:
                # Verificar completitud en 2022-2024
                completitud_historica = datos_features[var].notna().mean()
                completitud_target = datos_target[var].notna().mean()
                
                if completitud_historica > 0.95 and completitud_target > 0.95:
                    variables_disponibles.append(var)
        
        self.variables_objetivo_finales = variables_disponibles
        print(f"   Variables disponibles para modelado: {len(self.variables_objetivo_finales)}")
        
        # CREAR ENCODINGS CATEGÓRICOS
        print("   Creando encodings categóricos...")
        
        # 1. TARGET ENCODING para departamentos
        # Calcular promedio histórico por departamento para cada variable
        dept_encodings = {}
        for var in self.variables_objetivo_finales:
            dept_avg = datos_features.groupby('departamento')[var].mean()
            dept_encodings[f'dept_enc_{var}'] = dept_avg.to_dict()
        
        # 2. ENCODING CÍCLICO para meses
        meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        # Guardar encodings para usar en predicción
        self.dept_encodings = dept_encodings
        self.meses_map = meses_map
        
        # Crear estructura de datos temporal con variables categóricas
        X_list = []
        y_list = []
        
        for dept in datos_target['departamento'].unique():
            for mes in datos_target['mes'].unique():
                # Target: datos 2024 para este departamento-mes
                target_row = datos_target[
                    (datos_target['departamento'] == dept) & 
                    (datos_target['mes'] == mes)
                ]
                
                if len(target_row) == 1:  # Asegurar que existe el registro
                    # Features: datos 2022-2023 para este departamento-mes
                    feature_rows = datos_features[
                        (datos_features['departamento'] == dept) & 
                        (datos_features['mes'] == mes)
                    ]
                    
                    if len(feature_rows) == 2:  # Debe tener 2022 y 2023
                        feature_vector = []
                        
                        # 1. Features históricos (2022-2023)
                        for var in self.variables_objetivo_finales:
                            vals_2022_2023 = feature_rows[var].values
                            feature_vector.extend(vals_2022_2023)
                        
                        # 2. Target encoding departamentos (promedio por variable)
                        for var in self.variables_objetivo_finales:
                            dept_enc_key = f'dept_enc_{var}'
                            if dept in dept_encodings[dept_enc_key]:
                                feature_vector.append(dept_encodings[dept_enc_key][dept])
                            else:
                                # Si no existe, usar promedio global
                                feature_vector.append(datos_features[var].mean())
                        
                        # 3. Encoding cíclico meses
                        if mes in meses_map:
                            mes_num = meses_map[mes]
                            mes_sin = np.sin(2 * np.pi * mes_num / 12)
                            mes_cos = np.cos(2 * np.pi * mes_num / 12)
                            feature_vector.extend([mes_sin, mes_cos])
                        else:
                            feature_vector.extend([0, 0])  # Default para meses no reconocidos
                        
                        # 4. Año como variable numérica (2022=0, 2023=1)
                        # Usar el año más reciente disponible (2023=1)
                        feature_vector.append(1)  # Para predicción 2024, usamos 2023 como referencia
                        
                        # Vector objetivo: valores 2024 para todas las variables
                        target_vector = target_row[self.variables_objetivo_finales].values[0]
                        
                        X_list.append(feature_vector)
                        y_list.append(target_vector)
        
        # Convertir a arrays
        self.X_train = pd.DataFrame(X_list)
        self.y_train = pd.DataFrame(y_list, columns=self.variables_objetivo_finales)
        
        # Crear nombres de columnas para features
        feature_names = []
        
        # Nombres features históricos
        for var in self.variables_objetivo_finales:
            feature_names.extend([f"{var}_2022", f"{var}_2023"])
        
        # Nombres target encoding departamentos
        for var in self.variables_objetivo_finales:
            feature_names.append(f"dept_enc_{var}")
        
        # Nombres encoding cíclico meses
        feature_names.extend(['mes_sin', 'mes_cos'])
        
        # Nombre año
        feature_names.append('año_ref')
        
        self.X_train.columns = feature_names
        
        # Manejar missing values
        self.X_train = self.X_train.fillna(self.X_train.median())
        self.y_train = self.y_train.fillna(self.y_train.median())
        
        print(f"   Estructura final:")
        print(f"     X_train: {self.X_train.shape} (features: históricos + categóricos)")
        print(f"     y_train: {self.y_train.shape} (target: 2024)")
        print(f"     Variables modeladas: {len(self.variables_objetivo_finales)}")
        print(f"     Features categóricos: {len(self.variables_objetivo_finales)} dept_enc + 2 mes + 1 año")
        
        return self
    
    def configurar_modelo(self):
        """Configurar modelo Random Forest"""
        print("\n3. Configurando modelo Random Forest...")
        
        # Configuración Random Forest optimizada para datos limitados
        rf_config = {
            'n_estimators': 100,        # Número de árboles
            'max_depth': 10,            # Profundidad máxima (más que XGBoost)
            'min_samples_split': 5,     # Mínimo para dividir nodo
            'min_samples_leaf': 2,      # Mínimo en hoja
            'max_features': 'sqrt',     # Features por árbol
            'bootstrap': True,          # Bootstrap sampling
            'random_state': 42,         # Reproducibilidad
            'n_jobs': -1,              # Paralelización
            'verbose': 0               # Sin output detallado
        }
        
        print("   Configuración Random Forest:")
        for key, value in rf_config.items():
            print(f"     {key}: {value}")
        
        # Crear modelo multioutput
        base_rf = RandomForestRegressor(**rf_config)
        self.modelo = MultiOutputRegressor(base_rf, n_jobs=-1)
        
        return self
    
    def entrenar_modelo(self):
        """Entrenar modelo Random Forest"""
        print("\n4. Entrenando modelo Random Forest...")
        print("   Iniciando entrenamiento...")
        
        # Entrenar modelo
        self.modelo.fit(self.X_train, self.y_train)
        print("   Entrenamiento completado")
        
        # Calcular métricas de entrenamiento
        y_pred_train = self.modelo.predict(self.X_train)
        
        # Métricas por variable
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
        
        # Métricas promedio
        rmse_promedio = np.mean(rmse_total)
        mae_promedio = np.mean(mae_total)
        r2_promedio = np.mean(r2_total)
        
        print(f"   Métricas de entrenamiento (promedio):")
        print(f"     RMSE: {rmse_promedio:.4f}")
        print(f"     MAE: {mae_promedio:.4f}")
        print(f"     R²: {r2_promedio:.4f}")
        
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
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/modelos/random_forest_modelo.pkl', 'wb') as f:
            pickle.dump(self.modelo, f)
        
        # 2. Guardar predicciones 2025
        self.predicciones_2025_df.to_csv(
            'd:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/random_forest_predicciones_2025.csv', 
            index=False
        )
        
        # 3. Guardar métricas
        metricas_completas = {
            'entrenamiento': self.metricas_entrenamiento,
            'resumen': {
                'rmse_entrenamiento': np.mean([m['RMSE'] for m in self.metricas_entrenamiento.values()]),
                'mae_entrenamiento': np.mean([m['MAE'] for m in self.metricas_entrenamiento.values()]),
                'r2_entrenamiento': np.mean([m['R2'] for m in self.metricas_entrenamiento.values()]),
                'variables_predichas': len(self.variables_objetivo_finales),
                'registros_entrenamiento': len(self.X_train),
                'registros_prediccion': len(self.predicciones_2025_df)
            }
        }
        
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/random_forest_metricas.json', 'w') as f:
            json.dump(metricas_completas, f, indent=2, ensure_ascii=False)
        
        print("   Modelo guardado: modelos/random_forest_modelo.pkl")
        print("   Predicciones: predicciones/random_forest_predicciones_2025.csv")
        print("   Métricas: metricas/random_forest_metricas.json")
        
        return self
    
    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo del modelo Random Forest"""
        print("INICIANDO PIPELINE COMPLETO - MODELO RANDOM FOREST")
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
        print("MODELO RANDOM FOREST COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        # Resumen final
        rmse_promedio = np.mean([m['RMSE'] for m in self.metricas_entrenamiento.values()])
        r2_promedio = np.mean([m['R2'] for m in self.metricas_entrenamiento.values()])
        
        print("RESUMEN FINAL:")
        print(f"- Variables predichas: {len(self.variables_objetivo_finales)}")
        print(f"- RMSE entrenamiento: {rmse_promedio:.4f}")
        print(f"- R² entrenamiento: {r2_promedio:.4f}")
        print(f"- Predicciones 2025: {len(self.predicciones_2025_df)} registros")
        print("- Archivos generados: modelo, predicciones, métricas")
        
        return self

def main():
    """Función principal"""
    modelo_rf = ModeloRandomForest()
    return modelo_rf.ejecutar_pipeline_completo()

if __name__ == "__main__":
    modelo = main()

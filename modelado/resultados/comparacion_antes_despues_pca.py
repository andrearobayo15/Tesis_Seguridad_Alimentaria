#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN ANTES Y DESPUÉS DE PCA - TODOS LOS MODELOS
=====================================================
Genera una tabla simple comparando el rendimiento de todos los modelos
antes y después de aplicar PCA para reducción de dimensionalidad.

Autor: Sistema de Análisis ML
Fecha: 2025-08-26
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def cargar_metricas_modelo(archivo_metricas):
    """Carga métricas de un archivo JSON"""
    try:
        with open(archivo_metricas, 'r', encoding='utf-8') as f:
            metricas = json.load(f)
        
        # Extraer métricas de validación cruzada temporal
        if 'validacion_cruzada_temporal' in metricas:
            cv_data = metricas['validacion_cruzada_temporal']
        elif 'validacion_cruzada' in metricas:
            cv_data = metricas['validacion_cruzada']
        else:
            return None
        
        return {
            'RMSE': cv_data.get('RMSE_mean', cv_data.get('rmse_mean', 0)),
            'MAE': cv_data.get('MAE_mean', cv_data.get('mae_mean', 0)),
            'R2': cv_data.get('R2_mean', cv_data.get('r2_mean', 0))
        }
    except Exception as e:
        print(f"Error cargando {archivo_metricas}: {e}")
        return None

def main():
    print("COMPARACIÓN ANTES Y DESPUÉS DE PCA - TODOS LOS MODELOS")
    print("=" * 60)
    
    # Rutas de archivos de métricas
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas")
    
    modelos_config = {
        'XGBoost': {
            'original': base_path / "xgboost_metricas.json",
            'pca': base_path / "xgboost_pca_metricas.json"
        },
        'Random Forest': {
            'original': base_path / "random_forest_metricas.json", 
            'pca': base_path / "random_forest_pca_metricas.json"
        },
        'SVM': {
            'original': base_path / "svm_metricas.json",
            'pca': base_path / "svm_pca_metricas.json"
        },
        'Elastic Net': {
            'original': base_path / "elastic_net_original_fies_metricas.json",
            'pca': base_path / "elastic_net_pca_metricas.json"
        }
    }
    
    # Cargar todas las métricas
    resultados = []
    
    for modelo_nombre, archivos in modelos_config.items():
        print(f"\nCargando métricas para {modelo_nombre}...")
        
        # Cargar métricas originales
        metricas_original = cargar_metricas_modelo(archivos['original'])
        metricas_pca = cargar_metricas_modelo(archivos['pca'])
        
        if metricas_original and metricas_pca:
            # Calcular diferencias (mejora/empeoramiento)
            diff_rmse = metricas_pca['RMSE'] - metricas_original['RMSE']
            diff_mae = metricas_pca['MAE'] - metricas_original['MAE'] 
            diff_r2 = metricas_pca['R2'] - metricas_original['R2']
            
            resultados.append({
                'Modelo': modelo_nombre,
                'RMSE_Original': metricas_original['RMSE'],
                'RMSE_PCA': metricas_pca['RMSE'],
                'RMSE_Diferencia': diff_rmse,
                'MAE_Original': metricas_original['MAE'],
                'MAE_PCA': metricas_pca['MAE'],
                'MAE_Diferencia': diff_mae,
                'R2_Original': metricas_original['R2'],
                'R2_PCA': metricas_pca['R2'],
                'R2_Diferencia': diff_r2
            })
            print(f"  OK {modelo_nombre}: Original y PCA cargados")
        else:
            print(f"  X {modelo_nombre}: Error cargando métricas")
    
    if not resultados:
        print("No se pudieron cargar métricas de ningún modelo")
        return
    
    # Crear DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Crear tabla de comparación principal
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA: ANTES vs DESPUÉS DE PCA")
    print("=" * 80)
    
    tabla_comparacion = pd.DataFrame({
        'Modelo': df_resultados['Modelo'],
        'RMSE Antes': df_resultados['RMSE_Original'].round(4),
        'RMSE Después': df_resultados['RMSE_PCA'].round(4),
        'MAE Antes': df_resultados['MAE_Original'].round(4),
        'MAE Después': df_resultados['MAE_PCA'].round(4),
        'R2 Antes': df_resultados['R2_Original'].round(4),
        'R2 Despues': df_resultados['R2_PCA'].round(4)
    })
    
    print(tabla_comparacion.to_string(index=False))
    
    # Tabla de diferencias (mejoras/empeoramientos)
    print("\n" + "=" * 60)
    print("CAMBIOS CON PCA (Valores negativos = mejora para RMSE/MAE)")
    print("=" * 60)
    
    tabla_diferencias = pd.DataFrame({
        'Modelo': df_resultados['Modelo'],
        'Cambio RMSE': df_resultados['RMSE_Diferencia'].round(4),
        'Cambio MAE': df_resultados['MAE_Diferencia'].round(4),
        'Cambio R2': df_resultados['R2_Diferencia'].round(4)
    })
    
    print(tabla_diferencias.to_string(index=False))
    
    # Análisis de resultados
    print("\n" + "=" * 60)
    print("ANÁLISIS DE RESULTADOS")
    print("=" * 60)
    
    for _, row in df_resultados.iterrows():
        modelo = row['Modelo']
        
        # Determinar si PCA mejoró o empeoró
        rmse_mejor = row['RMSE_Diferencia'] < 0
        mae_mejor = row['MAE_Diferencia'] < 0
        r2_mejor = row['R2_Diferencia'] > 0
        
        mejoras = sum([rmse_mejor, mae_mejor, r2_mejor])
        
        if mejoras >= 2:
            resultado = "MEJORA"
        elif mejoras == 1:
            resultado = "MIXTO"
        else:
            resultado = "EMPEORA"
        
        print(f"{modelo:15}: {resultado}")
        print(f"  - RMSE: {row['RMSE_Original']:.4f} -> {row['RMSE_PCA']:.4f} ({row['RMSE_Diferencia']:+.4f})")
        print(f"  - MAE:  {row['MAE_Original']:.4f} -> {row['MAE_PCA']:.4f} ({row['MAE_Diferencia']:+.4f})")
        print(f"  - R2:   {row['R2_Original']:.4f} -> {row['R2_PCA']:.4f} ({row['R2_Diferencia']:+.4f})")
        print()
    
    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE
    x_pos = range(len(df_resultados))
    axes[0].bar([i-0.2 for i in x_pos], df_resultados['RMSE_Original'], 
                width=0.4, label='Antes PCA', alpha=0.7, color='skyblue')
    axes[0].bar([i+0.2 for i in x_pos], df_resultados['RMSE_PCA'], 
                width=0.4, label='Después PCA', alpha=0.7, color='orange')
    axes[0].set_title('RMSE: Antes vs Después de PCA')
    axes[0].set_ylabel('RMSE')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(df_resultados['Modelo'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].bar([i-0.2 for i in x_pos], df_resultados['MAE_Original'], 
                width=0.4, label='Antes PCA', alpha=0.7, color='skyblue')
    axes[1].bar([i+0.2 for i in x_pos], df_resultados['MAE_PCA'], 
                width=0.4, label='Después PCA', alpha=0.7, color='orange')
    axes[1].set_title('MAE: Antes vs Después de PCA')
    axes[1].set_ylabel('MAE')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(df_resultados['Modelo'], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].bar([i-0.2 for i in x_pos], df_resultados['R2_Original'], 
                width=0.4, label='Antes PCA', alpha=0.7, color='skyblue')
    axes[2].bar([i+0.2 for i in x_pos], df_resultados['R2_PCA'], 
                width=0.4, label='Después PCA', alpha=0.7, color='orange')
    axes[2].set_title('R2: Antes vs Despues de PCA')
    axes[2].set_ylabel('R2')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(df_resultados['Modelo'], rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar visualización
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    plt.savefig(output_path / "comparacion_antes_despues_pca.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar tablas en CSV
    tabla_comparacion.to_csv(output_path / "tabla_antes_despues_pca.csv", 
                            index=False, encoding='utf-8')
    tabla_diferencias.to_csv(output_path / "diferencias_pca.csv", 
                            index=False, encoding='utf-8')
    
    print("=" * 60)
    print("COMPARACIÓN COMPLETADA")
    print("=" * 60)
    print("Archivos generados:")
    print("- comparacion_antes_despues_pca.png")
    print("- tabla_antes_despues_pca.csv")
    print("- diferencias_pca.csv")

if __name__ == "__main__":
    main()

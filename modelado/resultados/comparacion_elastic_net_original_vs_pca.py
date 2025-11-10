#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN ELASTIC NET: VARIABLES ORIGINALES VS COMPONENTES PCA
================================================================

Compara el rendimiento del modelo Elastic Net usando:
1. Variables originales (sin PCA)
2. Componentes PCA (7 componentes)

Análisis incluye:
- Métricas de validación cruzada temporal
- Comparación de predicciones 2025
- Visualizaciones comparativas
- Reporte científico en JSON

Autor: Sistema de Análisis ML
Fecha: 2025
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def cargar_metricas_modelos():
    """Carga las métricas de ambos modelos Elastic Net"""
    
    # Rutas de archivos
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    
    # Cargar métricas
    metricas_original_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/elastic_net_original_fies_metricas.json")
    metricas_pca_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/elastic_net_pca_metricas.json")
    
    metricas = {}
    
    # Cargar métricas originales
    if metricas_original_path.exists():
        with open(metricas_original_path, 'r', encoding='utf-8') as f:
            metricas['original'] = json.load(f)
        print("OK Metricas Elastic Net original cargadas")
    else:
        print("AVISO Metricas Elastic Net original no encontradas")
        metricas['original'] = None
    
    # Cargar métricas PCA
    if metricas_pca_path.exists():
        with open(metricas_pca_path, 'r', encoding='utf-8') as f:
            metricas['pca'] = json.load(f)
        print("OK Metricas Elastic Net PCA cargadas")
    else:
        print("AVISO Metricas Elastic Net PCA no encontradas")
        metricas['pca'] = None
    
    return metricas

def cargar_predicciones_modelos():
    """Carga las predicciones de ambos modelos Elastic Net"""
    
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    
    # Predicciones Elastic Net original
    pred_original_path = base_path / "predicciones" / "elastic_net_original_fies_predicciones_2025.csv"
    
    # Predicciones Elastic Net PCA
    pred_pca_path = base_path / "predicciones" / "elastic_net_pca_predicciones_2025.csv"
    
    predicciones = {}
    
    # Cargar predicciones originales
    if pred_original_path.exists():
        predicciones['original'] = pd.read_csv(pred_original_path)
        print(f"OK Predicciones Elastic Net original: {len(predicciones['original'])} registros")
    else:
        print("AVISO Predicciones Elastic Net original no encontradas")
        predicciones['original'] = None
    
    # Cargar predicciones PCA
    if pred_pca_path.exists():
        predicciones['pca'] = pd.read_csv(pred_pca_path)
        print(f"OK Predicciones Elastic Net PCA: {len(predicciones['pca'])} registros")
    else:
        print("AVISO Predicciones Elastic Net PCA no encontradas")
        predicciones['pca'] = None
    
    return predicciones

def crear_tabla_comparativa(metricas):
    """Crea tabla comparativa de métricas"""
    
    if not metricas['original'] or not metricas['pca']:
        print("AVISO No se pueden comparar metricas - datos faltantes")
        return None
    
    # Extraer métricas de validación cruzada temporal
    def extraer_metricas_cv(metricas_dict):
        if 'validacion_cruzada_temporal' in metricas_dict:
            cv_data = metricas_dict['validacion_cruzada_temporal']
        elif 'validacion_cruzada' in metricas_dict:
            cv_data = metricas_dict['validacion_cruzada']
        else:
            return {'RMSE': 0, 'MAE': 0, 'R2': 0, 'RMSE_std': 0, 'MAE_std': 0, 'R2_std': 0}
        
        return {
            'RMSE': cv_data.get('RMSE_mean', cv_data.get('rmse_mean', 0)),
            'MAE': cv_data.get('MAE_mean', cv_data.get('mae_mean', 0)),
            'R2': cv_data.get('R2_mean', cv_data.get('r2_mean', 0)),
            'RMSE_std': cv_data.get('RMSE_std', cv_data.get('rmse_std', 0)),
            'MAE_std': cv_data.get('MAE_std', cv_data.get('mae_std', 0)),
            'R2_std': cv_data.get('R2_std', cv_data.get('r2_std', 0))
        }
    
    cv_original = extraer_metricas_cv(metricas['original'])
    cv_pca = extraer_metricas_cv(metricas['pca'])
    
    # Crear DataFrame comparativo
    comparacion = pd.DataFrame({
        'Métrica': ['RMSE', 'MAE', 'R²'],
        'Elastic Net Original': [
            f"{cv_original['RMSE']:.4f} ± {cv_original['RMSE_std']:.4f}",
            f"{cv_original['MAE']:.4f} ± {cv_original['MAE_std']:.4f}",
            f"{cv_original['R2']:.4f} ± {cv_original['R2_std']:.4f}"
        ],
        'Elastic Net PCA': [
            f"{cv_pca['RMSE']:.4f} ± {cv_pca['RMSE_std']:.4f}",
            f"{cv_pca['MAE']:.4f} ± {cv_pca['MAE_std']:.4f}",
            f"{cv_pca['R2']:.4f} ± {cv_pca['R2_std']:.4f}"
        ]
    })
    
    print("\nTABLA COMPARATIVA - ELASTIC NET ORIGINAL VS PCA")
    print("=" * 60)
    print(comparacion.to_string(index=False))
    
    return comparacion

def visualizar_metricas_comparativas(metricas):
    """Crea visualizaciones comparativas de métricas"""
    
    if not metricas['original'] or not metricas['pca']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('COMPARACIÓN ELASTIC NET: ORIGINAL vs PCA\nMétricas de Validación Cruzada Temporal', 
                 fontsize=16, fontweight='bold')
    
    # Extraer datos
    cv_original = metricas['original'].get('validacion_cruzada', {})
    cv_pca = metricas['pca'].get('validacion_cruzada', {})
    
    modelos = ['Original', 'PCA']
    
    # RMSE
    rmse_values = [cv_original.get('rmse_mean', 0), cv_pca.get('rmse_mean', 0)]
    rmse_errors = [cv_original.get('rmse_std', 0), cv_pca.get('rmse_std', 0)]
    
    axes[0,0].bar(modelos, rmse_values, yerr=rmse_errors, capsize=5, 
                  color=['lightcoral', 'lightblue'], alpha=0.7)
    axes[0,0].set_title('RMSE (menor es mejor)')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].grid(True, alpha=0.3)
    
    # MAE
    mae_values = [cv_original.get('mae_mean', 0), cv_pca.get('mae_mean', 0)]
    mae_errors = [cv_original.get('mae_std', 0), cv_pca.get('mae_std', 0)]
    
    axes[0,1].bar(modelos, mae_values, yerr=mae_errors, capsize=5,
                  color=['lightcoral', 'lightblue'], alpha=0.7)
    axes[0,1].set_title('MAE (menor es mejor)')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].grid(True, alpha=0.3)
    
    # R²
    r2_values = [cv_original.get('r2_mean', 0), cv_pca.get('r2_mean', 0)]
    r2_errors = [cv_original.get('r2_std', 0), cv_pca.get('r2_std', 0)]
    
    axes[1,0].bar(modelos, r2_values, yerr=r2_errors, capsize=5,
                  color=['lightcoral', 'lightblue'], alpha=0.7)
    axes[1,0].set_title('R² (mayor es mejor)')
    axes[1,0].set_ylabel('R²')
    axes[1,0].grid(True, alpha=0.3)
    
    # Comparación de estabilidad (desviación estándar)
    estabilidad = [
        np.mean([cv_original.get('rmse_std', 0), cv_original.get('mae_std', 0), cv_original.get('r2_std', 0)]),
        np.mean([cv_pca.get('rmse_std', 0), cv_pca.get('mae_std', 0), cv_pca.get('r2_std', 0)])
    ]
    
    axes[1,1].bar(modelos, estabilidad, color=['lightcoral', 'lightblue'], alpha=0.7)
    axes[1,1].set_title('Estabilidad Promedio (menor es mejor)')
    axes[1,1].set_ylabel('Desviación Estándar Promedio')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar visualización
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    plt.savefig(output_path / "comparacion_elastic_net_metricas.png", dpi=300, bbox_inches='tight')
    print("OK Visualizacion de metricas guardada")
    
    plt.show()

def comparar_predicciones(predicciones):
    """Compara las predicciones de ambos modelos"""
    
    if predicciones['original'] is None or predicciones['pca'] is None:
        return
    
    # Merge de predicciones por departamento y mes
    pred_merged = predicciones['original'].merge(
        predicciones['pca'], 
        on=['departamento', 'mes'], 
        suffixes=('_original', '_pca')
    )
    
    print(f"\nCOMPARACIÓN DE PREDICCIONES 2025")
    print("=" * 50)
    
    # Variables FIES
    variables_fies = ['FIES_moderado_grave', 'FIES_grave']
    
    for var in variables_fies:
        if f"{var}_original" in pred_merged.columns and f"{var}_pca" in pred_merged.columns:
            original_vals = pred_merged[f"{var}_original"]
            pca_vals = pred_merged[f"{var}_pca"]
            
            print(f"\n{var}:")
            print(f"  Original - Rango: {original_vals.min():.2f} - {original_vals.max():.2f}, Media: {original_vals.mean():.2f}")
            print(f"  PCA      - Rango: {pca_vals.min():.2f} - {pca_vals.max():.2f}, Media: {pca_vals.mean():.2f}")
            
            # Correlación entre predicciones
            correlacion = original_vals.corr(pca_vals)
            print(f"  Correlación entre modelos: {correlacion:.4f}")
    
    return pred_merged

def visualizar_predicciones_comparativas(pred_merged):
    """Visualiza comparación de predicciones"""
    
    if pred_merged is None:
        return
    
    variables_fies = ['FIES_moderado_grave', 'FIES_grave']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('COMPARACIÓN PREDICCIONES ELASTIC NET 2025: ORIGINAL vs PCA', 
                 fontsize=14, fontweight='bold')
    
    for i, var in enumerate(variables_fies):
        if f"{var}_original" in pred_merged.columns and f"{var}_pca" in pred_merged.columns:
            
            x = pred_merged[f"{var}_original"]
            y = pred_merged[f"{var}_pca"]
            
            # Scatter plot
            axes[i].scatter(x, y, alpha=0.6, s=30)
            
            # Línea de referencia (x=y)
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Línea perfecta')
            
            axes[i].set_xlabel(f'{var} - Original')
            axes[i].set_ylabel(f'{var} - PCA')
            axes[i].set_title(f'{var}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Correlación
            corr = x.corr(y)
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    plt.savefig(output_path / "comparacion_elastic_net_predicciones.png", dpi=300, bbox_inches='tight')
    print("OK Visualizacion de predicciones guardada")
    
    plt.show()

def generar_reporte_cientifico(metricas, comparacion_df, pred_merged):
    """Genera reporte científico en JSON"""
    
    reporte = {
        "titulo": "Comparación Elastic Net: Variables Originales vs Componentes PCA",
        "fecha_analisis": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metodologia": {
            "modelo": "Elastic Net Regression",
            "validacion": "Validación Cruzada Temporal (TimeSeriesSplit, 5 folds)",
            "metricas": ["RMSE", "MAE", "R²"],
            "variables_objetivo": ["FIES_moderado_grave", "FIES_grave"],
            "periodo_entrenamiento": "2022-2024",
            "periodo_prediccion": "2025"
        },
        "configuraciones": {
            "elastic_net_original": {
                "descripcion": "Modelo usando todas las variables originales",
                "features": "Variables climáticas, socioeconómicas, geográficas y temporales",
                "dimensionalidad": "Alta (múltiples variables correlacionadas)"
            },
            "elastic_net_pca": {
                "descripcion": "Modelo usando 7 componentes principales",
                "features": "7 componentes PCA + variables categóricas",
                "dimensionalidad": "Reducida (7 componentes + encodings)"
            }
        }
    }
    
    # Agregar resultados si están disponibles
    if metricas['original'] and metricas['pca']:
        cv_original = metricas['original'].get('validacion_cruzada', {})
        cv_pca = metricas['pca'].get('validacion_cruzada', {})
        
        reporte["resultados"] = {
            "metricas_validacion_cruzada": {
                "elastic_net_original": {
                    "rmse": f"{cv_original.get('rmse_mean', 0):.4f} ± {cv_original.get('rmse_std', 0):.4f}",
                    "mae": f"{cv_original.get('mae_mean', 0):.4f} ± {cv_original.get('mae_std', 0):.4f}",
                    "r2": f"{cv_original.get('r2_mean', 0):.4f} ± {cv_original.get('r2_std', 0):.4f}"
                },
                "elastic_net_pca": {
                    "rmse": f"{cv_pca.get('rmse_mean', 0):.4f} ± {cv_pca.get('rmse_std', 0):.4f}",
                    "mae": f"{cv_pca.get('mae_mean', 0):.4f} ± {cv_pca.get('mae_std', 0):.4f}",
                    "r2": f"{cv_pca.get('r2_mean', 0):.4f} ± {cv_pca.get('r2_std', 0):.4f}"
                }
            }
        }
        
        # Determinar modelo superior
        rmse_original = cv_original.get('rmse_mean', float('inf'))
        rmse_pca = cv_pca.get('rmse_mean', float('inf'))
        r2_original = cv_original.get('r2_mean', 0)
        r2_pca = cv_pca.get('r2_mean', 0)
        
        if rmse_pca < rmse_original and r2_pca > r2_original:
            modelo_superior = "PCA"
        elif rmse_original < rmse_pca and r2_original > r2_pca:
            modelo_superior = "Original"
        else:
            modelo_superior = "Mixto"
        
        reporte["conclusiones"] = {
            "modelo_superior": modelo_superior,
            "justificacion_tecnica": {
                "pca": "Reduce dimensionalidad y multicolinealidad, mejora generalización",
                "original": "Mantiene interpretabilidad directa de variables originales",
                "consideraciones": "PCA puede perder información específica pero mejora estabilidad"
            },
            "recomendacion_uso": {
                "pca": "Recomendado para predicción robusta con datos correlacionados",
                "original": "Útil cuando se requiere interpretabilidad directa de variables"
            }
        }
    
    # Guardar reporte
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    with open(output_path / "comparacion_elastic_net_reporte.json", 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print("OK Reporte cientifico guardado")
    return reporte

def main():
    """Función principal de comparación"""
    
    print("COMPARACIÓN ELASTIC NET: VARIABLES ORIGINALES VS COMPONENTES PCA")
    print("=" * 70)
    
    # 1. Cargar métricas
    print("\n1. Cargando métricas de modelos...")
    metricas = cargar_metricas_modelos()
    
    # 2. Cargar predicciones
    print("\n2. Cargando predicciones...")
    predicciones = cargar_predicciones_modelos()
    
    # 3. Crear tabla comparativa
    print("\n3. Creando tabla comparativa...")
    comparacion_df = crear_tabla_comparativa(metricas)
    
    # 4. Visualizar métricas
    print("\n4. Generando visualizaciones de métricas...")
    visualizar_metricas_comparativas(metricas)
    
    # 5. Comparar predicciones
    print("\n5. Comparando predicciones...")
    pred_merged = comparar_predicciones(predicciones)
    
    # 6. Visualizar predicciones
    print("\n6. Visualizando comparación de predicciones...")
    visualizar_predicciones_comparativas(pred_merged)
    
    # 7. Generar reporte
    print("\n7. Generando reporte científico...")
    reporte = generar_reporte_cientifico(metricas, comparacion_df, pred_merged)
    
    print("\n" + "=" * 70)
    print("COMPARACIÓN ELASTIC NET COMPLETADA")
    print("=" * 70)
    print("Archivos generados:")
    print("- comparacion_elastic_net_metricas.png")
    print("- comparacion_elastic_net_predicciones.png")  
    print("- comparacion_elastic_net_reporte.json")
    
    return metricas, predicciones, reporte

if __name__ == "__main__":
    metricas, predicciones, reporte = main()

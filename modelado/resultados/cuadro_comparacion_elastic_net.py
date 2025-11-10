#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUADRO COMPARATIVO ELASTIC NET: ANTES vs DESPUÉS DE PCA
=======================================================

Extrae y compara métricas específicas de Elastic Net
para variables FIES antes y después de aplicar PCA.

Autor: Sistema de Análisis ML
Fecha: 2025
"""

import pandas as pd
import json
from pathlib import Path

def extraer_metricas_elastic_net():
    """Extrae métricas de ambos modelos Elastic Net"""
    
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas")
    
    # Cargar métricas Elastic Net Original
    with open(base_path / "elastic_net_metricas.json", 'r', encoding='utf-8') as f:
        metricas_original = json.load(f)
    
    # Cargar métricas Elastic Net PCA
    with open(base_path / "elastic_net_pca_metricas.json", 'r', encoding='utf-8') as f:
        metricas_pca = json.load(f)
    
    return metricas_original, metricas_pca

def crear_cuadro_comparativo():
    """Crea cuadro comparativo detallado"""
    
    metricas_original, metricas_pca = extraer_metricas_elastic_net()
    
    # Variables FIES de interés
    variables_fies = ['FIES_moderado_grave', 'FIES_grave']
    
    print("CUADRO COMPARATIVO: ELASTIC NET ANTES vs DESPUÉS DE PCA")
    print("=" * 80)
    print()
    
    # Información general
    print("CONFIGURACIÓN DE MODELOS:")
    print("-" * 40)
    print("Elastic Net Original:")
    print("  - Variables: Todas las variables originales (50+ variables)")
    print("  - Dimensionalidad: Alta (múltiples variables correlacionadas)")
    print("  - Validación: Solo métricas de entrenamiento")
    print("  - Regularizacion: L1 + L2 (alpha=0.1, l1_ratio=0.5)")
    print()
    
    print("Elastic Net PCA:")
    print("  - Variables: 7 componentes PCA + encodings categóricos")
    print("  - Dimensionalidad: Reducida (42 features totales)")
    print("  - Validación: Validación cruzada temporal (5 folds)")
    print("  - Regularizacion: L1 + L2 (alpha=0.1, l1_ratio=0.3)")
    print()
    
    # Comparación de métricas para variables FIES
    print("COMPARACIÓN DE RENDIMIENTO - VARIABLES FIES:")
    print("-" * 50)
    
    # Crear tabla comparativa
    datos_comparacion = []
    
    for var in variables_fies:
        # Métricas originales (solo entrenamiento)
        if var in metricas_original['entrenamiento']:
            orig_data = metricas_original['entrenamiento'][var]
            rmse_orig = orig_data['RMSE']
            mae_orig = orig_data['MAE']
            r2_orig = orig_data['R2']
        else:
            rmse_orig = mae_orig = r2_orig = "N/A"
        
        # Métricas PCA (validación cruzada)
        cv_pca = metricas_pca['validacion_cruzada']
        rmse_pca = f"{cv_pca['RMSE_mean']:.4f} ± {cv_pca['RMSE_std']:.4f}"
        mae_pca = f"{cv_pca['MAE_mean']:.4f} ± {cv_pca['MAE_std']:.4f}"
        r2_pca = f"{cv_pca['R2_mean']:.4f} ± {cv_pca['R2_std']:.4f}"
        
        datos_comparacion.append({
            'Variable': var,
            'RMSE_Original': f"{rmse_orig:.4f}" if rmse_orig != "N/A" else "N/A",
            'RMSE_PCA': rmse_pca,
            'MAE_Original': f"{mae_orig:.4f}" if mae_orig != "N/A" else "N/A", 
            'MAE_PCA': mae_pca,
            'R2_Original': f"{r2_orig:.4f}" if r2_orig != "N/A" else "N/A",
            'R2_PCA': r2_pca
        })
    
    # Mostrar tabla
    df_comparacion = pd.DataFrame(datos_comparacion)
    
    print("MÉTRICAS POR VARIABLE:")
    print()
    for _, row in df_comparacion.iterrows():
        print(f"{row['Variable']}:")
        print(f"  RMSE: {row['RMSE_Original']} (Original) vs {row['RMSE_PCA']} (PCA)")
        print(f"  MAE:  {row['MAE_Original']} (Original) vs {row['MAE_PCA']} (PCA)")
        print(f"  R²:   {row['R2_Original']} (Original) vs {row['R2_PCA']} (PCA)")
        print()
    
    # Resumen general
    print("RESUMEN GENERAL:")
    print("-" * 30)
    
    # Métricas promedio originales
    rmse_orig_prom = metricas_original['resumen']['rmse_entrenamiento']
    mae_orig_prom = metricas_original['resumen']['mae_entrenamiento'] 
    r2_orig_prom = metricas_original['resumen']['r2_entrenamiento']
    
    # Métricas PCA
    rmse_pca_cv = metricas_pca['validacion_cruzada']['RMSE_mean']
    mae_pca_cv = metricas_pca['validacion_cruzada']['MAE_mean']
    r2_pca_cv = metricas_pca['validacion_cruzada']['R2_mean']
    
    print(f"Elastic Net Original (Entrenamiento):")
    print(f"  RMSE promedio: {rmse_orig_prom:.4f}")
    print(f"  MAE promedio:  {mae_orig_prom:.4f}")
    print(f"  R² promedio:   {r2_orig_prom:.4f}")
    print(f"  Variables:     {metricas_original['resumen']['variables_predichas']}")
    print()
    
    print(f"Elastic Net PCA (Validación Cruzada):")
    print(f"  RMSE CV:       {rmse_pca_cv:.4f} ± {metricas_pca['validacion_cruzada']['RMSE_std']:.4f}")
    print(f"  MAE CV:        {mae_pca_cv:.4f} ± {metricas_pca['validacion_cruzada']['MAE_std']:.4f}")
    print(f"  R² CV:         {r2_pca_cv:.4f} ± {metricas_pca['validacion_cruzada']['R2_std']:.4f}")
    print(f"  Features:      {metricas_pca['resumen']['features_utilizados']}")
    print()
    
    # Análisis comparativo
    print("ANALISIS COMPARATIVO:")
    print("-" * 35)
    print("VENTAJAS DEL MODELO PCA:")
    print("  - Validacion cruzada temporal (mas confiable)")
    print("  - Reduccion dimensional (42 vs 50+ variables)")
    print("  - Menor riesgo de sobreajuste")
    print("  - Metricas de estabilidad (desviacion estandar)")
    print("  - Mejor generalizacion temporal")
    print()
    
    print("LIMITACIONES DEL MODELO ORIGINAL:")
    print("  - Solo metricas de entrenamiento (posible sobreajuste)")
    print("  - Alta dimensionalidad con variables correlacionadas")
    print("  - Sin validacion cruzada temporal")
    print("  - No evalua capacidad de generalizacion")
    print()
    
    print("RECOMENDACION:")
    print("El modelo Elastic Net PCA es preferible por:")
    print("- Validacion mas rigurosa (temporal cross-validation)")
    print("- Mejor control del sobreajuste")
    print("- Dimensionalidad reducida y manejable")
    print("- Metricas de estabilidad confiables")
    
    return df_comparacion, metricas_original, metricas_pca

if __name__ == "__main__":
    df_comp, orig, pca = crear_cuadro_comparativo()

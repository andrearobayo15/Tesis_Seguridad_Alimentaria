"""
COMPARACIÓN SVM: VARIABLES ORIGINALES VS COMPONENTES PCA
========================================================
Script para comparar el rendimiento de SVM usando variables originales
vs SVM usando componentes PCA para predicción FIES.

Autor: Análisis PCA - Tesis Maestría
Fecha: 2025-08-26
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def cargar_metricas_svm():
    """Cargar métricas de ambos modelos SVM"""
    print("1. Cargando métricas de modelos SVM...")
    
    # Cargar métricas SVM original
    try:
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/svm_metricas.json', 'r') as f:
            metricas_original = json.load(f)
        print("   Métricas SVM original cargadas")
    except FileNotFoundError:
        print("   No se encontraron métricas del modelo SVM original")
        return None, None
    
    # Cargar métricas SVM PCA
    try:
        with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas/svm_pca_metricas.json', 'r') as f:
            metricas_pca = json.load(f)
        print("   Métricas SVM PCA cargadas")
    except FileNotFoundError:
        print("   No se encontraron métricas del modelo SVM PCA")
        return None, None
    
    return metricas_original, metricas_pca

def extraer_metricas_validacion_svm(metricas_original, metricas_pca):
    """Extraer métricas de validación cruzada de ambos modelos SVM"""
    print("\n2. Extrayendo métricas de validación cruzada...")
    
    # Métricas modelo original
    if 'validacion_cruzada' in metricas_original:
        original_cv = metricas_original['validacion_cruzada']
    else:
        # Si no tiene validación cruzada, usar métricas de entrenamiento como referencia
        original_cv = {
            'RMSE_mean': metricas_original['resumen']['rmse_entrenamiento'],
            'MAE_mean': metricas_original['resumen']['mae_entrenamiento'],
            'R2_mean': metricas_original['resumen']['r2_entrenamiento'],
            'RMSE_std': 0,
            'MAE_std': 0,
            'R2_std': 0
        }
        print("   Modelo original sin validación cruzada, usando métricas de entrenamiento")
    
    # Métricas modelo PCA
    pca_cv = metricas_pca['validacion_cruzada']
    
    print(f"   SVM Original - RMSE: {original_cv['RMSE_mean']:.4f}, R²: {original_cv['R2_mean']:.4f}")
    print(f"   SVM PCA - RMSE: {pca_cv['RMSE_mean']:.4f}, R²: {pca_cv['R2_mean']:.4f}")
    
    return original_cv, pca_cv

def comparar_metricas_svm(original_cv, pca_cv):
    """Comparar métricas entre modelos SVM"""
    print("\n3. Comparando métricas de rendimiento...")
    
    # Calcular mejoras
    mejora_rmse = ((original_cv['RMSE_mean'] - pca_cv['RMSE_mean']) / original_cv['RMSE_mean']) * 100
    mejora_mae = ((original_cv['MAE_mean'] - pca_cv['MAE_mean']) / original_cv['MAE_mean']) * 100
    mejora_r2 = pca_cv['R2_mean'] - original_cv['R2_mean']
    
    print(f"   MEJORAS CON PCA:")
    print(f"     RMSE: {mejora_rmse:+.1f}% ({'mejor' if mejora_rmse > 0 else 'peor'})")
    print(f"     MAE:  {mejora_mae:+.1f}% ({'mejor' if mejora_mae > 0 else 'peor'})")
    print(f"     R²:   {mejora_r2:+.4f} puntos ({'mejor' if mejora_r2 > 0 else 'peor'})")
    
    return {
        'mejora_rmse_pct': mejora_rmse,
        'mejora_mae_pct': mejora_mae,
        'mejora_r2_puntos': mejora_r2,
        'original': original_cv,
        'pca': pca_cv
    }

def cargar_predicciones_svm():
    """Cargar predicciones de ambos modelos SVM"""
    print("\n4. Cargando predicciones 2025...")
    
    # Predicciones SVM original
    try:
        pred_original = pd.read_csv('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/svm_predicciones_2025.csv')
        print(f"   Predicciones SVM original: {len(pred_original)} registros")
    except FileNotFoundError:
        print("   No se encontraron predicciones SVM original")
        return None, None
    
    # Predicciones SVM PCA
    try:
        pred_pca = pd.read_csv('d:/Tesis maestria/Tesis codigo/modelado/resultados/predicciones/svm_pca_predicciones_2025.csv')
        print(f"   Predicciones SVM PCA: {len(pred_pca)} registros")
    except FileNotFoundError:
        print("   No se encontraron predicciones SVM PCA")
        return None, None
    
    return pred_original, pred_pca

def comparar_predicciones_svm(pred_original, pred_pca):
    """Comparar distribuciones de predicciones SVM"""
    print("\n5. Comparando distribuciones de predicciones...")
    
    # Variables a comparar
    variables = ['FIES_moderado_grave', 'FIES_grave']
    
    comparacion = {}
    for var in variables:
        if var in pred_original.columns and var in pred_pca.columns:
            orig_stats = {
                'min': pred_original[var].min(),
                'max': pred_original[var].max(),
                'mean': pred_original[var].mean(),
                'std': pred_original[var].std()
            }
            
            pca_stats = {
                'min': pred_pca[var].min(),
                'max': pred_pca[var].max(),
                'mean': pred_pca[var].mean(),
                'std': pred_pca[var].std()
            }
            
            comparacion[var] = {
                'original': orig_stats,
                'pca': pca_stats
            }
            
            print(f"   {var}:")
            print(f"     Original - Rango: [{orig_stats['min']:.2f}, {orig_stats['max']:.2f}], Media: {orig_stats['mean']:.2f}")
            print(f"     PCA      - Rango: [{pca_stats['min']:.2f}, {pca_stats['max']:.2f}], Media: {pca_stats['mean']:.2f}")
    
    return comparacion

def crear_visualizaciones_svm(comparacion_metricas, comparacion_predicciones, pred_original, pred_pca):
    """Crear visualizaciones comparativas SVM"""
    print("\n6. Creando visualizaciones comparativas...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparación SVM: Variables Originales vs PCA', fontsize=16, fontweight='bold')
    
    # 1. Comparación de métricas
    ax1 = axes[0, 0]
    metricas = ['RMSE', 'MAE', 'R²']
    original_vals = [comparacion_metricas['original']['RMSE_mean'], 
                    comparacion_metricas['original']['MAE_mean'],
                    comparacion_metricas['original']['R2_mean']]
    pca_vals = [comparacion_metricas['pca']['RMSE_mean'],
               comparacion_metricas['pca']['MAE_mean'], 
               comparacion_metricas['pca']['R2_mean']]
    
    x = np.arange(len(metricas))
    width = 0.35
    
    ax1.bar(x - width/2, original_vals, width, label='SVM Original', alpha=0.8)
    ax1.bar(x + width/2, pca_vals, width, label='SVM PCA', alpha=0.8)
    ax1.set_xlabel('Métricas')
    ax1.set_ylabel('Valores')
    ax1.set_title('Comparación de Métricas (Validación Cruzada)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metricas)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución FIES_moderado_grave
    ax2 = axes[0, 1]
    if 'FIES_moderado_grave' in pred_original.columns and 'FIES_moderado_grave' in pred_pca.columns:
        ax2.hist(pred_original['FIES_moderado_grave'], bins=20, alpha=0.6, label='SVM Original', density=True)
        ax2.hist(pred_pca['FIES_moderado_grave'], bins=20, alpha=0.6, label='SVM PCA', density=True)
        ax2.set_xlabel('FIES Moderado-Grave (%)')
        ax2.set_ylabel('Densidad')
        ax2.set_title('Distribución Predicciones FIES_moderado_grave')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Distribución FIES_grave
    ax3 = axes[1, 0]
    if 'FIES_grave' in pred_original.columns and 'FIES_grave' in pred_pca.columns:
        ax3.hist(pred_original['FIES_grave'], bins=20, alpha=0.6, label='SVM Original', density=True)
        ax3.hist(pred_pca['FIES_grave'], bins=20, alpha=0.6, label='SVM PCA', density=True)
        ax3.set_xlabel('FIES Grave (%)')
        ax3.set_ylabel('Densidad')
        ax3.set_title('Distribución Predicciones FIES_grave')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Mejoras porcentuales
    ax4 = axes[1, 1]
    mejoras = ['RMSE', 'MAE', 'R²']
    valores_mejora = [comparacion_metricas['mejora_rmse_pct'], 
                     comparacion_metricas['mejora_mae_pct'],
                     comparacion_metricas['mejora_r2_puntos'] * 100]  # R² en puntos porcentuales
    
    colores = ['green' if x > 0 else 'red' for x in valores_mejora]
    bars = ax4.bar(mejoras, valores_mejora, color=colores, alpha=0.7)
    ax4.set_xlabel('Métricas')
    ax4.set_ylabel('Mejora (%)')
    ax4.set_title('Mejoras con PCA vs Original')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, valores_mejora):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{valor:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    
    # Guardar visualización
    plt.savefig('d:/Tesis maestria/Tesis codigo/modelado/resultados/comparacion_svm_original_vs_pca.png', 
                dpi=300, bbox_inches='tight')
    print("   Visualización guardada: comparacion_svm_original_vs_pca.png")
    
    plt.show()

def generar_reporte_svm_final(comparacion_metricas, comparacion_predicciones, metricas_original, metricas_pca):
    """Generar reporte final JSON con conclusiones SVM"""
    print("\n7. Generando reporte final...")
    
    # Determinar el modelo ganador
    mejor_modelo = "PCA" if comparacion_metricas['mejora_r2_puntos'] > 0 else "Original"
    
    reporte = {
        "titulo": "Comparación SVM: Variables Originales vs PCA",
        "fecha_analisis": datetime.now().isoformat(),
        "modelos_comparados": {
            "original": {
                "descripcion": "SVM con variables originales",
                "features": metricas_original['resumen'].get('features_utilizados', 'N/A'),
                "variables_predichas": metricas_original['resumen']['variables_predichas'],
                "hiperparametros": metricas_original.get('hiperparametros_optimizados', {})
            },
            "pca": {
                "descripcion": "SVM con 7 componentes PCA",
                "features": metricas_pca['resumen']['features_utilizados'],
                "componentes_pca": metricas_pca['componentes_pca'],
                "variables_predichas": metricas_pca['resumen']['variables_predichas'],
                "hiperparametros": metricas_pca['hiperparametros_optimizados']
            }
        },
        "metricas_validacion_cruzada": {
            "original": comparacion_metricas['original'],
            "pca": comparacion_metricas['pca']
        },
        "mejoras_con_pca": {
            "rmse_mejora_porcentaje": comparacion_metricas['mejora_rmse_pct'],
            "mae_mejora_porcentaje": comparacion_metricas['mejora_mae_pct'],
            "r2_mejora_puntos": comparacion_metricas['mejora_r2_puntos']
        },
        "comparacion_predicciones": comparacion_predicciones,
        "modelo_recomendado": mejor_modelo,
        "conclusiones": {
            "rendimiento": f"El modelo SVM {'PCA' if mejor_modelo == 'PCA' else 'Original'} muestra mejor rendimiento",
            "rmse": f"RMSE {'mejoró' if comparacion_metricas['mejora_rmse_pct'] > 0 else 'empeoró'} en {abs(comparacion_metricas['mejora_rmse_pct']):.1f}%",
            "r2": f"R² {'mejoró' if comparacion_metricas['mejora_r2_puntos'] > 0 else 'empeoró'} en {abs(comparacion_metricas['mejora_r2_puntos']):.4f} puntos",
            "eficiencia": f"Modelo PCA usa {metricas_pca['resumen']['features_utilizados']} features vs {metricas_original['resumen'].get('features_utilizados', 'N/A')} del original",
            "escalado": "SVM PCA incluye escalado automático de features y targets"
        },
        "recomendacion_cientifica": {
            "modelo_preferido": mejor_modelo,
            "justificacion": f"Basado en métricas de validación cruzada, el modelo {mejor_modelo} ofrece mejor balance entre precisión y generalización",
            "ventajas_pca": [
                "Reducción dimensional significativa",
                "Eliminación de multicolinealidad",
                "Escalado automático incluido",
                "Optimización de hiperparámetros con GridSearch",
                "Mejor convergencia del algoritmo SVM"
            ] if mejor_modelo == "PCA" else [],
            "limitaciones": [
                "Pérdida de interpretabilidad directa de variables originales",
                "Dependencia de la calidad de la transformación PCA",
                "Tiempo adicional de procesamiento para escalado"
            ] if mejor_modelo == "PCA" else []
        }
    }
    
    # Guardar reporte
    with open('d:/Tesis maestria/Tesis codigo/modelado/resultados/reporte_comparacion_svm.json', 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print("   Reporte final guardado: reporte_comparacion_svm.json")
    
    # Mostrar resumen
    print(f"\n   RESUMEN EJECUTIVO:")
    print(f"     Modelo recomendado: SVM {mejor_modelo}")
    print(f"     RMSE mejora: {comparacion_metricas['mejora_rmse_pct']:+.1f}%")
    print(f"     R² mejora: {comparacion_metricas['mejora_r2_puntos']:+.4f} puntos")
    print(f"     Features: {metricas_pca['resumen']['features_utilizados']} (PCA) vs {metricas_original['resumen'].get('features_utilizados', 'N/A')} (Original)")
    
    return reporte

def main():
    """Función principal para ejecutar comparación SVM"""
    
    print("COMPARACIÓN SVM: VARIABLES ORIGINALES VS COMPONENTES PCA")
    print("=" * 60)
    
    # 1. Cargar métricas
    metricas_original, metricas_pca = cargar_metricas_svm()
    if metricas_original is None or metricas_pca is None:
        print("No se pudieron cargar las métricas necesarias")
        return None
    
    # 2. Extraer métricas de validación
    original_cv, pca_cv = extraer_metricas_validacion_svm(metricas_original, metricas_pca)
    
    # 3. Comparar métricas
    comparacion_metricas = comparar_metricas_svm(original_cv, pca_cv)
    
    # 4. Cargar y comparar predicciones
    pred_original, pred_pca = cargar_predicciones_svm()
    if pred_original is not None and pred_pca is not None:
        comparacion_predicciones = comparar_predicciones_svm(pred_original, pred_pca)
        
        # 5. Crear visualizaciones
        crear_visualizaciones_svm(comparacion_metricas, comparacion_predicciones, pred_original, pred_pca)
    else:
        comparacion_predicciones = {}
        print("   No se pudieron cargar predicciones para comparación visual")
    
    # 6. Generar reporte final
    reporte = generar_reporte_svm_final(comparacion_metricas, comparacion_predicciones, metricas_original, metricas_pca)
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN SVM COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    
    return reporte

if __name__ == "__main__":
    reporte = main()

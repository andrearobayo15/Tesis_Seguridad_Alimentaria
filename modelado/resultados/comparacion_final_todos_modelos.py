#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN FINAL: TODOS LOS MODELOS PCA vs ORIGINALES
======================================================

Análisis comparativo completo de todos los modelos implementados:
- XGBoost (Original vs PCA)
- Random Forest (Original vs PCA)  
- SVM (Original vs PCA)
- Elastic Net (Original vs PCA)

Incluye:
- Tabla comparativa consolidada
- Ranking de modelos por rendimiento
- Análisis de estabilidad y generalización
- Recomendaciones finales para la tesis
- Visualizaciones comprehensivas

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
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def cargar_todas_metricas():
    """Carga métricas de todos los modelos"""
    
    base_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados/metricas")
    
    modelos = {
        'XGBoost Original': 'xgboost_metricas.json',
        'XGBoost PCA': 'xgboost_pca_metricas.json',
        'Random Forest Original': 'random_forest_metricas.json',
        'Random Forest PCA': 'random_forest_pca_metricas.json',
        'SVM Original': 'svm_metricas.json',
        'SVM PCA': 'svm_pca_metricas.json',
        'Elastic Net Original': 'elastic_net_metricas.json',
        'Elastic Net PCA': 'elastic_net_pca_metricas.json'
    }
    
    metricas_cargadas = {}
    
    for nombre_modelo, archivo in modelos.items():
        archivo_path = base_path / archivo
        if archivo_path.exists():
            try:
                with open(archivo_path, 'r', encoding='utf-8') as f:
                    metricas_cargadas[nombre_modelo] = json.load(f)
                print(f"OK {nombre_modelo}: metricas cargadas")
            except Exception as e:
                print(f"ERROR {nombre_modelo}: {e}")
                metricas_cargadas[nombre_modelo] = None
        else:
            print(f"AVISO {nombre_modelo}: archivo no encontrado")
            metricas_cargadas[nombre_modelo] = None
    
    return metricas_cargadas

def crear_tabla_comparativa_completa(metricas):
    """Crea tabla comparativa de todos los modelos"""
    
    resultados = []
    
    for nombre_modelo, datos in metricas.items():
        if datos is None:
            continue
            
        # Extraer métricas de validación cruzada
        cv = datos.get('validacion_cruzada', {})
        
        # Determinar si tiene validación cruzada real
        rmse_std = cv.get('RMSE_std', cv.get('rmse_std', 0))
        tiene_cv = rmse_std > 0
        
        resultado = {
            'Modelo': nombre_modelo,
            'Tipo': 'PCA' if 'PCA' in nombre_modelo else 'Original',
            'Algoritmo': nombre_modelo.replace(' Original', '').replace(' PCA', ''),
            'RMSE_mean': cv.get('RMSE_mean', cv.get('rmse_mean', 0)),
            'RMSE_std': rmse_std,
            'MAE_mean': cv.get('MAE_mean', cv.get('mae_mean', 0)),
            'MAE_std': cv.get('MAE_std', cv.get('mae_std', 0)),
            'R2_mean': cv.get('R2_mean', cv.get('r2_mean', 0)),
            'R2_std': cv.get('R2_std', cv.get('r2_std', 0)),
            'Tiene_CV': tiene_cv
        }
        
        resultados.append(resultado)
    
    df_resultados = pd.DataFrame(resultados)
    
    # Crear tabla formateada
    if not df_resultados.empty:
        print("\nTABLA COMPARATIVA COMPLETA - TODOS LOS MODELOS")
        print("=" * 120)
        
        tabla_formato = df_resultados.copy()
        tabla_formato['RMSE'] = tabla_formato.apply(
            lambda x: f"{x['RMSE_mean']:.4f} ± {x['RMSE_std']:.4f}" if x['Tiene_CV'] else f"{x['RMSE_mean']:.4f} (sin CV)", 
            axis=1
        )
        tabla_formato['MAE'] = tabla_formato.apply(
            lambda x: f"{x['MAE_mean']:.4f} ± {x['MAE_std']:.4f}" if x['Tiene_CV'] else f"{x['MAE_mean']:.4f} (sin CV)", 
            axis=1
        )
        tabla_formato['R²'] = tabla_formato.apply(
            lambda x: f"{x['R2_mean']:.4f} ± {x['R2_std']:.4f}" if x['Tiene_CV'] else f"{x['R2_mean']:.4f} (sin CV)", 
            axis=1
        )
        
        tabla_display = tabla_formato[['Modelo', 'RMSE', 'MAE', 'R²']].copy()
        print(tabla_display.to_string(index=False))
    
    return df_resultados

def calcular_ranking_modelos(df_resultados):
    """Calcula ranking de modelos basado en múltiples criterios"""
    
    if df_resultados.empty:
        return None
    
    # Filtrar solo modelos con validación cruzada real
    df_cv = df_resultados[df_resultados['Tiene_CV'] == True].copy()
    
    if df_cv.empty:
        print("AVISO: No hay modelos con validación cruzada temporal")
        return None
    
    # Normalizar métricas (0-1) para ranking
    df_ranking = df_cv.copy()
    
    # RMSE y MAE: menor es mejor (invertir)
    df_ranking['RMSE_norm'] = 1 - (df_cv['RMSE_mean'] - df_cv['RMSE_mean'].min()) / (df_cv['RMSE_mean'].max() - df_cv['RMSE_mean'].min())
    df_ranking['MAE_norm'] = 1 - (df_cv['MAE_mean'] - df_cv['MAE_mean'].min()) / (df_cv['MAE_mean'].max() - df_cv['MAE_mean'].min())
    
    # R²: mayor es mejor
    df_ranking['R2_norm'] = (df_cv['R2_mean'] - df_cv['R2_mean'].min()) / (df_cv['R2_mean'].max() - df_cv['R2_mean'].min())
    
    # Estabilidad: menor desviación estándar es mejor
    df_ranking['Estabilidad_RMSE'] = 1 - (df_cv['RMSE_std'] - df_cv['RMSE_std'].min()) / (df_cv['RMSE_std'].max() - df_cv['RMSE_std'].min())
    df_ranking['Estabilidad_R2'] = 1 - (df_cv['R2_std'] - df_cv['R2_std'].min()) / (df_cv['R2_std'].max() - df_cv['R2_std'].min())
    
    # Score compuesto (ponderado)
    df_ranking['Score_Rendimiento'] = (
        df_ranking['RMSE_norm'] * 0.3 + 
        df_ranking['MAE_norm'] * 0.2 + 
        df_ranking['R2_norm'] * 0.3 +
        df_ranking['Estabilidad_RMSE'] * 0.1 +
        df_ranking['Estabilidad_R2'] * 0.1
    )
    
    # Ordenar por score
    df_ranking = df_ranking.sort_values('Score_Rendimiento', ascending=False)
    
    print("\nRANKING DE MODELOS (Solo con Validación Cruzada Temporal)")
    print("=" * 80)
    
    for i, (idx, row) in enumerate(df_ranking.iterrows(), 1):
        print(f"{i}. {row['Modelo']}")
        print(f"   Score: {row['Score_Rendimiento']:.4f}")
        print(f"   RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}")
        print(f"   R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")
        print()
    
    return df_ranking

def analizar_pca_vs_original(df_resultados):
    """Analiza rendimiento PCA vs Original por algoritmo"""
    
    if df_resultados.empty:
        return
    
    print("\nANÁLISIS PCA vs ORIGINAL POR ALGORITMO")
    print("=" * 60)
    
    algoritmos = df_resultados['Algoritmo'].unique()
    
    comparaciones = []
    
    for algoritmo in algoritmos:
        datos_alg = df_resultados[df_resultados['Algoritmo'] == algoritmo]
        
        original = datos_alg[datos_alg['Tipo'] == 'Original']
        pca = datos_alg[datos_alg['Tipo'] == 'PCA']
        
        if len(original) > 0 and len(pca) > 0:
            orig_row = original.iloc[0]
            pca_row = pca.iloc[0]
            
            print(f"\n{algoritmo}:")
            print(f"  Original: RMSE={orig_row['RMSE_mean']:.4f}, R²={orig_row['R2_mean']:.4f}, CV={orig_row['Tiene_CV']}")
            print(f"  PCA:      RMSE={pca_row['RMSE_mean']:.4f}, R²={pca_row['R2_mean']:.4f}, CV={pca_row['Tiene_CV']}")
            
            # Determinar ganador (solo si ambos tienen CV)
            if orig_row['Tiene_CV'] and pca_row['Tiene_CV']:
                if pca_row['RMSE_mean'] < orig_row['RMSE_mean'] and pca_row['R2_mean'] > orig_row['R2_mean']:
                    ganador = "PCA"
                elif orig_row['RMSE_mean'] < pca_row['RMSE_mean'] and orig_row['R2_mean'] > pca_row['R2_mean']:
                    ganador = "Original"
                else:
                    ganador = "Mixto"
                
                print(f"  Ganador: {ganador}")
                
                comparaciones.append({
                    'Algoritmo': algoritmo,
                    'Ganador': ganador,
                    'RMSE_mejora': ((orig_row['RMSE_mean'] - pca_row['RMSE_mean']) / orig_row['RMSE_mean']) * 100,
                    'R2_mejora': ((pca_row['R2_mean'] - orig_row['R2_mean']) / orig_row['R2_mean']) * 100
                })
            else:
                print(f"  Ganador: No comparable (falta CV)")
    
    return comparaciones

def visualizar_comparacion_completa(df_resultados):
    """Crea visualizaciones comparativas completas"""
    
    if df_resultados.empty:
        return
    
    # Filtrar modelos con CV para visualización principal
    df_cv = df_resultados[df_resultados['Tiene_CV'] == True].copy()
    
    if df_cv.empty:
        print("AVISO: No hay suficientes datos para visualización")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('COMPARACIÓN COMPLETA: TODOS LOS MODELOS ML PARA PREDICCIÓN FIES', 
                 fontsize=16, fontweight='bold')
    
    # 1. RMSE por modelo
    axes[0,0].bar(range(len(df_cv)), df_cv['RMSE_mean'], 
                  yerr=df_cv['RMSE_std'], capsize=5, alpha=0.7)
    axes[0,0].set_title('RMSE por Modelo (menor es mejor)')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_xticks(range(len(df_cv)))
    axes[0,0].set_xticklabels(df_cv['Modelo'], rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. R² por modelo
    axes[0,1].bar(range(len(df_cv)), df_cv['R2_mean'], 
                  yerr=df_cv['R2_std'], capsize=5, alpha=0.7, color='green')
    axes[0,1].set_title('R² por Modelo (mayor es mejor)')
    axes[0,1].set_ylabel('R²')
    axes[0,1].set_xticks(range(len(df_cv)))
    axes[0,1].set_xticklabels(df_cv['Modelo'], rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. MAE por modelo
    axes[0,2].bar(range(len(df_cv)), df_cv['MAE_mean'], 
                  yerr=df_cv['MAE_std'], capsize=5, alpha=0.7, color='orange')
    axes[0,2].set_title('MAE por Modelo (menor es mejor)')
    axes[0,2].set_ylabel('MAE')
    axes[0,2].set_xticks(range(len(df_cv)))
    axes[0,2].set_xticklabels(df_cv['Modelo'], rotation=45, ha='right')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Comparación PCA vs Original
    pca_models = df_cv[df_cv['Tipo'] == 'PCA']
    orig_models = df_cv[df_cv['Tipo'] == 'Original']
    
    if len(pca_models) > 0 and len(orig_models) > 0:
        tipos = ['Original', 'PCA']
        rmse_por_tipo = [orig_models['RMSE_mean'].mean(), pca_models['RMSE_mean'].mean()]
        r2_por_tipo = [orig_models['R2_mean'].mean(), pca_models['R2_mean'].mean()]
        
        axes[1,0].bar(tipos, rmse_por_tipo, alpha=0.7, color=['lightcoral', 'lightblue'])
        axes[1,0].set_title('RMSE Promedio: Original vs PCA')
        axes[1,0].set_ylabel('RMSE Promedio')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].bar(tipos, r2_por_tipo, alpha=0.7, color=['lightcoral', 'lightblue'])
        axes[1,1].set_title('R² Promedio: Original vs PCA')
        axes[1,1].set_ylabel('R² Promedio')
        axes[1,1].grid(True, alpha=0.3)
    
    # 5. Estabilidad (desviación estándar)
    estabilidad = df_cv['RMSE_std'] + df_cv['R2_std']
    axes[1,2].bar(range(len(df_cv)), estabilidad, alpha=0.7, color='purple')
    axes[1,2].set_title('Estabilidad Total (menor es mejor)')
    axes[1,2].set_ylabel('Suma Desviaciones Estándar')
    axes[1,2].set_xticks(range(len(df_cv)))
    axes[1,2].set_xticklabels(df_cv['Modelo'], rotation=45, ha='right')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    plt.savefig(output_path / "comparacion_final_todos_modelos.png", dpi=300, bbox_inches='tight')
    print("OK Visualizacion completa guardada")
    
    plt.show()

def generar_reporte_final(df_resultados, ranking, comparaciones):
    """Genera reporte final completo"""
    
    reporte = {
        "titulo": "Comparación Final: Todos los Modelos ML para Predicción FIES",
        "fecha_analisis": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resumen_ejecutivo": {
            "modelos_evaluados": len(df_resultados) if not df_resultados.empty else 0,
            "modelos_con_cv": len(df_resultados[df_resultados['Tiene_CV'] == True]) if not df_resultados.empty else 0,
            "algoritmos": list(df_resultados['Algoritmo'].unique()) if not df_resultados.empty else [],
            "enfoque_pca": "7 componentes principales + variables categóricas"
        },
        "metodologia": {
            "validacion": "Validación Cruzada Temporal (TimeSeriesSplit, 5 folds)",
            "metricas": ["RMSE", "MAE", "R²"],
            "variables_objetivo": ["FIES_moderado_grave", "FIES_grave"],
            "periodo_entrenamiento": "2022-2024",
            "periodo_prediccion": "2025",
            "criterios_ranking": {
                "rmse": "30%",
                "mae": "20%", 
                "r2": "30%",
                "estabilidad_rmse": "10%",
                "estabilidad_r2": "10%"
            }
        }
    }
    
    # Agregar resultados si están disponibles
    if not df_resultados.empty:
        # Mejores modelos
        if ranking is not None and len(ranking) > 0:
            reporte["ranking_modelos"] = {}
            for i, (idx, row) in enumerate(ranking.head(3).iterrows(), 1):
                reporte["ranking_modelos"][f"puesto_{i}"] = {
                    "modelo": row['Modelo'],
                    "score": float(row['Score_Rendimiento']),
                    "rmse": f"{row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}",
                    "r2": f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}",
                    "mae": f"{row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}"
                }
        
        # Análisis PCA vs Original
        if comparaciones:
            reporte["analisis_pca_vs_original"] = {}
            for comp in comparaciones:
                reporte["analisis_pca_vs_original"][comp['Algoritmo']] = {
                    "ganador": comp['Ganador'],
                    "mejora_rmse_pct": f"{comp['RMSE_mejora']:.2f}%",
                    "mejora_r2_pct": f"{comp['R2_mejora']:.2f}%"
                }
        
        # Estadísticas generales
        df_cv = df_resultados[df_resultados['Tiene_CV'] == True]
        if not df_cv.empty:
            pca_models = df_cv[df_cv['Tipo'] == 'PCA']
            orig_models = df_cv[df_cv['Tipo'] == 'Original']
            
            reporte["estadisticas_generales"] = {
                "pca_vs_original": {
                    "pca_rmse_promedio": float(pca_models['RMSE_mean'].mean()) if len(pca_models) > 0 else None,
                    "original_rmse_promedio": float(orig_models['RMSE_mean'].mean()) if len(orig_models) > 0 else None,
                    "pca_r2_promedio": float(pca_models['R2_mean'].mean()) if len(pca_models) > 0 else None,
                    "original_r2_promedio": float(orig_models['R2_mean'].mean()) if len(orig_models) > 0 else None
                }
            }
    
    # Conclusiones y recomendaciones
    reporte["conclusiones"] = {
        "mejor_algoritmo": "Basado en ranking con validación cruzada temporal",
        "efectividad_pca": "Análisis de reducción dimensional vs variables originales",
        "recomendaciones_tesis": [
            "Usar modelos con validación cruzada temporal para resultados confiables",
            "Considerar PCA para reducir multicolinealidad en datos correlacionados",
            "Evaluar trade-off entre interpretabilidad y rendimiento predictivo",
            "Documentar limitaciones de cada enfoque metodológico"
        ],
        "limitaciones": [
            "Modelos sin validación cruzada pueden mostrar sobreajuste",
            "PCA reduce interpretabilidad directa de variables originales",
            "Resultados específicos para dataset y período temporal analizado"
        ]
    }
    
    # Guardar reporte
    output_path = Path("d:/Tesis maestria/Tesis codigo/modelado/resultados")
    with open(output_path / "reporte_final_todos_modelos.json", 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print("OK Reporte final completo guardado")
    return reporte

def main():
    """Función principal de comparación final"""
    
    print("COMPARACIÓN FINAL: TODOS LOS MODELOS ML PARA PREDICCIÓN FIES")
    print("=" * 80)
    
    # 1. Cargar todas las métricas
    print("\n1. Cargando métricas de todos los modelos...")
    metricas = cargar_todas_metricas()
    
    # 2. Crear tabla comparativa
    print("\n2. Creando tabla comparativa completa...")
    df_resultados = crear_tabla_comparativa_completa(metricas)
    
    # 3. Calcular ranking
    print("\n3. Calculando ranking de modelos...")
    ranking = calcular_ranking_modelos(df_resultados)
    
    # 4. Análisis PCA vs Original
    print("\n4. Analizando PCA vs Original...")
    comparaciones = analizar_pca_vs_original(df_resultados)
    
    # 5. Visualizaciones
    print("\n5. Generando visualizaciones completas...")
    visualizar_comparacion_completa(df_resultados)
    
    # 6. Reporte final
    print("\n6. Generando reporte final...")
    reporte = generar_reporte_final(df_resultados, ranking, comparaciones)
    
    print("\n" + "=" * 80)
    print("COMPARACIÓN FINAL COMPLETADA")
    print("=" * 80)
    print("Archivos generados:")
    print("- comparacion_final_todos_modelos.png")
    print("- reporte_final_todos_modelos.json")
    
    return df_resultados, ranking, reporte

if __name__ == "__main__":
    df_resultados, ranking, reporte = main()

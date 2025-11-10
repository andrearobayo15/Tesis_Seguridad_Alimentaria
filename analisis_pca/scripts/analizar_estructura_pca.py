"""
Análisis de Estructura de Datos PCA para Modelado
Definir variables para entrenamiento de modelos ML
"""

import pandas as pd
import numpy as np

def analizar_estructura_pca():
    print("=" * 60)
    print("ANÁLISIS ESTRUCTURA DATOS PCA PARA MODELADO")
    print("=" * 60)
    
    # Cargar datos PCA
    archivo_pca = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/datos_pca_transformados.csv"
    df_pca = pd.read_csv(archivo_pca)
    
    print(f"Datos PCA cargados: {df_pca.shape[0]:,} registros x {df_pca.shape[1]} variables")
    
    # Mostrar estructura
    print(f"\nESTRUCTURA DE LA BASE DE DATOS PCA:")
    print(f"Columnas: {list(df_pca.columns)}")
    
    # Variables identificadoras
    vars_id = ['departamento', 'año', 'mes', 'fecha', 'clave']
    vars_pca = [col for col in df_pca.columns if col.startswith('PC')]
    
    print(f"\nVARIABLES IDENTIFICADORAS ({len(vars_id)}):")
    for var in vars_id:
        print(f"  - {var}")
    
    print(f"\nCOMPONENTES PRINCIPALES ({len(vars_pca)}):")
    for var in vars_pca:
        print(f"  - {var}")
    
    # Análisis por año
    print(f"\nDISTRIBUCIÓN POR AÑO:")
    for año in sorted(df_pca['año'].unique()):
        count = len(df_pca[df_pca['año'] == año])
        print(f"  - {año}: {count:,} registros")
    
    # Verificar datos completos
    print(f"\nCOMPLETITUD DE DATOS:")
    for col in vars_pca:
        missing = df_pca[col].isnull().sum()
        print(f"  - {col}: {missing} valores faltantes ({(missing/len(df_pca)*100):.1f}%)")
    
    return df_pca, vars_id, vars_pca

def definir_variables_modelado(df_pca, vars_pca):
    print(f"\n" + "=" * 40)
    print("DEFINICIÓN DE VARIABLES PARA MODELADO")
    print("=" * 40)
    
    # Basándome en las memorias: objetivo es predecir variables FIES para 2022
    # usando datos de entrenamiento 2023-2024
    
    print("CONFIGURACIÓN BASADA EN OBJETIVO DE TESIS:")
    print("- OBJETIVO: Predecir condiciones socioeconómicas 2025")
    print("- DATOS ENTRENAMIENTO: 2022-2024 (datos completos)")
    print("- DATOS PREDICCIÓN: 2025 (estructura vacía)")
    
    # Separar por años
    datos_entrenamiento = df_pca[df_pca['año'].isin([2022, 2023, 2024])].copy()
    datos_prediccion = df_pca[df_pca['año'] == 2025].copy()
    
    print(f"\nDATOS PARA ENTRENAMIENTO:")
    print(f"  - Período: 2022-2024")
    print(f"  - Registros: {len(datos_entrenamiento):,}")
    print(f"  - Departamentos: {datos_entrenamiento['departamento'].nunique()}")
    
    print(f"\nDATOS PARA PREDICCIÓN:")
    print(f"  - Período: 2025")
    print(f"  - Registros: {len(datos_prediccion):,}")
    print(f"  - Departamentos: {datos_prediccion['departamento'].nunique()}")
    
    # Variables features (X) y target (y)
    print(f"\nVARIABLES PARA MODELADO:")
    print(f"  - FEATURES (X): {vars_pca} ({len(vars_pca)} componentes)")
    print(f"  - TARGET (y): Depende del objetivo específico")
    print(f"  - IDENTIFICADORES: departamento, año, mes")
    
    # Verificar completitud en entrenamiento
    print(f"\nCOMPLETITUD EN DATOS ENTRENAMIENTO:")
    for col in vars_pca:
        missing = datos_entrenamiento[col].isnull().sum()
        total = len(datos_entrenamiento)
        print(f"  - {col}: {total-missing}/{total} completos ({((total-missing)/total*100):.1f}%)")
    
    return datos_entrenamiento, datos_prediccion, vars_pca

def analizar_variabilidad_componentes(datos_entrenamiento, vars_pca):
    print(f"\n" + "=" * 40)
    print("ANÁLISIS DE VARIABILIDAD DE COMPONENTES")
    print("=" * 40)
    
    print("ESTADÍSTICAS DESCRIPTIVAS DE COMPONENTES PRINCIPALES:")
    stats = datos_entrenamiento[vars_pca].describe()
    print(stats.round(3))
    
    print(f"\nRANGOS DE VARIACIÓN:")
    for col in vars_pca:
        min_val = datos_entrenamiento[col].min()
        max_val = datos_entrenamiento[col].max()
        rango = max_val - min_val
        print(f"  - {col}: [{min_val:.2f}, {max_val:.2f}] (rango: {rango:.2f})")
    
    # Verificar si hay variabilidad suficiente
    print(f"\nVERIFICACIÓN DE VARIABILIDAD:")
    for col in vars_pca:
        std = datos_entrenamiento[col].std()
        if std > 0.1:
            status = "Suficiente"
        else:
            status = "Baja"
        print(f"  - {col}: std = {std:.3f} ({status})")

def generar_resumen_modelado():
    print(f"\n" + "=" * 60)
    print("RESUMEN PARA MODELADO CON PCA")
    print("=" * 60)
    
    resumen = """
CONFIGURACIÓN RECOMENDADA PARA MODELOS ML:

1. DATOS DE ENTRENAMIENTO:
   - Período: 2022-2024 (1,152 registros)
   - Features: PC1, PC2, PC3, PC4, PC5, PC6, PC7 (7 componentes)
   - Reducción dimensional: 50 → 7 variables (86% reducción)

2. DATOS DE PREDICCIÓN:
   - Período: 2025 (384 registros)
   - Estructura: Mismos componentes principales

3. VENTAJAS DEL ENFOQUE PCA:
   - Eliminación de multicolinealidad (64 correlaciones altas resueltas)
   - Reducción significativa de parámetros (menos sobreajuste)
   - Componentes ortogonales (independientes entre sí)
   - 81% de varianza original preservada

4. INTERPRETACIÓN DE COMPONENTES:
   - PC1 (41.9%): Pobreza y condiciones habitacionales
   - PC2 (13.1%): Seguridad alimentaria (FIES)
   - PC3 (8.6%): Calidad de vida y bienestar
   - PC4 (6.5%): Acceso a salud y seguridad social
   - PC5-PC7: Aspectos específicos complementarios

5. PRÓXIMOS PASOS:
   - Definir variable objetivo específica para predicción
   - Entrenar XGBoost con 7 componentes principales
   - Comparar performance vs modelo original (50 variables)
   - Evaluar reducción de sobreajuste
"""
    
    print(resumen)
    
    # Guardar resumen
    with open("d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/resumen_modelado_pca.md", 'w', encoding='utf-8') as f:
        f.write("# RESUMEN PARA MODELADO CON PCA\n\n" + resumen)
    
    print("Resumen guardado: resumen_modelado_pca.md")

def main():
    # 1. Analizar estructura
    df_pca, vars_id, vars_pca = analizar_estructura_pca()
    
    # 2. Definir variables para modelado
    datos_entrenamiento, datos_prediccion, vars_pca = definir_variables_modelado(df_pca, vars_pca)
    
    # 3. Analizar variabilidad
    analizar_variabilidad_componentes(datos_entrenamiento, vars_pca)
    
    # 4. Generar resumen
    generar_resumen_modelado()
    
    return df_pca, datos_entrenamiento, datos_prediccion, vars_pca

if __name__ == "__main__":
    df_pca, datos_entrenamiento, datos_prediccion, vars_pca = main()

"""
Investigación de Variables Climáticas en Componentes Principales
Análisis detallado de por qué solo 2 de 3 variables climáticas aparecen
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def investigar_variables_climaticas():
    print("=" * 60)
    print("INVESTIGACIÓN VARIABLES CLIMÁTICAS EN PCA")
    print("=" * 60)
    
    # Cargar datos originales
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    
    # Identificar variables climáticas
    variables_climaticas = ['precipitacion_promedio', 'temperatura_promedio', 'ndvi_promedio']
    
    print(f"Variables climáticas esperadas: {len(variables_climaticas)}")
    for var in variables_climaticas:
        if var in df.columns:
            print(f"  ✓ {var}: Encontrada")
        else:
            print(f"  ✗ {var}: NO encontrada")
    
    # Verificar datos climáticos
    print(f"\nANÁLISIS DE DATOS CLIMÁTICOS:")
    for var in variables_climaticas:
        if var in df.columns:
            total = len(df)
            missing = df[var].isnull().sum()
            complete = total - missing
            pct_complete = (complete / total) * 100
            
            print(f"\n{var}:")
            print(f"  - Total registros: {total:,}")
            print(f"  - Datos completos: {complete:,} ({pct_complete:.1f}%)")
            print(f"  - Datos faltantes: {missing:,} ({(missing/total*100):.1f}%)")
            
            if complete > 0:
                stats = df[var].describe()
                print(f"  - Min: {stats['min']:.3f}")
                print(f"  - Max: {stats['max']:.3f}")
                print(f"  - Mean: {stats['mean']:.3f}")
                print(f"  - Std: {stats['std']:.3f}")
    
    return df, variables_climaticas

def analizar_loadings_climaticos():
    print(f"\n" + "=" * 50)
    print("ANÁLISIS DE LOADINGS VARIABLES CLIMÁTICAS")
    print("=" * 50)
    
    # Cargar datos y ejecutar PCA
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    
    # Preparar datos
    variables_excluir = ['departamento', 'año', 'mes', 'fecha', 'clave']
    variables_numericas = [col for col in df.columns if col not in variables_excluir]
    X = df[variables_numericas].copy()
    
    # Imputar y estandarizar
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Obtener loadings para primeros 7 componentes
    loadings = pca.components_[:7]
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i+1}' for i in range(7)],
        index=variables_numericas
    )
    
    # Filtrar solo variables climáticas
    variables_climaticas = ['precipitacion_promedio', 'temperatura_promedio', 'ndvi_promedio']
    loadings_climaticos = loadings_df.loc[variables_climaticas]
    
    print("LOADINGS DE VARIABLES CLIMÁTICAS EN COMPONENTES PRINCIPALES:")
    print(loadings_climaticos.round(4))
    
    # Identificar en qué componentes aparecen más fuerte
    print(f"\nCONTRIBUCIÓN POR COMPONENTE (|loading| > 0.1):")
    for var in variables_climaticas:
        print(f"\n{var}:")
        for pc in loadings_climaticos.columns:
            loading_val = loadings_climaticos.loc[var, pc]
            if abs(loading_val) > 0.1:
                print(f"  - {pc}: {loading_val:.4f}")
    
    # Análisis de correlaciones entre variables climáticas
    print(f"\n" + "=" * 40)
    print("CORRELACIONES ENTRE VARIABLES CLIMÁTICAS")
    print("=" * 40)
    
    corr_climaticas = X_imputed[variables_climaticas].corr()
    print(corr_climaticas.round(4))
    
    return loadings_climaticos, corr_climaticas

def revisar_interpretacion_componentes():
    print(f"\n" + "=" * 50)
    print("REVISIÓN DE INTERPRETACIÓN DE COMPONENTES")
    print("=" * 50)
    
    # Del análisis anterior, sabemos que:
    interpretaciones_anteriores = {
        'PC7': ['Desempleo_larga_duracion', 'ndvi_promedio', 'IPC_Total', 'temperatura_promedio']
    }
    
    print("COMPONENTES DONDE APARECEN VARIABLES CLIMÁTICAS:")
    print("PC7 - Variables más importantes:")
    print("  - Desempleo_larga_duracion: 0.591")
    print("  - ndvi_promedio: 0.360")
    print("  - IPC_Total: 0.262")
    print("  - temperatura_promedio: -0.258")
    
    print(f"\nOBSERVACIÓN:")
    print(f"- Solo 2 de 3 variables climáticas aparecen en PC7")
    print(f"- precipitacion_promedio NO aparece en los top loadings")
    print(f"- Esto puede deberse a:")
    print(f"  1. Baja correlación con otras variables")
    print(f"  2. Distribución en múltiples componentes")
    print(f"  3. Menor variabilidad relativa")

def crear_base_datos_pca_con_objetivos():
    print(f"\n" + "=" * 60)
    print("CREANDO BASE DE DATOS PCA CON VARIABLES OBJETIVO")
    print("=" * 60)
    
    # Cargar datos PCA transformados
    archivo_pca = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/datos_pca_transformados.csv"
    df_pca = pd.read_csv(archivo_pca)
    
    # Cargar datos originales para obtener variables objetivo
    archivo_original = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df_original = pd.read_csv(archivo_original)
    
    print(f"Datos PCA: {df_pca.shape}")
    print(f"Datos originales: {df_original.shape}")
    
    # Variables objetivo
    variables_objetivo = ['FIES_moderado_grave', 'FIES_grave']
    
    # Verificar que las variables objetivo existen
    for var in variables_objetivo:
        if var in df_original.columns:
            print(f"✓ {var}: Disponible")
        else:
            print(f"✗ {var}: NO disponible")
    
    # Agregar variables objetivo a datos PCA
    df_pca_completo = df_pca.copy()
    
    for var in variables_objetivo:
        if var in df_original.columns:
            df_pca_completo[var] = df_original[var]
    
    print(f"\nBase de datos PCA completa: {df_pca_completo.shape}")
    print(f"Columnas: {list(df_pca_completo.columns)}")
    
    # Verificar completitud de variables objetivo
    print(f"\nCOMPLETITUD VARIABLES OBJETIVO:")
    for var in variables_objetivo:
        if var in df_pca_completo.columns:
            missing = df_pca_completo[var].isnull().sum()
            total = len(df_pca_completo)
            pct_complete = ((total - missing) / total) * 100
            print(f"  - {var}: {total - missing}/{total} ({pct_complete:.1f}%)")
    
    # Guardar base de datos completa
    archivo_salida = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/base_pca_con_objetivos.csv"
    df_pca_completo.to_csv(archivo_salida, index=False)
    
    print(f"\nArchivo guardado: base_pca_con_objetivos.csv")
    
    return df_pca_completo

def analizar_estructura_final(df_pca_completo):
    print(f"\n" + "=" * 50)
    print("ESTRUCTURA FINAL PARA MODELADO")
    print("=" * 50)
    
    # Separar por tipo de variable
    vars_id = ['departamento', 'año', 'mes', 'fecha', 'clave']
    vars_pca = [col for col in df_pca_completo.columns if col.startswith('PC')]
    vars_objetivo = ['FIES_moderado_grave', 'FIES_grave']
    
    print(f"VARIABLES IDENTIFICADORAS ({len(vars_id)}):")
    for var in vars_id:
        print(f"  - {var}")
    
    print(f"\nCOMPONENTES PRINCIPALES ({len(vars_pca)}):")
    for var in vars_pca:
        print(f"  - {var}")
    
    print(f"\nVARIABLES OBJETIVO ({len(vars_objetivo)}):")
    for var in vars_objetivo:
        if var in df_pca_completo.columns:
            print(f"  - {var} ✓")
        else:
            print(f"  - {var} ✗")
    
    # Análisis por período
    print(f"\nDISTRIBUCIÓN TEMPORAL:")
    for año in sorted(df_pca_completo['año'].unique()):
        count = len(df_pca_completo[df_pca_completo['año'] == año])
        print(f"  - {año}: {count:,} registros")
    
    # Configuración para modelado
    print(f"\nCONFIGURACIÓN PARA MODELADO:")
    print(f"  - Features (X): {vars_pca} ({len(vars_pca)} componentes)")
    print(f"  - Targets (y): {vars_objetivo}")
    print(f"  - Entrenamiento: 2022-2024 ({len(df_pca_completo[df_pca_completo['año'].isin([2022, 2023, 2024])]):,} registros)")
    print(f"  - Predicción: 2025 ({len(df_pca_completo[df_pca_completo['año'] == 2025]):,} registros)")

def main():
    # 1. Investigar variables climáticas
    df, variables_climaticas = investigar_variables_climaticas()
    
    # 2. Analizar loadings climáticos
    loadings_climaticos, corr_climaticas = analizar_loadings_climaticos()
    
    # 3. Revisar interpretación
    revisar_interpretacion_componentes()
    
    # 4. Crear base de datos PCA con objetivos
    df_pca_completo = crear_base_datos_pca_con_objetivos()
    
    # 5. Analizar estructura final
    analizar_estructura_final(df_pca_completo)
    
    return df_pca_completo, loadings_climaticos

if __name__ == "__main__":
    df_pca_completo, loadings_climaticos = main()

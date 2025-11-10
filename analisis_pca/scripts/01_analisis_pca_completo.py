"""
Análisis de Componentes Principales (PCA) - Completo
Carpeta dedicada para análisis PCA separado de modelos ML

Objetivo: Reducir dimensionalidad y multicolinealidad
Basado en recomendación del profesor para manejar sobreajuste
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Configuración
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def ejecutar_pca_completo():
    print("=" * 60)
    print("ANALISIS DE COMPONENTES PRINCIPALES COMPLETO")
    print("=" * 60)
    
    # Crear directorio resultados
    os.makedirs("d:/Tesis maestria/Tesis codigo/analisis_pca/resultados", exist_ok=True)
    
    # 1. Cargar y preparar datos
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    print(f"Datos cargados: {df.shape[0]:,} registros x {df.shape[1]} variables")
    
    # Variables numéricas
    variables_excluir = ['departamento', 'año', 'mes', 'fecha', 'clave']
    variables_numericas = [col for col in df.columns if col not in variables_excluir]
    X = df[variables_numericas].copy()
    
    print(f"Variables numericas: {len(variables_numericas)}")
    print(f"Valores faltantes: {X.isnull().sum().sum():,}")
    
    # 2. Imputación y estandarización
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print("Datos preparados: imputacion y estandarizacion completadas")
    
    # 3. Ejecutar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    # Criterios de componentes
    idx_80 = np.where(varianza_acumulada >= 0.80)[0][0] + 1
    idx_90 = np.where(varianza_acumulada >= 0.90)[0][0] + 1
    
    print(f"\nRESULTADOS PCA:")
    print(f"- Componentes para 80% varianza: {idx_80}")
    print(f"- Componentes para 90% varianza: {idx_90}")
    print(f"- Reduccion dimensional (80%): {len(variables_numericas)} -> {idx_80} ({((len(variables_numericas)-idx_80)/len(variables_numericas)*100):.1f}% reduccion)")
    
    return pca, X_scaled, scaler, df, variables_numericas, X_imputed, idx_80, idx_90

def analizar_correlaciones(X_imputed, variables_numericas):
    print("\n" + "=" * 40)
    print("ANALISIS DE CORRELACIONES")
    print("=" * 40)
    
    # Matriz de correlación
    corr_matrix = X_imputed.corr()
    
    # Correlaciones altas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'Var1': corr_matrix.columns[i],
                    'Var2': corr_matrix.columns[j],
                    'Correlacion': corr_val
                })
    
    print(f"Correlaciones altas (|r| > 0.8): {len(high_corr_pairs)}")
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlacion', key=abs, ascending=False)
        print("Top 10 correlaciones:")
        for _, row in high_corr_df.head(10).iterrows():
            print(f"  {row['Var1']} <-> {row['Var2']}: r = {row['Correlacion']:.3f}")
    
    return corr_matrix, high_corr_pairs

def interpretar_componentes(pca, variables_numericas, n_comp=7):
    print(f"\n" + "=" * 40)
    print(f"INTERPRETACION DE COMPONENTES (PC1-PC{n_comp})")
    print("=" * 40)
    
    # Loadings
    loadings = pca.components_[:n_comp]
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i+1}' for i in range(n_comp)],
        index=variables_numericas
    )
    
    # Interpretar cada componente
    interpretaciones = {}
    for i in range(n_comp):
        pc_name = f'PC{i+1}'
        varianza = pca.explained_variance_ratio_[i]
        
        print(f"\n{pc_name} - Varianza explicada: {varianza:.3f} ({varianza*100:.1f}%)")
        
        # Variables más importantes
        loadings_abs = loadings_df[pc_name].abs().sort_values(ascending=False)
        top_vars = loadings_abs.head(8)
        
        print("  Variables mas importantes:")
        variables_componente = []
        for var in top_vars.index:
            loading_val = loadings_df.loc[var, pc_name]
            print(f"    {var}: {loading_val:.3f}")
            variables_componente.append((var, loading_val))
        
        interpretaciones[pc_name] = variables_componente
    
    return loadings_df, interpretaciones

def crear_visualizaciones(pca, idx_80, idx_90):
    print(f"\n" + "=" * 40)
    print("CREANDO VISUALIZACIONES")
    print("=" * 40)
    
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    # 1. Scree Plot y Varianza Acumulada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scree plot (primeros 20 componentes)
    n_show = min(20, len(varianza_explicada))
    componentes = range(1, n_show + 1)
    
    ax1.plot(componentes, varianza_explicada[:n_show], 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada')
    ax1.set_title('Scree Plot - Varianza por Componente')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% varianza')
    ax1.legend()
    
    # Varianza acumulada
    ax2.plot(componentes, varianza_acumulada[:n_show], 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80% varianza')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% varianza')
    ax2.axvline(x=idx_80, color='g', linestyle=':', alpha=0.7)
    ax2.axvline(x=idx_90, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Numero de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.set_title('Varianza Explicada Acumulada')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/pca_varianza.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graficos guardados: pca_varianza.png")

def generar_datos_transformados(pca, X_scaled, scaler, df, variables_numericas, n_comp=7):
    print(f"\n" + "=" * 40)
    print(f"GENERANDO DATOS TRANSFORMADOS ({n_comp} componentes)")
    print("=" * 40)
    
    # PCA con componentes óptimos
    pca_optimo = PCA(n_components=n_comp)
    X_pca_optimo = pca_optimo.fit_transform(X_scaled)
    
    # Crear DataFrame
    columnas_pc = [f'PC{i+1}' for i in range(n_comp)]
    df_pca = pd.DataFrame(X_pca_optimo, columns=columnas_pc, index=df.index)
    
    # Agregar identificadores
    variables_id = ['departamento', 'año', 'mes', 'fecha', 'clave']
    for var in variables_id:
        df_pca[var] = df[var]
    
    # Reordenar
    columnas_finales = variables_id + columnas_pc
    df_pca = df_pca[columnas_finales]
    
    # Guardar
    archivo_pca = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/datos_pca_transformados.csv"
    df_pca.to_csv(archivo_pca, index=False)
    
    print(f"Dataset transformado: {df_pca.shape[0]:,} registros x {df_pca.shape[1]} variables")
    print(f"Archivo guardado: datos_pca_transformados.csv")
    
    return df_pca, pca_optimo

def generar_reporte_pca(pca, interpretaciones, high_corr_pairs, idx_80, idx_90, variables_numericas):
    print(f"\n" + "=" * 40)
    print("GENERANDO REPORTE PCA")
    print("=" * 40)
    
    reporte = f"""# REPORTE ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)

## RESUMEN EJECUTIVO
- **Variables originales**: {len(variables_numericas)}
- **Componentes principales (80% varianza)**: {idx_80}
- **Componentes principales (90% varianza)**: {idx_90}
- **Reducción dimensional**: {((len(variables_numericas)-idx_80)/len(variables_numericas)*100):.1f}%
- **Correlaciones altas identificadas**: {len(high_corr_pairs)}

## VARIANZA EXPLICADA POR COMPONENTE
"""
    
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        varianza = pca.explained_variance_ratio_[i]
        varianza_acum = np.cumsum(pca.explained_variance_ratio_)[i]
        reporte += f"- **PC{i+1}**: {varianza:.3f} ({varianza*100:.1f}%) - Acumulada: {varianza_acum:.3f}\n"
    
    reporte += f"\n## INTERPRETACIÓN DE COMPONENTES PRINCIPALES\n"
    
    for pc_name, variables in interpretaciones.items():
        reporte += f"\n### {pc_name}\n"
        reporte += "Variables más importantes:\n"
        for var, loading in variables:
            reporte += f"- {var}: {loading:.3f}\n"
    
    reporte += f"""
## RECOMENDACIONES
1. **Usar {idx_80} componentes principales** para capturar 80% de la varianza
2. **Reducción significativa** de dimensionalidad: {len(variables_numericas)} → {idx_80} variables
3. **Solución al sobreajuste**: Menos parámetros a estimar en modelos ML
4. **Eliminación de multicolinealidad**: {len(high_corr_pairs)} correlaciones altas identificadas

## ARCHIVOS GENERADOS
- `datos_pca_transformados.csv`: Dataset con componentes principales
- `pca_varianza.png`: Visualizaciones de varianza explicada
- `reporte_pca.md`: Este reporte

## PRÓXIMOS PASOS
1. Re-entrenar XGBoost con componentes principales
2. Comparar performance con modelo original
3. Evaluar reducción de sobreajuste
"""
    
    # Guardar reporte
    archivo_reporte = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/reporte_pca.md"
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(f"Reporte guardado: reporte_pca.md")
    return reporte

def main():
    """Función principal del análisis PCA"""
    
    # 1. Ejecutar PCA completo
    pca, X_scaled, scaler, df, variables_numericas, X_imputed, idx_80, idx_90 = ejecutar_pca_completo()
    
    # 2. Analizar correlaciones
    corr_matrix, high_corr_pairs = analizar_correlaciones(X_imputed, variables_numericas)
    
    # 3. Interpretar componentes
    loadings_df, interpretaciones = interpretar_componentes(pca, variables_numericas, n_comp=idx_80)
    
    # 4. Crear visualizaciones
    crear_visualizaciones(pca, idx_80, idx_90)
    
    # 5. Generar datos transformados
    df_pca, pca_optimo = generar_datos_transformados(pca, X_scaled, scaler, df, variables_numericas, n_comp=idx_80)
    
    # 6. Generar reporte
    reporte = generar_reporte_pca(pca, interpretaciones, high_corr_pairs, idx_80, idx_90, variables_numericas)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL DEL ANÁLISIS PCA")
    print("=" * 60)
    print(f"Variables originales: {len(variables_numericas)}")
    print(f"Componentes principales (80% varianza): {idx_80}")
    print(f"Reduccion dimensional: {((len(variables_numericas)-idx_80)/len(variables_numericas)*100):.1f}%")
    print(f"Correlaciones altas identificadas: {len(high_corr_pairs)}")
    print(f"\nArchivos generados en analisis_pca/resultados/:")
    print(f"  - datos_pca_transformados.csv")
    print(f"  - pca_varianza.png")
    print(f"  - reporte_pca.md")
    
    return df_pca, pca_optimo, scaler, loadings_df, interpretaciones

if __name__ == "__main__":
    df_pca, pca_optimo, scaler, loadings_df, interpretaciones = main()

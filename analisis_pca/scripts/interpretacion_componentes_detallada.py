"""
Interpretación Detallada de Componentes Principales
Análisis completo de loadings y significado de cada PC1-PC7
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_y_preparar_datos():
    """Cargar datos y ejecutar PCA para obtener loadings actualizados"""
    print("=" * 60)
    print("CARGANDO Y PREPARANDO DATOS PARA INTERPRETACION PCA")
    print("=" * 60)
    
    # Cargar datos originales
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    
    # Preparar datos para PCA
    variables_excluir = ['departamento', 'año', 'mes', 'fecha', 'clave']
    variables_numericas = [col for col in df.columns if col not in variables_excluir]
    X = df[variables_numericas].copy()
    
    print(f"Variables numericas incluidas: {len(variables_numericas)}")
    print(f"Registros totales: {len(df):,}")
    
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Ejecutar PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    return pca, variables_numericas, X_imputed

def generar_interpretacion_detallada(pca, variables_numericas):
    """Generar interpretación detallada de cada componente"""
    print("\n" + "=" * 60)
    print("INTERPRETACION DETALLADA DE COMPONENTES PRINCIPALES")
    print("=" * 60)
    
    # Crear DataFrame de loadings
    loadings = pca.components_[:7]  # Solo primeros 7 componentes
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i+1}' for i in range(7)],
        index=variables_numericas
    )
    
    # Varianza explicada
    varianza_explicada = pca.explained_variance_ratio_[:7]
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    interpretaciones = {}
    
    for i in range(7):
        pc_name = f'PC{i+1}'
        print(f"\n{'-'*50}")
        print(f"COMPONENTE PRINCIPAL {i+1} (PC{i+1})")
        print(f"{'-'*50}")
        print(f"Varianza explicada: {varianza_explicada[i]:.3f} ({varianza_explicada[i]*100:.1f}%)")
        print(f"Varianza acumulada: {varianza_acumulada[i]:.3f} ({varianza_acumulada[i]*100:.1f}%)")
        
        # Obtener loadings ordenados por valor absoluto
        loadings_pc = loadings_df[pc_name].abs().sort_values(ascending=False)
        
        print(f"\nVARIABLES MAS IMPORTANTES (|loading| > 0.15):")
        variables_importantes = []
        for var in loadings_pc.index:
            loading_val = loadings_df.loc[var, pc_name]
            if abs(loading_val) > 0.15:
                signo = "+" if loading_val > 0 else "-"
                variables_importantes.append({
                    'variable': var,
                    'loading': loading_val,
                    'abs_loading': abs(loading_val)
                })
                print(f"  {signo} {var}: {loading_val:.4f}")
        
        # Categorizar variables por tipo
        categorias = categorizar_variables(variables_importantes)
        
        print(f"\nCATEGORIAS DE VARIABLES:")
        for categoria, vars_cat in categorias.items():
            if vars_cat:
                print(f"  - {categoria}: {len(vars_cat)} variables")
                for var_info in vars_cat[:3]:  # Mostrar top 3
                    print(f"    * {var_info['variable']}: {var_info['loading']:.3f}")
        
        # Interpretación conceptual
        interpretacion = interpretar_componente(pc_name, variables_importantes, categorias)
        interpretaciones[pc_name] = interpretacion
        
        print(f"\nINTERPRETACION CONCEPTUAL:")
        print(f"  {interpretacion}")
    
    return loadings_df, interpretaciones

def categorizar_variables(variables_importantes):
    """Categorizar variables por tipo temático"""
    categorias = {
        'Socioeconómicas IPM': [],
        'Calidad de Vida ECV': [],
        'Seguridad Alimentaria FIES': [],
        'Climáticas': [],
        'Económicas': [],
        'Vivienda y Servicios': [],
        'Otras': []
    }
    
    for var_info in variables_importantes:
        var = var_info['variable']
        
        if any(x in var.lower() for x in ['ipm', 'analfabetismo', 'logro_educativo', 'asistencia_escolar', 'rezago_escolar', 'barreras_servicios_salud', 'barreras_cuidado_primera_infancia', 'trabajo_infantil', 'desempleo_larga_duracion', 'empleo_informal']):
            categorias['Socioeconómicas IPM'].append(var_info)
        elif any(x in var.lower() for x in ['vida_general', 'salud', 'trabajo', 'ingresos', 'vivienda', 'educacion', 'tiempo_libre', 'seguridad']):
            categorias['Calidad de Vida ECV'].append(var_info)
        elif 'fies' in var.lower():
            categorias['Seguridad Alimentaria FIES'].append(var_info)
        elif any(x in var.lower() for x in ['precipitacion', 'temperatura', 'ndvi']):
            categorias['Climáticas'].append(var_info)
        elif any(x in var.lower() for x in ['ipc', 'pobreza_monetaria', 'ingresos']):
            categorias['Económicas'].append(var_info)
        elif any(x in var.lower() for x in ['hacinamiento', 'material_pisos', 'material_paredes', 'fuente_agua', 'eliminacion_excretas', 'energia_cocinar', 'recoleccion_basuras']):
            categorias['Vivienda y Servicios'].append(var_info)
        else:
            categorias['Otras'].append(var_info)
    
    return categorias

def interpretar_componente(pc_name, variables_importantes, categorias):
    """Generar interpretación conceptual del componente"""
    interpretaciones_base = {
        'PC1': "Condiciones socioeconómicas generales y calidad de vida",
        'PC2': "Educación y desarrollo humano", 
        'PC3': "Condiciones climáticas y ambientales",
        'PC4': "Seguridad alimentaria y acceso a alimentos",
        'PC5': "Condiciones de vivienda y servicios básicos",
        'PC6': "Empleo y condiciones laborales",
        'PC7': "Factores económicos y ambientales específicos"
    }
    
    # Interpretación más específica basada en variables dominantes
    if len(variables_importantes) > 0:
        var_principal = variables_importantes[0]['variable']
        
        # Ajustar interpretación según variable principal
        if 'vida_general' in var_principal.lower() or 'salud' in var_principal.lower():
            return "Bienestar general y percepción de calidad de vida"
        elif 'educacion' in var_principal.lower() or 'analfabetismo' in var_principal.lower():
            return "Acceso y calidad educativa"
        elif 'fies' in var_principal.lower():
            return "Inseguridad alimentaria y acceso a alimentos"
        elif any(x in var_principal.lower() for x in ['precipitacion', 'temperatura', 'ndvi']):
            return "Condiciones climáticas y ambientales"
        elif 'vivienda' in var_principal.lower() or 'hacinamiento' in var_principal.lower():
            return "Condiciones habitacionales y servicios"
        elif 'empleo' in var_principal.lower() or 'trabajo' in var_principal.lower():
            return "Condiciones laborales y empleo"
        elif 'pobreza' in var_principal.lower() or 'ipc' in var_principal.lower():
            return "Condiciones económicas y monetarias"
    
    return interpretaciones_base.get(pc_name, "Componente multidimensional")

def crear_matriz_loadings_visual(loadings_df):
    """Crear visualización de matriz de loadings"""
    print(f"\n" + "=" * 50)
    print("CREANDO VISUALIZACION DE LOADINGS")
    print("=" * 50)
    
    # Filtrar solo loadings significativos (>0.15)
    loadings_significativos = loadings_df.copy()
    loadings_significativos[abs(loadings_significativos) < 0.15] = 0
    
    # Crear heatmap
    plt.figure(figsize=(12, 20))
    sns.heatmap(loadings_significativos, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.3f',
                cbar_kws={'label': 'Loading Value'})
    plt.title('Matriz de Loadings Significativos (|loading| > 0.15)', fontsize=14, fontweight='bold')
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')
    plt.tight_layout()
    
    archivo_plot = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/matriz_loadings_detallada.png"
    plt.savefig(archivo_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualización guardada: matriz_loadings_detallada.png")
    
    return loadings_significativos

def generar_reporte_interpretacion(interpretaciones, loadings_df, varianza_explicada):
    """Generar reporte completo de interpretación"""
    print(f"\n" + "=" * 50)
    print("GENERANDO REPORTE DE INTERPRETACION")
    print("=" * 50)
    
    reporte = []
    reporte.append("# INTERPRETACIÓN DETALLADA DE COMPONENTES PRINCIPALES")
    reporte.append("## Análisis PCA - Base Master Tesis")
    reporte.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("")
    
    reporte.append("## RESUMEN GENERAL")
    reporte.append(f"- **Componentes analizados**: 7 (PC1-PC7)")
    reporte.append(f"- **Variables originales**: {len(loadings_df.index)}")
    reporte.append(f"- **Varianza total explicada**: {sum(varianza_explicada):.3f} ({sum(varianza_explicada)*100:.1f}%)")
    reporte.append("")
    
    for i, (pc_name, interpretacion) in enumerate(interpretaciones.items()):
        reporte.append(f"## {pc_name.upper()}")
        reporte.append(f"**Interpretación**: {interpretacion}")
        reporte.append(f"**Varianza explicada**: {varianza_explicada[i]:.3f} ({varianza_explicada[i]*100:.1f}%)")
        
        # Variables más importantes
        loadings_pc = loadings_df[pc_name].abs().sort_values(ascending=False)
        reporte.append("**Variables principales**:")
        
        count = 0
        for var in loadings_pc.index:
            loading_val = loadings_df.loc[var, pc_name]
            if abs(loading_val) > 0.15 and count < 8:
                signo = "+" if loading_val > 0 else "-"
                reporte.append(f"- {signo} **{var}**: {loading_val:.4f}")
                count += 1
        reporte.append("")
    
    # Guardar reporte
    archivo_reporte = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/INTERPRETACION_COMPONENTES_DETALLADA.md"
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte))
    
    print(f"Reporte guardado: INTERPRETACION_COMPONENTES_DETALLADA.md")
    
    return reporte

def exportar_loadings_excel(loadings_df):
    """Exportar matriz de loadings a Excel"""
    print(f"\n" + "=" * 40)
    print("EXPORTANDO LOADINGS A EXCEL")
    print("=" * 40)
    
    # Crear Excel con múltiples hojas
    archivo_excel = "d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/loadings_componentes_detallados.xlsx"
    
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        # Hoja 1: Todos los loadings
        loadings_df.to_excel(writer, sheet_name='Loadings_Completos')
        
        # Hoja 2: Solo loadings significativos
        loadings_significativos = loadings_df.copy()
        loadings_significativos[abs(loadings_significativos) < 0.15] = np.nan
        loadings_significativos.to_excel(writer, sheet_name='Loadings_Significativos')
        
        # Hoja 3: Resumen por componente
        resumen_componentes = []
        for pc in loadings_df.columns:
            loadings_pc = loadings_df[pc].abs().sort_values(ascending=False)
            top_vars = []
            for var in loadings_pc.index[:10]:
                if abs(loadings_df.loc[var, pc]) > 0.1:
                    top_vars.append(f"{var}: {loadings_df.loc[var, pc]:.4f}")
            
            resumen_componentes.append({
                'Componente': pc,
                'Variables_Principales': '; '.join(top_vars[:5])
            })
        
        resumen_df = pd.DataFrame(resumen_componentes)
        resumen_df.to_excel(writer, sheet_name='Resumen_Componentes', index=False)
    
    print(f"Excel exportado: loadings_componentes_detallados.xlsx")

def main():
    """Función principal"""
    print("INICIANDO ANALISIS DETALLADO DE INTERPRETACION PCA")
    
    # 1. Cargar y preparar datos
    pca, variables_numericas, X_imputed = cargar_y_preparar_datos()
    
    # 2. Generar interpretación detallada
    loadings_df, interpretaciones = generar_interpretacion_detallada(pca, variables_numericas)
    
    # 3. Crear visualización
    loadings_significativos = crear_matriz_loadings_visual(loadings_df)
    
    # 4. Generar reporte
    varianza_explicada = pca.explained_variance_ratio_[:7]
    reporte = generar_reporte_interpretacion(interpretaciones, loadings_df, varianza_explicada)
    
    # 5. Exportar a Excel
    exportar_loadings_excel(loadings_df)
    
    print(f"\n" + "=" * 60)
    print("ANALISIS COMPLETADO")
    print("=" * 60)
    print("Archivos generados:")
    print("- INTERPRETACION_COMPONENTES_DETALLADA.md")
    print("- loadings_componentes_detallados.xlsx")
    print("- matriz_loadings_detallada.png")
    
    return loadings_df, interpretaciones

if __name__ == "__main__":
    loadings_df, interpretaciones = main()

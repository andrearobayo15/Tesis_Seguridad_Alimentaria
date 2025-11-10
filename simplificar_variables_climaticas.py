#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para simplificar variables climáticas en la base master
Mantener solo: precipitacion_promedio, temperatura_promedio, ndvi_promedio
"""

import pandas as pd
from pathlib import Path

def simplificar_variables_climaticas():
    """
    Simplifica las variables climáticas manteniendo solo las 3 variables promedio
    """
    
    print("=== SIMPLIFICANDO VARIABLES CLIMÁTICAS ===")
    
    # Cargar base master corregida
    archivo_entrada = Path("data/base de datos central/base_master_consolidada_2022_2025_CORREGIDA.csv")
    
    if not archivo_entrada.exists():
        print(f"ERROR: Archivo no encontrado: {archivo_entrada}")
        return
    
    print(f"Cargando base master corregida: {archivo_entrada}")
    df = pd.read_csv(archivo_entrada)
    
    print(f"Base original: {len(df):,} registros, {len(df.columns)} variables")
    
    # Identificar todas las variables climáticas actuales
    variables_climaticas_actuales = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['precipitacion', 'temperatura', 'ndvi', 'lst']):
            variables_climaticas_actuales.append(col)
    
    print(f"\nVariables climáticas actuales encontradas: {len(variables_climaticas_actuales)}")
    for var in sorted(variables_climaticas_actuales):
        print(f"  {var}")
    
    # Variables climáticas que queremos mantener
    variables_mantener = [
        'precipitacion_promedio',
        'temperatura_promedio', 
        'ndvi_promedio'
    ]
    
    print(f"\nVariables climáticas a mantener: {len(variables_mantener)}")
    for var in variables_mantener:
        print(f"  {var}")
    
    # Verificar que las variables a mantener existen
    variables_mantener_existentes = []
    variables_faltantes = []
    
    for var in variables_mantener:
        if var in df.columns:
            variables_mantener_existentes.append(var)
            # Verificar datos disponibles
            datos_disponibles = df[var].notna().sum()
            total_registros = len(df)
            print(f"  OK {var}: {datos_disponibles}/{total_registros} ({datos_disponibles/total_registros*100:.1f}%)")
        else:
            variables_faltantes.append(var)
            print(f"  ERROR {var}: NO ENCONTRADA")
    
    # Variables climáticas a eliminar
    variables_eliminar = []
    for var in variables_climaticas_actuales:
        if var not in variables_mantener:
            variables_eliminar.append(var)
    
    print(f"\nVariables climáticas a eliminar: {len(variables_eliminar)}")
    for var in sorted(variables_eliminar):
        datos_disponibles = df[var].notna().sum()
        print(f"  ELIMINAR {var}: {datos_disponibles} datos")
    
    # Crear base simplificada
    print(f"\n=== CREANDO BASE SIMPLIFICADA ===")
    
    # Eliminar variables climáticas no deseadas
    df_simplificado = df.drop(columns=variables_eliminar)
    
    print(f"Variables eliminadas: {len(variables_eliminar)}")
    print(f"Base simplificada: {len(df_simplificado):,} registros, {len(df_simplificado.columns)} variables")
    
    # Verificar estructura final
    print(f"\n=== VERIFICACIÓN ESTRUCTURA FINAL ===")
    
    # Categorizar variables restantes
    categorias_finales = {
        'Identificadores': ['departamento', 'año', 'mes', 'fecha', 'mes_nombre'],
        'IPM': [col for col in df_simplificado.columns if any(keyword in col for keyword in ['IPM', 'Analfabetismo', 'Bajo_logro', 'Barreras', 'Desempleo', 'Inasistencia', 'Rezago', 'Sin_aseguramiento', 'Trabajo_informal'])],
        'FIES': [col for col in df_simplificado.columns if any(keyword in col for keyword in ['Comio_menos', 'Preocupacion_alimentos', 'Sin_alimentos', 'Hambre', 'leve_moderado', 'grave', 'moderada', 'inseguridad_alimentaria'])],
        'ECV': [col for col in df_simplificado.columns if any(keyword in col for keyword in ['Vida_general', 'Salud', 'Seguridad', 'Trabajo_actividad', 'Tiempo_libre', 'Ingreso', 'Propia_totalmente', 'En_arriendo', 'Con_permiso', 'Posesion', 'Propiedad_colectiva', 'Deficit', 'gastos', 'Pobreza_monetaria', 'Energia', 'Gas_natural', 'Acueducto', 'Alcantarillado', 'Recoleccion_basura', 'Telefono_fijo', 'Ningun_servicio'])],
        'Climáticas': variables_mantener_existentes
    }
    
    print("\nEstructura final por categorías:")
    total_variables_categorizadas = 0
    
    for categoria, variables in categorias_finales.items():
        variables_existentes = [var for var in variables if var in df_simplificado.columns]
        print(f"  {categoria}: {len(variables_existentes)} variables")
        total_variables_categorizadas += len(variables_existentes)
        
        if categoria == 'Climáticas':
            for var in variables_existentes:
                datos = df_simplificado[var].notna().sum()
                print(f"    {var}: {datos} registros con datos")
    
    print(f"\nTotal variables categorizadas: {total_variables_categorizadas}")
    print(f"Total variables en base: {len(df_simplificado.columns)}")
    
    # Verificar datos climáticos por año
    print(f"\n=== COBERTURA DATOS CLIMÁTICOS POR AÑO ===")
    
    for año in [2022, 2023, 2024, 2025]:
        df_año = df_simplificado[df_simplificado['año'] == año]
        print(f"\nAño {año} ({len(df_año)} registros):")
        
        for var in variables_mantener_existentes:
            datos_año = df_año[var].notna().sum()
            cobertura = datos_año / len(df_año) * 100
            print(f"  {var}: {datos_año}/{len(df_año)} ({cobertura:.1f}%)")
    
    # Ejemplo de datos para verificación
    print(f"\n=== EJEMPLO DATOS CLIMÁTICOS ===")
    
    # Amazonas 2022
    amazonas_2022 = df_simplificado[(df_simplificado['departamento'] == 'Amazonas') & (df_simplificado['año'] == 2022)]
    print(f"\nAmazonas 2022 (primeros 3 meses):")
    
    cols_ejemplo = ['fecha'] + variables_mantener_existentes
    ejemplo_data = amazonas_2022[cols_ejemplo].head(3)
    
    for idx, row in ejemplo_data.iterrows():
        fecha = row['fecha']
        valores = []
        for var in variables_mantener_existentes:
            valor = row[var]
            if pd.notna(valor):
                valores.append(f"{var.split('_')[0]}: {valor:.2f}")
            else:
                valores.append(f"{var.split('_')[0]}: N/A")
        
        print(f"  {fecha}: {', '.join(valores)}")
    
    # Bogotá 2022 para verificar corrección
    bogota_2022 = df_simplificado[(df_simplificado['departamento'] == 'Bogotá D.C.') & (df_simplificado['año'] == 2022)]
    print(f"\nBogotá D.C. 2022 (primeros 3 meses):")
    
    ejemplo_bogota = bogota_2022[cols_ejemplo].head(3)
    
    for idx, row in ejemplo_bogota.iterrows():
        fecha = row['fecha']
        valores = []
        for var in variables_mantener_existentes:
            valor = row[var]
            if pd.notna(valor):
                valores.append(f"{var.split('_')[0]}: {valor:.2f}")
            else:
                valores.append(f"{var.split('_')[0]}: N/A")
        
        print(f"  {fecha}: {', '.join(valores)}")
    
    # Guardar base simplificada
    print(f"\n=== GUARDANDO BASE SIMPLIFICADA ===")
    
    archivo_salida = Path("data/base de datos central/base_master_2022_2025_FINAL_SIMPLIFICADA.csv")
    
    df_simplificado.to_csv(archivo_salida, index=False)
    print(f"Base simplificada guardada: {archivo_salida}")
    
    # También Excel
    archivo_excel = Path("data/base de datos central/base_master_2022_2025_FINAL_SIMPLIFICADA.xlsx")
    
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_simplificado.to_excel(writer, sheet_name='Base_Master_Final', index=False)
        
        # Hoja de resumen de variables
        resumen_variables = []
        for categoria, variables in categorias_finales.items():
            variables_existentes = [var for var in variables if var in df_simplificado.columns]
            for var in variables_existentes:
                datos_totales = df_simplificado[var].notna().sum()
                resumen_variables.append({
                    'categoria': categoria,
                    'variable': var,
                    'datos_disponibles': datos_totales,
                    'total_registros': len(df_simplificado),
                    'cobertura_pct': datos_totales / len(df_simplificado) * 100
                })
        
        df_resumen = pd.DataFrame(resumen_variables)
        df_resumen.to_excel(writer, sheet_name='Resumen_Variables', index=False)
    
    print(f"Versión Excel guardada: {archivo_excel}")
    
    # Estadísticas finales
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"OK Registros: {len(df_simplificado):,}")
    print(f"OK Variables totales: {len(df_simplificado.columns)}")
    print(f"OK Variables climáticas: {len(variables_mantener_existentes)} (solo promedios)")
    print(f"OK Período: {df_simplificado['año'].min()}-{df_simplificado['año'].max()}")
    print(f"OK Orden: Cronológico (Año - Mes - Departamento)")
    
    print(f"\nVARIABLES CLIMÁTICAS FINALES:")
    for var in variables_mantener_existentes:
        datos = df_simplificado[var].notna().sum()
        print(f"   {var}: {datos:,} registros con datos")
    
    print(f"\nBASE MASTER SIMPLIFICADA COMPLETADA!")
    print(f"Archivo principal: {archivo_salida}")
    
    return df_simplificado

if __name__ == "__main__":
    df_resultado = simplificar_variables_climaticas()

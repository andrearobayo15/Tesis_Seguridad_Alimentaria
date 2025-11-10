#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Procesamiento IPC con extrapolación correcta por departamento
"""

import pandas as pd
import numpy as np

def procesar_ipc_extrapolacion_correcta():
    """
    Procesa IPC extrapolando cada ciudad a su departamento
    """
    
    print("=== PROCESAMIENTO IPC CON EXTRAPOLACIÓN CORRECTA ===")
    
    # 1. CARGAR DATOS IPC ORIGINALES
    print(f"\n=== 1. CARGAR DATOS IPC ===")
    df_ipc = pd.read_excel("data/original/IPC todos los departamentos.xls")
    print(f"IPC cargado: {len(df_ipc)} registros")
    
    # Renombrar columnas
    df_ipc = df_ipc.rename(columns={
        'Año': 'año',
        'Mes': 'mes', 
        'Ciudad': 'ciudad',
        'Número índice': 'IPC_Total'
    })
    
    ciudades_ipc = sorted(df_ipc['ciudad'].unique())
    print(f"Ciudades IPC ({len(ciudades_ipc)}): {ciudades_ipc}")
    
    # 2. MAPEAR CIUDADES A DEPARTAMENTOS (32 departamentos)
    print(f"\n=== 2. MAPEAR CIUDADES A DEPARTAMENTOS ===")
    
    mapeo_ciudad_departamento = {
        'ARMENIA': 'Quindío',
        'BARRANQUILLA': 'Atlántico',
        'BOGOTÁ, D.C.': 'Bogotá D.C.',  # También se aplicará a Cundinamarca
        'BUCARAMANGA': 'Santander',
        'CALI': 'Valle Del Cauca',
        'CARTAGENA': 'Bolívar',
        'CÚCUTA': 'Norte De Santander',
        'FLORENCIA': 'Caquetá',
        'IBAGUÉ': 'Tolima',
        'LETICIA': 'Amazonas',
        'MANIZALES': 'Caldas',
        'MEDELLÍN': 'Antioquia',
        'MONTERÍA': 'Córdoba',
        'NEIVA': 'Huila',
        'PASTO': 'Nariño',
        'PEREIRA': 'Risaralda',
        'POPAYÁN': 'Cauca',
        'QUIBDÓ': 'Chocó',
        'RIOHACHA': 'Guajira',
        'SANTA MARTA': 'Magdalena',
        'SINCELEJO': 'Sucre',
        'TUNJA': 'Boyacá',
        'VALLEDUPAR': 'Cesar',
        'VILLAVICENCIO': 'Meta',
        'YOPAL': 'Casanare',
        'INÍRIDA': 'Guainía',
        'MITÚ': 'Vaupés',
        'MOCOA': 'Putumayo',
        'PUERTO CARREÑO': 'Vichada',
        'SAN JOSÉ DEL GUAVIARE': 'Guaviare',
        'ARAUCA': 'Arauca'
        # San Andrés EXCLUIDO intencionalmente
    }
    
    # Aplicar mapeo
    df_ipc['departamento'] = df_ipc['ciudad'].map(mapeo_ciudad_departamento)
    
    # Filtrar solo registros con departamento mapeado (excluye San Andrés)
    df_ipc = df_ipc[df_ipc['departamento'].notna()].copy()
    
    departamentos_ipc = sorted(df_ipc['departamento'].unique())
    print(f"Departamentos mapeados ({len(departamentos_ipc)}): {departamentos_ipc}")
    
    # 3. DUPLICAR BOGOTÁ PARA CUNDINAMARCA
    print(f"\n=== 3. DUPLICAR BOGOTÁ PARA CUNDINAMARCA ===")
    
    # Crear registros de Cundinamarca con los mismos datos de Bogotá D.C.
    df_bogota = df_ipc[df_ipc['departamento'] == 'Bogotá D.C.'].copy()
    df_cundinamarca = df_bogota.copy()
    df_cundinamarca['departamento'] = 'Cundinamarca'
    
    # Agregar Cundinamarca al dataset
    df_ipc = pd.concat([df_ipc, df_cundinamarca], ignore_index=True)
    
    departamentos_final = sorted(df_ipc['departamento'].unique())
    print(f"Departamentos finales ({len(departamentos_final)}): {departamentos_final}")
    print(f"Registros Bogotá: {len(df_ipc[df_ipc['departamento'] == 'Bogotá D.C.'])}")
    print(f"Registros Cundinamarca: {len(df_ipc[df_ipc['departamento'] == 'Cundinamarca'])}")
    
    # 4. NORMALIZAR MESES
    print(f"\n=== 4. NORMALIZAR MESES ===")
    
    mapeo_meses = {
        'Ene': 'enero', 'Feb': 'febrero', 'Mar': 'marzo', 'Abr': 'abril',
        'May': 'mayo', 'Jun': 'junio', 'Jul': 'julio', 'Ago': 'agosto',
        'Sep': 'septiembre', 'Oct': 'octubre', 'Nov': 'noviembre', 'Dic': 'diciembre'
    }
    
    df_ipc['mes'] = df_ipc['mes'].map(mapeo_meses)
    
    # Verificar años y meses
    años_ipc = sorted(df_ipc['año'].unique())
    meses_ipc = sorted(df_ipc['mes'].unique())
    print(f"Años IPC: {años_ipc}")
    print(f"Meses IPC: {meses_ipc}")
    
    # 5. CARGAR BASE MASTER Y SIMPLIFICAR IPM
    print(f"\n=== 5. CARGAR Y SIMPLIFICAR BASE MASTER ===")
    df_base = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_COMPLETO_DEFINITIVO.csv")
    print(f"Base master: {len(df_base)} registros, {len(df_base.columns)} columnas")
    
    # Eliminar variables IPM innecesarias
    variables_ipm_eliminar = ['IPM_Cabeceras', 'IPM_Centros_Poblados']
    columnas_eliminadas = [col for col in variables_ipm_eliminar if col in df_base.columns]
    df_base = df_base.drop(columns=columnas_eliminadas)
    print(f"Variables IPM eliminadas: {columnas_eliminadas}")
    print(f"Columnas después de eliminar IPM: {len(df_base.columns)}")
    
    # Verificar departamentos en base master
    departamentos_base = sorted(df_base['departamento'].unique())
    print(f"Departamentos en base master ({len(departamentos_base)}): {departamentos_base}")
    
    # 6. AGREGAR COLUMNA IPC
    print(f"\n=== 6. AGREGAR IPC A BASE MASTER ===")
    df_base['IPC_Total'] = np.nan
    
    # 7. CONSOLIDAR IPC
    print(f"\n=== 7. CONSOLIDAR IPC ===")
    
    registros_actualizados = 0
    registros_no_encontrados = 0
    departamentos_actualizados = set()
    
    for _, row_ipc in df_ipc.iterrows():
        departamento = row_ipc['departamento']
        año = row_ipc['año']
        mes = row_ipc['mes']
        ipc_valor = row_ipc['IPC_Total']
        
        if pd.notna(departamento) and pd.notna(año) and pd.notna(mes) and pd.notna(ipc_valor):
            # Buscar registro correspondiente en base master
            mask = (df_base['departamento'] == departamento) & \
                   (df_base['año'] == año) & \
                   (df_base['mes'] == mes)
            
            if mask.any():
                df_base.loc[mask, 'IPC_Total'] = ipc_valor
                registros_actualizados += 1
                departamentos_actualizados.add(departamento)
            else:
                registros_no_encontrados += 1
    
    print(f"Registros IPC actualizados: {registros_actualizados}")
    print(f"Departamentos con IPC: {len(departamentos_actualizados)}")
    print(f"Registros no encontrados: {registros_no_encontrados}")
    
    # 8. VERIFICAR COBERTURA DETALLADA
    print(f"\n=== 8. VERIFICAR COBERTURA IPC ===")
    
    cobertura_ipc_total = df_base['IPC_Total'].notna().sum()
    porcentaje_ipc = cobertura_ipc_total / len(df_base) * 100
    print(f"Cobertura IPC_Total: {cobertura_ipc_total}/{len(df_base)} ({porcentaje_ipc:.1f}%)")
    
    # Por año
    print(f"\nCobertura por año:")
    for año in [2022, 2023, 2024, 2025]:
        df_año = df_base[df_base['año'] == año]
        cobertura_año = df_año['IPC_Total'].notna().sum()
        porcentaje_año = cobertura_año / len(df_año) * 100 if len(df_año) > 0 else 0
        print(f"  {año}: {cobertura_año}/{len(df_año)} ({porcentaje_año:.1f}%)")
    
    # Por departamento
    print(f"\nCobertura por departamento:")
    cobertura_dept = df_base.groupby('departamento').agg({
        'IPC_Total': ['count', lambda x: x.notna().sum()]
    }).reset_index()
    cobertura_dept.columns = ['departamento', 'total', 'con_ipc']
    cobertura_dept['porcentaje'] = cobertura_dept['con_ipc'] / cobertura_dept['total'] * 100
    cobertura_dept = cobertura_dept.sort_values('porcentaje', ascending=False)
    
    # Mostrar departamentos con IPC
    dept_con_ipc = cobertura_dept[cobertura_dept['con_ipc'] > 0]
    print(f"Departamentos con datos IPC ({len(dept_con_ipc)}):")
    for _, row in dept_con_ipc.iterrows():
        print(f"  {row['departamento']}: {row['con_ipc']}/{row['total']} ({row['porcentaje']:.1f}%)")
    
    # Mostrar departamentos sin IPC
    dept_sin_ipc = cobertura_dept[cobertura_dept['con_ipc'] == 0]
    print(f"\nDepartamentos SIN datos IPC ({len(dept_sin_ipc)}):")
    for _, row in dept_sin_ipc.iterrows():
        print(f"  {row['departamento']}")
    
    # 9. VERIFICAR LA GUAJIRA ESPECÍFICAMENTE
    print(f"\n=== 9. VERIFICAR LA GUAJIRA ===")
    df_guajira = df_base[df_base['departamento'] == 'Guajira']
    if len(df_guajira) > 0:
        ipc_guajira = df_guajira['IPC_Total'].notna().sum()
        print(f"La Guajira IPC: {ipc_guajira}/{len(df_guajira)} ({ipc_guajira/len(df_guajira)*100:.1f}%)")
        
        if ipc_guajira > 0:
            print("Ejemplos IPC La Guajira:")
            ejemplos = df_guajira[df_guajira['IPC_Total'].notna()][['año', 'mes', 'IPC_Total']].head(3)
            print(ejemplos.to_string(index=False))
    
    # 10. VERIFICAR BOGOTÁ Y CUNDINAMARCA
    print(f"\n=== 10. VERIFICAR BOGOTÁ Y CUNDINAMARCA ===")
    
    for dept in ['Bogotá D.C.', 'Cundinamarca']:
        df_dept = df_base[df_base['departamento'] == dept]
        ipc_dept = df_dept['IPC_Total'].notna().sum()
        print(f"{dept}: {ipc_dept}/{len(df_dept)} ({ipc_dept/len(df_dept)*100:.1f}%)")
    
    # Verificar que tienen los mismos valores
    df_bogota_vals = df_base[df_base['departamento'] == 'Bogotá D.C.'][['año', 'mes', 'IPC_Total']].sort_values(['año', 'mes'])
    df_cundi_vals = df_base[df_base['departamento'] == 'Cundinamarca'][['año', 'mes', 'IPC_Total']].sort_values(['año', 'mes'])
    
    if len(df_bogota_vals) == len(df_cundi_vals):
        valores_iguales = (df_bogota_vals['IPC_Total'].values == df_cundi_vals['IPC_Total'].values).all()
        print(f"Bogotá y Cundinamarca tienen valores idénticos: {valores_iguales}")
    
    # 11. GUARDAR RESULTADO
    print(f"\n=== 11. GUARDAR RESULTADO ===")
    
    # CSV
    archivo_csv = "data/base de datos central/base_master_2022_2025_FINAL_IPM_IPC_CORRECTO.csv"
    df_base.to_csv(archivo_csv, index=False)
    print(f"Guardado CSV: {archivo_csv}")
    
    # Excel
    archivo_excel = "data/base de datos central/base_master_2022_2025_FINAL_IPM_IPC_CORRECTO.xlsx"
    df_base.to_excel(archivo_excel, index=False)
    print(f"Guardado Excel: {archivo_excel}")
    
    # 12. RESUMEN FINAL
    print(f"\n=== 12. RESUMEN FINAL ===")
    print(f"Registros totales: {len(df_base)}")
    print(f"Columnas totales: {len(df_base.columns)}")
    print(f"IPM simplificado: Solo IPM_Total")
    print(f"IPC agregado: {cobertura_ipc_total} registros ({porcentaje_ipc:.1f}%)")
    print(f"Departamentos con IPC: {len(dept_con_ipc)}/32")
    print(f"Bogotá = Cundinamarca: Sí")
    print(f"San Andrés excluido: Sí")
    
    return df_base

if __name__ == "__main__":
    df_final = procesar_ipc_extrapolacion_correcta()
    print("\n=== PROCESO COMPLETADO ===")

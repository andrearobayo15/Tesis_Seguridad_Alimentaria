#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combinar IPC y FIES en base master final
"""

import pandas as pd
import numpy as np

def combinar_ipc_fies_final():
    """
    Combina las variables IPC y FIES en una base master final
    """
    
    print("=== COMBINACION FINAL IPC + FIES ===")
    
    # 1. CARGAR BASE CON FIES
    print(f"\n=== 1. CARGAR BASE CON FIES ===")
    
    df_base = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_CON_FIES.csv")
    print(f"Base con FIES: {len(df_base)} registros, {len(df_base.columns)} columnas")
    print(f"Período: {df_base['año'].min()}-{df_base['año'].max()}")
    
    # Variables FIES actuales
    fies_vars = [col for col in df_base.columns if 'FIES' in col]
    print(f"Variables FIES: {fies_vars}")
    
    # 2. AGREGAR IPC DESDE ARCHIVO ORIGINAL
    print(f"\n=== 2. AGREGAR IPC ===")
    
    # Cargar datos IPC originales
    df_ipc = pd.read_excel("data/original/IPC todos los departamentos.xls")
    df_ipc.columns = ['año', 'mes', 'ciudad', 'IPC_Total', 'variacion_mensual']
    
    print(f"Datos IPC originales: {len(df_ipc)} registros")
    print(f"Ciudades IPC: {df_ipc['ciudad'].nunique()}")
    
    # Mapeo ciudades a departamentos (usando mapeo corregido)
    mapeo_capitales = {
        'ARMENIA': 'Quindio',
        'BARRANQUILLA': 'Atlantico',
        'BOGOTÁ, D.C.': 'Bogotá',
        'BUCARAMANGA': 'Santander',
        'CALI': 'Valle Del Cauca',
        'CARTAGENA DE INDIAS': 'Bolivar',
        'CÚCUTA': 'Norte De Santander',
        'FLORENCIA': 'Caqueta',
        'IBAGUÉ': 'Tolima',
        'MANIZALES': 'Caldas',
        'MEDELLÍN': 'Antioquia',
        'MONTERÍA': 'Cordoba',
        'NEIVA': 'Huila',
        'PASTO': 'Narino',
        'PEREIRA': 'Risaralda',
        'POPAYÁN': 'Cauca',
        'RIOHACHA': 'Guajira',
        'SANTA MARTA': 'Magdalena',
        'SINCELEJO': 'Sucre',
        'TUNJA': 'Boyaca',
        'VALLEDUPAR': 'Cesar',
        'VILLAVICENCIO': 'Meta'
    }
    
    mapeo_otras_areas = {
        'YOPAL': 'Casanare',
        'INÍRIDA': 'Guainia',
        'PUERTO CARREÑO': 'Vichada',
        'ARAUCA': 'Arauca',
        'LETICIA': 'Amazonas',
        'MITÚ': 'Vaupes',
        'SAN JOSÉ DEL GUAVIARE': 'Guaviare',
        'QUIBDÓ': 'Choco',
        'MOCOA': 'Putumayo'
    }
    
    # Combinar mapeos
    mapeo_completo = {**mapeo_capitales, **mapeo_otras_areas}
    
    # Mapear ciudades individuales
    df_ipc['departamento'] = df_ipc['ciudad'].map(mapeo_completo)
    
    # Procesar "Otras areas urbanas"
    mask_otras = df_ipc['ciudad'] == 'Otras areas urbanas'
    if mask_otras.any():
        df_otras_original = df_ipc[mask_otras].copy()
        df_ipc = df_ipc[~mask_otras]  # Remover originales
        
        # Crear registros para cada departamento en "Otras areas"
        for ciudad, dept in mapeo_otras_areas.items():
            df_dept = df_otras_original.copy()
            df_dept['departamento'] = dept
            df_dept['ciudad'] = f'Otras_areas_{dept}'
            df_ipc = pd.concat([df_ipc, df_dept], ignore_index=True)
    
    # Duplicar Bogotá para Cundinamarca
    df_bogota = df_ipc[df_ipc['departamento'] == 'Bogotá'].copy()
    df_cundinamarca = df_bogota.copy()
    df_cundinamarca['departamento'] = 'Cundinamarca'
    df_ipc = pd.concat([df_ipc, df_cundinamarca], ignore_index=True)
    
    # Normalizar meses
    mapeo_meses = {
        'Ene': 'enero', 'Feb': 'febrero', 'Mar': 'marzo', 'Abr': 'abril',
        'May': 'mayo', 'Jun': 'junio', 'Jul': 'julio', 'Ago': 'agosto',
        'Sep': 'septiembre', 'Oct': 'octubre', 'Nov': 'noviembre', 'Dic': 'diciembre'
    }
    df_ipc['mes'] = df_ipc['mes'].map(mapeo_meses)
    
    print(f"Datos IPC procesados: {len(df_ipc)} registros")
    print(f"Departamentos IPC: {df_ipc['departamento'].nunique()}")
    
    # 3. INTEGRAR IPC EN BASE MASTER
    print(f"\n=== 3. INTEGRAR IPC EN BASE MASTER ===")
    
    # Inicializar IPC si no existe
    if 'IPC_Total' not in df_base.columns:
        df_base['IPC_Total'] = np.nan
    
    registros_actualizados = 0
    
    for _, row_ipc in df_ipc.iterrows():
        departamento = row_ipc['departamento']
        año = row_ipc['año']
        mes = row_ipc['mes']
        ipc_valor = row_ipc['IPC_Total']
        
        if pd.notna(departamento) and pd.notna(año) and pd.notna(mes) and pd.notna(ipc_valor):
            mask = (df_base['departamento'] == departamento) & \
                   (df_base['año'] == año) & \
                   (df_base['mes'] == mes)
            
            if mask.any():
                df_base.loc[mask, 'IPC_Total'] = ipc_valor
                registros_actualizados += 1
    
    print(f"Registros IPC actualizados: {registros_actualizados}")
    
    # 4. VERIFICAR COBERTURA FINAL
    print(f"\n=== 4. VERIFICAR COBERTURA FINAL ===")
    
    # IPM
    if 'IPM_Total' in df_base.columns:
        ipm_cobertura = df_base['IPM_Total'].notna().sum()
        ipm_porcentaje = ipm_cobertura / len(df_base) * 100
        print(f"IPM_Total: {ipm_cobertura}/{len(df_base)} ({ipm_porcentaje:.1f}%)")
    
    # IPC
    ipc_cobertura = df_base['IPC_Total'].notna().sum()
    ipc_porcentaje = ipc_cobertura / len(df_base) * 100
    print(f"IPC_Total: {ipc_cobertura}/{len(df_base)} ({ipc_porcentaje:.1f}%)")
    
    # FIES
    for var in fies_vars:
        fies_cobertura = df_base[var].notna().sum()
        fies_porcentaje = fies_cobertura / len(df_base) * 100
        print(f"{var}: {fies_cobertura}/{len(df_base)} ({fies_porcentaje:.1f}%)")
    
    # 5. COBERTURA POR AÑO
    print(f"\n=== 5. COBERTURA POR AÑO ===")
    
    for año in sorted(df_base['año'].unique()):
        df_año = df_base[df_base['año'] == año]
        print(f"\n{año}:")
        
        if 'IPM_Total' in df_base.columns:
            ipm_año = df_año['IPM_Total'].notna().sum()
            print(f"  IPM_Total: {ipm_año}/{len(df_año)} ({ipm_año/len(df_año)*100:.1f}%)")
        
        ipc_año = df_año['IPC_Total'].notna().sum()
        print(f"  IPC_Total: {ipc_año}/{len(df_año)} ({ipc_año/len(df_año)*100:.1f}%)")
        
        if fies_vars:
            fies_año = df_año[fies_vars[0]].notna().sum()
            print(f"  FIES: {fies_año}/{len(df_año)} ({fies_año/len(df_año)*100:.1f}%)")
    
    # 6. COBERTURA DEPARTAMENTAL
    print(f"\n=== 6. COBERTURA DEPARTAMENTAL ===")
    
    # IPC por departamento
    dept_ipc = df_base.groupby('departamento')['IPC_Total'].apply(lambda x: x.notna().sum()).reset_index()
    dept_ipc.columns = ['departamento', 'registros_ipc']
    dept_con_ipc = dept_ipc[dept_ipc['registros_ipc'] > 0]
    dept_sin_ipc = dept_ipc[dept_ipc['registros_ipc'] == 0]
    
    print(f"Departamentos CON IPC: {len(dept_con_ipc)}/32")
    if len(dept_sin_ipc) > 0:
        print(f"Departamentos SIN IPC: {sorted(dept_sin_ipc['departamento'].tolist())}")
    else:
        print("TODOS los departamentos tienen IPC")
    
    # FIES por departamento
    if fies_vars:
        dept_fies = df_base.groupby('departamento')[fies_vars[0]].apply(lambda x: x.notna().sum()).reset_index()
        dept_fies.columns = ['departamento', 'registros_fies']
        dept_con_fies = dept_fies[dept_fies['registros_fies'] > 0]
        print(f"Departamentos CON FIES: {len(dept_con_fies)}/32")
    
    # 7. GUARDAR RESULTADO FINAL
    print(f"\n=== 7. GUARDAR RESULTADO FINAL ===")
    
    archivo_csv = "data/base de datos central/BASE_MASTER_2022_2025_FINAL_COMPLETA.csv"
    df_base.to_csv(archivo_csv, index=False)
    print(f"Guardado CSV: {archivo_csv}")
    
    archivo_excel = "data/base de datos central/BASE_MASTER_2022_2025_FINAL_COMPLETA.xlsx"
    df_base.to_excel(archivo_excel, index=False)
    print(f"Guardado Excel: {archivo_excel}")
    
    # 8. RESUMEN FINAL
    print(f"\n=== 8. RESUMEN FINAL ===")
    print(f"ESTRUCTURA:")
    print(f"  - Registros: {len(df_base):,}")
    print(f"  - Columnas: {len(df_base.columns)}")
    print(f"  - Departamentos: {df_base['departamento'].nunique()}")
    print(f"  - Período: {df_base['año'].min()}-{df_base['año'].max()}")
    
    print(f"\nVARIABLES PRINCIPALES:")
    
    if 'IPM_Total' in df_base.columns:
        ipm_cobertura = df_base['IPM_Total'].notna().sum()
        ipm_porcentaje = ipm_cobertura / len(df_base) * 100
        print(f"  - IPM_Total: {ipm_cobertura:,} registros ({ipm_porcentaje:.1f}%)")
    
    ipc_cobertura = df_base['IPC_Total'].notna().sum()
    ipc_porcentaje = ipc_cobertura / len(df_base) * 100
    print(f"  - IPC_Total: {ipc_cobertura:,} registros ({ipc_porcentaje:.1f}%)")
    
    if fies_vars:
        fies_cobertura = df_base[fies_vars[0]].notna().sum()
        fies_porcentaje = fies_cobertura / len(df_base) * 100
        print(f"  - Variables FIES ({len(fies_vars)}): {fies_cobertura:,} registros ({fies_porcentaje:.1f}%)")
    
    print(f"\nCOBERTURA DEPARTAMENTAL:")
    print(f"  - IPC: {len(dept_con_ipc)}/32 departamentos")
    if fies_vars:
        print(f"  - FIES: {len(dept_con_fies)}/32 departamentos")
    
    print(f"\nESTADO: BASE MASTER COMPLETA")
    print(f"Variables socioeconomicas clave integradas:")
    print(f"• IPM (Pobreza Multidimensional)")
    print(f"• IPC (Pobreza de Consumo)")  
    print(f"• FIES (Seguridad Alimentaria)")
    print(f"• Variables climáticas y ECV")
    print(f"\nLISTA PARA ANÁLISIS DE TESIS")
    
    return df_base

if __name__ == "__main__":
    df_final = combinar_ipc_fies_final()
    print("\n=== BASE MASTER FINAL COMPLETA CREADA ===")

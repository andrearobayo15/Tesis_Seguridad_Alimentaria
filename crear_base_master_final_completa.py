#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crear base master final completa con IPM + IPC + FIES
"""

import pandas as pd
import numpy as np

def crear_base_master_final_completa():
    """
    Combina la base master con IPC completo y las variables FIES
    """
    
    print("=== BASE MASTER FINAL COMPLETA ===")
    
    # 1. CARGAR BASES
    print(f"\n=== 1. CARGAR BASES ===")
    
    # Base con IPC completo (si existe)
    try:
        df_ipc = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_IPM_IPC_COMPLETO_TODOS.csv")
        print(f"Base con IPC: {len(df_ipc)} registros, {len(df_ipc.columns)} columnas")
        base_principal = df_ipc.copy()
        tiene_ipc = True
    except FileNotFoundError:
        print("Archivo con IPC no encontrado, usando base con FIES")
        df_fies = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_CON_FIES.csv")
        print(f"Base con FIES: {len(df_fies)} registros, {len(df_fies.columns)} columnas")
        base_principal = df_fies.copy()
        tiene_ipc = False
    
    # Base con FIES
    df_fies = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_CON_FIES.csv")
    print(f"Base con FIES: {len(df_fies)} registros, {len(df_fies.columns)} columnas")
    
    # 2. VERIFICAR VARIABLES EXISTENTES
    print(f"\n=== 2. VERIFICAR VARIABLES EXISTENTES ===")
    
    # Variables IPM
    ipm_vars = [col for col in base_principal.columns if 'IPM' in col]
    print(f"Variables IPM: {ipm_vars}")
    
    # Variables IPC
    ipc_vars = [col for col in base_principal.columns if 'IPC' in col]
    print(f"Variables IPC: {ipc_vars}")
    
    # Variables FIES en base principal
    fies_vars_base = [col for col in base_principal.columns if 'FIES' in col]
    print(f"Variables FIES en base principal: {fies_vars_base}")
    
    # Variables FIES disponibles
    fies_vars_disponibles = [col for col in df_fies.columns if 'FIES' in col]
    print(f"Variables FIES disponibles: {fies_vars_disponibles}")
    
    # 3. INTEGRAR VARIABLES FALTANTES
    print(f"\n=== 3. INTEGRAR VARIABLES FALTANTES ===")
    
    if not tiene_ipc:
        print("Integrando IPC desde archivo separado...")
        # Cargar IPC si no está en la base principal
        try:
            df_ipc_sep = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_IPM_IPC_COMPLETO_TODOS.csv")
            if 'IPC_Total' in df_ipc_sep.columns:
                # Merge IPC
                base_principal = base_principal.merge(
                    df_ipc_sep[['departamento', 'año', 'mes', 'IPC_Total']], 
                    on=['departamento', 'año', 'mes'], 
                    how='left'
                )
                print(f"IPC integrado exitosamente")
            else:
                print("IPC_Total no encontrado en archivo separado")
        except FileNotFoundError:
            print("Archivo IPC separado no encontrado")
    
    if not fies_vars_base:
        print("Integrando variables FIES...")
        # Merge FIES variables
        fies_columns = ['departamento', 'año', 'mes'] + fies_vars_disponibles
        base_principal = base_principal.merge(
            df_fies[fies_columns], 
            on=['departamento', 'año', 'mes'], 
            how='left'
        )
        print(f"Variables FIES integradas: {fies_vars_disponibles}")
    
    # 4. VERIFICAR COBERTURA FINAL
    print(f"\n=== 4. VERIFICAR COBERTURA FINAL ===")
    
    # IPM
    if 'IPM_Total' in base_principal.columns:
        ipm_cobertura = base_principal['IPM_Total'].notna().sum()
        ipm_porcentaje = ipm_cobertura / len(base_principal) * 100
        print(f"IPM_Total: {ipm_cobertura}/{len(base_principal)} ({ipm_porcentaje:.1f}%)")
    
    # IPC
    if 'IPC_Total' in base_principal.columns:
        ipc_cobertura = base_principal['IPC_Total'].notna().sum()
        ipc_porcentaje = ipc_cobertura / len(base_principal) * 100
        print(f"IPC_Total: {ipc_cobertura}/{len(base_principal)} ({ipc_porcentaje:.1f}%)")
    
    # FIES
    for var in fies_vars_disponibles:
        if var in base_principal.columns:
            fies_cobertura = base_principal[var].notna().sum()
            fies_porcentaje = fies_cobertura / len(base_principal) * 100
            print(f"{var}: {fies_cobertura}/{len(base_principal)} ({fies_porcentaje:.1f}%)")
    
    # 5. COBERTURA POR AÑO
    print(f"\n=== 5. COBERTURA POR AÑO ===")
    
    for año in sorted(base_principal['año'].unique()):
        df_año = base_principal[base_principal['año'] == año]
        print(f"\\n{año}:")
        
        if 'IPM_Total' in base_principal.columns:
            ipm_año = df_año['IPM_Total'].notna().sum()
            print(f"  IPM_Total: {ipm_año}/{len(df_año)} ({ipm_año/len(df_año)*100:.1f}%)")
        
        if 'IPC_Total' in base_principal.columns:
            ipc_año = df_año['IPC_Total'].notna().sum()
            print(f"  IPC_Total: {ipc_año}/{len(df_año)} ({ipc_año/len(df_año)*100:.1f}%)")
        
        if 'FIES_leve_moderado' in base_principal.columns:
            fies_año = df_año['FIES_leve_moderado'].notna().sum()
            print(f"  FIES: {fies_año}/{len(df_año)} ({fies_año/len(df_año)*100:.1f}%)")
    
    # 6. GUARDAR RESULTADO FINAL
    print(f"\n=== 6. GUARDAR RESULTADO FINAL ===")
    
    archivo_csv = "data/base de datos central/BASE_MASTER_2022_2025_FINAL_COMPLETA.csv"
    base_principal.to_csv(archivo_csv, index=False)
    print(f"Guardado CSV: {archivo_csv}")
    
    archivo_excel = "data/base de datos central/BASE_MASTER_2022_2025_FINAL_COMPLETA.xlsx"
    base_principal.to_excel(archivo_excel, index=False)
    print(f"Guardado Excel: {archivo_excel}")
    
    # 7. RESUMEN FINAL COMPLETO
    print(f"\n=== 7. RESUMEN FINAL COMPLETO ===")
    print(f"📊 ESTRUCTURA:")
    print(f"  - Registros: {len(base_principal):,}")
    print(f"  - Columnas: {len(base_principal.columns)}")
    print(f"  - Departamentos: {base_principal['departamento'].nunique()}")
    print(f"  - Período: {base_principal['año'].min()}-{base_principal['año'].max()}")
    
    print(f"\\n📈 VARIABLES PRINCIPALES:")
    
    # IPM
    if 'IPM_Total' in base_principal.columns:
        ipm_cobertura = base_principal['IPM_Total'].notna().sum()
        ipm_porcentaje = ipm_cobertura / len(base_principal) * 100
        print(f"  - IPM_Total: {ipm_cobertura:,} registros ({ipm_porcentaje:.1f}%)")
    
    # IPC
    if 'IPC_Total' in base_principal.columns:
        ipc_cobertura = base_principal['IPC_Total'].notna().sum()
        ipc_porcentaje = ipc_cobertura / len(base_principal) * 100
        print(f"  - IPC_Total: {ipc_cobertura:,} registros ({ipc_porcentaje:.1f}%)")
    
    # FIES
    fies_vars_finales = [col for col in base_principal.columns if 'FIES' in col]
    if fies_vars_finales:
        fies_cobertura = base_principal[fies_vars_finales[0]].notna().sum()
        fies_porcentaje = fies_cobertura / len(base_principal) * 100
        print(f"  - Variables FIES ({len(fies_vars_finales)}): {fies_cobertura:,} registros ({fies_porcentaje:.1f}%)")
    
    print(f"\\n✅ COBERTURA DEPARTAMENTAL:")
    
    # Departamentos con cada variable
    if 'IPM_Total' in base_principal.columns:
        dept_ipm = base_principal.groupby('departamento')['IPM_Total'].apply(lambda x: x.notna().any()).sum()
        print(f"  - IPM: {dept_ipm}/32 departamentos")
    
    if 'IPC_Total' in base_principal.columns:
        dept_ipc = base_principal.groupby('departamento')['IPC_Total'].apply(lambda x: x.notna().any()).sum()
        print(f"  - IPC: {dept_ipc}/32 departamentos")
    
    if fies_vars_finales:
        dept_fies = base_principal.groupby('departamento')[fies_vars_finales[0]].apply(lambda x: x.notna().any()).sum()
        print(f"  - FIES: {dept_fies}/32 departamentos")
    
    print(f"\\n🎯 ESTADO FINAL:")
    print(f"  Base master completa con variables socioeconomicas clave:")
    print(f"  • IPM (Índice Pobreza Multidimensional)")
    print(f"  • IPC (Índice Pobreza Consumo)")  
    print(f"  • FIES (Seguridad Alimentaria)")
    print(f"  • Variables climáticas y socioeconómicas adicionales")
    print(f"  \\n  LISTA PARA ANÁLISIS DE TESIS")
    
    return base_principal

if __name__ == "__main__":
    df_final = crear_base_master_final_completa()
    print("\\n=== BASE MASTER FINAL COMPLETA CREADA ===")

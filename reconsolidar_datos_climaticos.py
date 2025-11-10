#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-consolidación de datos climáticos para resolver el problema de cobertura cero
"""

import pandas as pd
import numpy as np

def reconsolidar_datos_climaticos():
    """
    Re-consolida los datos climáticos en la base master
    """
    
    print("=== RE-CONSOLIDACION DE DATOS CLIMATICOS ===")
    
    # Cargar base master corregida (sin datos climáticos)
    df_master = pd.read_csv("data/base de datos central/base_master_2022_2025_FINAL_SIMPLIFICADA_CORREGIDA.csv")
    
    print(f"Base master cargada: {len(df_master)} registros")
    print(f"Departamentos: {len(df_master['departamento'].unique())}")
    print(f"Período: {df_master['año'].min()}-{df_master['año'].max()}")
    
    # Verificar datos climáticos actuales
    clima_vars = ['precipitacion_promedio', 'temperatura_promedio', 'ndvi_promedio']
    print(f"\nCobertura climática actual:")
    for var in clima_vars:
        datos_actuales = df_master[var].notna().sum()
        print(f"  {var}: {datos_actuales}/{len(df_master)} ({datos_actuales/len(df_master)*100:.1f}%)")
    
    # Cargar archivos procesados
    archivos_climaticos = {
        'precipitacion_promedio': 'data/procesado/precipitacion_filtrada.csv',
        'temperatura_promedio': 'data/procesado/lst_completo.csv',
        'ndvi_promedio': 'data/procesado/ndvi_final.csv'
    }
    
    # Crear copia para modificar
    df_consolidado = df_master.copy()
    
    # Procesar cada variable climática
    for variable, archivo in archivos_climaticos.items():
        print(f"\n--- PROCESANDO {variable} ---")
        
        try:
            # Cargar archivo procesado
            df_clima = pd.read_csv(archivo)
            print(f"Archivo cargado: {len(df_clima)} registros")
            print(f"Departamentos en archivo: {len(df_clima['departamento'].unique())}")
            
            # Verificar columnas
            print(f"Columnas disponibles: {list(df_clima.columns)}")
            
            # Determinar columna de valor según el archivo
            if 'precipitacion_media' in df_clima.columns:
                columna_valor = 'precipitacion_media'
            elif 'temperatura_media' in df_clima.columns:
                columna_valor = 'temperatura_media'
            elif 'ndvi_medio' in df_clima.columns:
                columna_valor = 'ndvi_medio'
            elif 'ndvi_media' in df_clima.columns:
                columna_valor = 'ndvi_media'
            elif 'mean' in df_clima.columns:
                columna_valor = 'mean'
            else:
                print(f"  ERROR: No se encontró columna de valor en {archivo}")
                continue
            
            print(f"  Usando columna: {columna_valor}")
            
            # Convertir fecha si es necesario
            if 'fecha' in df_clima.columns:
                df_clima['fecha'] = pd.to_datetime(df_clima['fecha'])
                df_clima['año'] = df_clima['fecha'].dt.year
                df_clima['mes'] = df_clima['fecha'].dt.month
            
            # Agrupar por departamento, año y mes para obtener promedio mensual
            df_mensual = df_clima.groupby(['departamento', 'año', 'mes'])[columna_valor].mean().reset_index()
            df_mensual.rename(columns={columna_valor: variable}, inplace=True)
            
            print(f"  Datos mensuales agrupados: {len(df_mensual)} registros")
            print(f"  Rango temporal: {df_mensual['año'].min()}-{df_mensual['año'].max()}")
            
            # Verificar algunos datos
            muestra = df_mensual.head()
            print(f"  Muestra de datos:")
            for _, row in muestra.iterrows():
                print(f"    {row['departamento']} {row['año']}-{row['mes']:02d}: {row[variable]:.2f}")
            
            # Hacer merge con la base master
            print(f"  Haciendo merge con base master...")
            
            # Merge usando departamento, año y mes
            df_consolidado = df_consolidado.merge(
                df_mensual[['departamento', 'año', 'mes', variable]], 
                on=['departamento', 'año', 'mes'], 
                how='left',
                suffixes=('', '_nuevo')
            )
            
            # Si hay columna duplicada, usar la nueva
            if f'{variable}_nuevo' in df_consolidado.columns:
                # Actualizar valores donde hay datos nuevos
                mask_nuevos = df_consolidado[f'{variable}_nuevo'].notna()
                df_consolidado.loc[mask_nuevos, variable] = df_consolidado.loc[mask_nuevos, f'{variable}_nuevo']
                df_consolidado.drop(columns=[f'{variable}_nuevo'], inplace=True)
                print(f"  Actualizados {mask_nuevos.sum()} registros con datos nuevos")
            
            # Verificar cobertura después del merge
            datos_nuevos = df_consolidado[variable].notna().sum()
            print(f"  Cobertura después del merge: {datos_nuevos}/{len(df_consolidado)} ({datos_nuevos/len(df_consolidado)*100:.1f}%)")
            
        except Exception as e:
            print(f"  ERROR procesando {archivo}: {e}")
            continue
    
    # Verificar cobertura final
    print(f"\n=== COBERTURA FINAL ===")
    for var in clima_vars:
        datos_finales = df_consolidado[var].notna().sum()
        print(f"  {var}: {datos_finales}/{len(df_consolidado)} ({datos_finales/len(df_consolidado)*100:.1f}%)")
    
    # Verificar departamentos específicos que tenían problemas
    departamentos_problema = ['Atlantico', 'Bolivar', 'Boyaca', 'Choco', 'Caqueta', 
                             'Narino', 'Cordoba', 'Guainia', 'Quindio', 'Vaupes']
    
    print(f"\n=== VERIFICACION DEPARTAMENTOS PROBLEMA ===")
    for dept in departamentos_problema:
        df_dept = df_consolidado[df_consolidado['departamento'] == dept]
        if len(df_dept) > 0:
            print(f"\n{dept}: {len(df_dept)} registros")
            for var in clima_vars:
                datos = df_dept[var].notna().sum()
                print(f"  {var}: {datos}/{len(df_dept)} ({datos/len(df_dept)*100:.1f}%)")
        else:
            print(f"\n{dept}: NO ENCONTRADO")
    
    # Guardar base consolidada
    archivo_salida = "data/base de datos central/base_master_2022_2025_FINAL_RECONSOLIDADA.csv"
    df_consolidado.to_csv(archivo_salida, index=False, encoding='utf-8')
    
    print(f"\nOK Base reconsolidada guardada en: {archivo_salida}")
    print(f"  Registros: {len(df_consolidado)}")
    print(f"  Departamentos únicos: {len(df_consolidado['departamento'].unique())}")
    
    return df_consolidado

if __name__ == "__main__":
    df_final = reconsolidar_datos_climaticos()
    print("\n=== RE-CONSOLIDACION COMPLETADA ===")

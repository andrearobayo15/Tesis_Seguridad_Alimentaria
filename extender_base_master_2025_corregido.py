#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script corregido para extender la base master hasta diciembre 2025
Incluye datos climáticos disponibles hasta junio-julio 2025
"""

import pandas as pd
from pathlib import Path
import numpy as np

def extender_base_master_2025_corregido():
    """
    Extiende la base master hasta diciembre 2025 con nombres de columnas correctos
    """
    
    print("=== EXTENDIENDO BASE MASTER HASTA DICIEMBRE 2025 (CORREGIDO) ===")
    
    # Cargar base master actual
    archivo_base = Path("data/base de datos central/base_master_consolidada_v2_cronologica.csv")
    
    if not archivo_base.exists():
        print(f"ERROR: Archivo no encontrado: {archivo_base}")
        return
    
    print(f"Cargando base actual: {archivo_base}")
    df_base = pd.read_csv(archivo_base)
    
    print(f"Base actual: {len(df_base):,} registros")
    print(f"Período actual: {df_base['año'].min()}-{df_base['mes'].min():02d} a {df_base['año'].max()}-{df_base['mes'].max():02d}")
    
    # Obtener lista de departamentos
    departamentos = sorted(df_base['departamento'].unique())
    print(f"Departamentos: {len(departamentos)}")
    
    # Crear estructura temporal extendida hasta diciembre 2025
    print("\nCreando estructura temporal extendida...")
    
    registros_nuevos = []
    
    # Meses que faltan en 2024 (septiembre a diciembre)
    for mes in range(9, 13):  # septiembre a diciembre 2024
        for depto in departamentos:
            fecha = f"2024-{mes:02d}-01"
            mes_nombre = pd.to_datetime(fecha).strftime('%B')
            
            registro = {
                'departamento': depto,
                'año': 2024,
                'mes': mes,
                'fecha': fecha,
                'mes_nombre': mes_nombre
            }
            
            # Inicializar todas las otras columnas con NaN
            for col in df_base.columns:
                if col not in registro:
                    registro[col] = np.nan
            
            registros_nuevos.append(registro)
    
    # Todo el año 2025 (enero a diciembre)
    for mes in range(1, 13):  # enero a diciembre 2025
        for depto in departamentos:
            fecha = f"2025-{mes:02d}-01"
            mes_nombre = pd.to_datetime(fecha).strftime('%B')
            
            registro = {
                'departamento': depto,
                'año': 2025,
                'mes': mes,
                'fecha': fecha,
                'mes_nombre': mes_nombre
            }
            
            # Inicializar todas las otras columnas con NaN
            for col in df_base.columns:
                if col not in registro:
                    registro[col] = np.nan
            
            registros_nuevos.append(registro)
    
    print(f"Registros nuevos creados: {len(registros_nuevos):,}")
    
    # Crear DataFrame con registros nuevos
    df_nuevos = pd.DataFrame(registros_nuevos)
    
    # Combinar con base existente
    print("Combinando con base existente...")
    df_extendido = pd.concat([df_base, df_nuevos], ignore_index=True)
    
    # Reordenar cronológicamente
    print("Reordenando cronológicamente...")
    df_extendido = df_extendido.sort_values(['año', 'mes', 'departamento']).reset_index(drop=True)
    
    print(f"Base extendida: {len(df_extendido):,} registros")
    print(f"Nuevo período: {df_extendido['año'].min()}-{df_extendido['mes'].min():02d} a {df_extendido['año'].max()}-{df_extendido['mes'].max():02d}")
    
    # Ahora cargar datos climáticos disponibles para 2024-2025
    print("\n=== CARGANDO DATOS CLIMATICOS 2024-2025 ===")
    
    # 1. Precipitación (hasta mayo 2025)
    try:
        print("Cargando precipitación...")
        df_prec = pd.read_csv("data/procesado/precipitacion_filtrada.csv")
        
        # Verificar columnas disponibles
        print(f"Columnas precipitación: {list(df_prec.columns)}")
        
        df_prec['fecha_dt'] = pd.to_datetime(df_prec['fecha'])
        df_prec['año'] = df_prec['fecha_dt'].dt.year
        df_prec['mes'] = df_prec['fecha_dt'].dt.month
        
        # Filtrar datos 2024-2025
        prec_2024_2025 = df_prec[df_prec['año'].isin([2024, 2025])].copy()
        
        print(f"Precipitación 2024-2025: {len(prec_2024_2025):,} registros")
        print(f"Período precipitación: {prec_2024_2025['fecha'].min()} a {prec_2024_2025['fecha'].max()}")
        
        # Agrupar por departamento, año, mes para obtener promedios mensuales
        if 'precipitacion_media' in prec_2024_2025.columns:
            prec_mensual = prec_2024_2025.groupby(['departamento', 'año', 'mes']).agg({
                'precipitacion_media': ['mean', 'sum', 'std']
            }).reset_index()
            
            # Aplanar columnas
            prec_mensual.columns = ['departamento', 'año', 'mes', 'precipitacion_promedio', 'precipitacion_total', 'precipitacion_std']
            
            print(f"Datos precipitación mensuales: {len(prec_mensual)} registros")
            
            # Merge con base extendida
            df_extendido = df_extendido.merge(
                prec_mensual,
                on=['departamento', 'año', 'mes'],
                how='left',
                suffixes=('', '_new')
            )
            
            # Actualizar valores donde hay datos nuevos
            for col in ['precipitacion_promedio', 'precipitacion_total', 'precipitacion_std']:
                col_new = f"{col}_new"
                if col_new in df_extendido.columns:
                    mask = df_extendido[col_new].notna()
                    df_extendido.loc[mask, col] = df_extendido.loc[mask, col_new]
                    df_extendido.drop(columns=[col_new], inplace=True)
        
    except Exception as e:
        print(f"Error cargando precipitación: {e}")
    
    # 2. Temperatura LST (hasta julio 2025)
    try:
        print("Cargando temperatura...")
        df_lst = pd.read_csv("data/procesado/lst_completo.csv")
        
        # Verificar columnas disponibles
        print(f"Columnas temperatura: {list(df_lst.columns)}")
        
        df_lst['fecha_dt'] = pd.to_datetime(df_lst['fecha'])
        df_lst['año'] = df_lst['fecha_dt'].dt.year
        df_lst['mes'] = df_lst['fecha_dt'].dt.month
        
        # Filtrar datos 2024-2025
        lst_2024_2025 = df_lst[df_lst['año'].isin([2024, 2025])].copy()
        
        print(f"Temperatura 2024-2025: {len(lst_2024_2025):,} registros")
        print(f"Período temperatura: {lst_2024_2025['fecha'].min()} a {lst_2024_2025['fecha'].max()}")
        
        # Identificar columnas de temperatura
        temp_cols = [col for col in lst_2024_2025.columns if 'temperatura' in col.lower() or 'lst' in col.lower()]
        print(f"Columnas temperatura encontradas: {temp_cols}")
        
        if temp_cols:
            # Agrupar por departamento, año, mes
            agg_dict = {}
            for col in temp_cols:
                if col not in ['departamento', 'fecha', 'año', 'mes']:
                    agg_dict[col] = ['mean', 'min', 'max', 'std']
            
            if agg_dict:
                temp_mensual = lst_2024_2025.groupby(['departamento', 'año', 'mes']).agg(agg_dict).reset_index()
                
                # Aplanar columnas
                new_cols = ['departamento', 'año', 'mes']
                for col in temp_cols:
                    if col in agg_dict:
                        new_cols.extend([f"{col}_promedio", f"{col}_min", f"{col}_max", f"{col}_std"])
                
                temp_mensual.columns = new_cols[:len(temp_mensual.columns)]
                
                print(f"Datos temperatura mensuales: {len(temp_mensual)} registros")
                
                # Merge con base extendida
                df_extendido = df_extendido.merge(
                    temp_mensual,
                    on=['departamento', 'año', 'mes'],
                    how='left',
                    suffixes=('', '_new')
                )
                
                # Actualizar valores donde hay datos nuevos
                for col in temp_mensual.columns:
                    if col not in ['departamento', 'año', 'mes']:
                        col_new = f"{col}_new"
                        if col_new in df_extendido.columns:
                            mask = df_extendido[col_new].notna()
                            df_extendido.loc[mask, col] = df_extendido.loc[mask, col_new]
                            df_extendido.drop(columns=[col_new], inplace=True)
        
    except Exception as e:
        print(f"Error cargando temperatura: {e}")
    
    # 3. NDVI (hasta junio 2025)
    try:
        print("Cargando NDVI...")
        df_ndvi = pd.read_csv("data/procesado/ndvi_final.csv")
        
        # Verificar columnas disponibles
        print(f"Columnas NDVI: {list(df_ndvi.columns)}")
        
        df_ndvi['fecha_dt'] = pd.to_datetime(df_ndvi['fecha'])
        df_ndvi['año'] = df_ndvi['fecha_dt'].dt.year
        df_ndvi['mes'] = df_ndvi['fecha_dt'].dt.month
        
        # Filtrar datos 2024-2025
        ndvi_2024_2025 = df_ndvi[df_ndvi['año'].isin([2024, 2025])].copy()
        
        print(f"NDVI 2024-2025: {len(ndvi_2024_2025):,} registros")
        print(f"Período NDVI: {ndvi_2024_2025['fecha'].min()} a {ndvi_2024_2025['fecha'].max()}")
        
        # Identificar columnas NDVI
        ndvi_cols = [col for col in ndvi_2024_2025.columns if 'ndvi' in col.lower()]
        print(f"Columnas NDVI encontradas: {ndvi_cols}")
        
        if ndvi_cols:
            # Agrupar por departamento, año, mes
            agg_dict = {}
            for col in ndvi_cols:
                if col not in ['departamento', 'fecha', 'año', 'mes']:
                    agg_dict[col] = ['mean', 'min', 'max', 'std']
            
            if agg_dict:
                ndvi_mensual = ndvi_2024_2025.groupby(['departamento', 'año', 'mes']).agg(agg_dict).reset_index()
                
                # Aplanar columnas
                new_cols = ['departamento', 'año', 'mes']
                for col in ndvi_cols:
                    if col in agg_dict:
                        new_cols.extend([f"{col}_promedio", f"{col}_min", f"{col}_max", f"{col}_std"])
                
                ndvi_mensual.columns = new_cols[:len(ndvi_mensual.columns)]
                
                print(f"Datos NDVI mensuales: {len(ndvi_mensual)} registros")
                
                # Merge con base extendida
                df_extendido = df_extendido.merge(
                    ndvi_mensual,
                    on=['departamento', 'año', 'mes'],
                    how='left',
                    suffixes=('', '_new')
                )
                
                # Actualizar valores donde hay datos nuevos
                for col in ndvi_mensual.columns:
                    if col not in ['departamento', 'año', 'mes']:
                        col_new = f"{col}_new"
                        if col_new in df_extendido.columns:
                            mask = df_extendido[col_new].notna()
                            df_extendido.loc[mask, col] = df_extendido.loc[mask, col_new]
                            df_extendido.drop(columns=[col_new], inplace=True)
        
    except Exception as e:
        print(f"Error cargando NDVI: {e}")
    
    # Verificar estructura final
    print(f"\n=== VERIFICACION ESTRUCTURA FINAL ===")
    print(f"Total registros: {len(df_extendido):,}")
    print(f"Total variables: {len(df_extendido.columns)}")
    print(f"Período completo: {df_extendido['año'].min()}-{df_extendido['mes'].min():02d} a {df_extendido['año'].max()}-{df_extendido['mes'].max():02d}")
    
    # Distribución por año
    print("\nDistribución por año:")
    for año in sorted(df_extendido['año'].unique()):
        registros_año = len(df_extendido[df_extendido['año'] == año])
        meses_año = len(df_extendido[df_extendido['año'] == año]['mes'].unique())
        print(f"{año}: {registros_año:,} registros ({meses_año} meses)")
    
    # Verificar datos climáticos integrados
    print("\n=== VERIFICACION DATOS CLIMATICOS INTEGRADOS ===")
    
    # Verificar 2025
    df_2025 = df_extendido[df_extendido['año'] == 2025]
    
    # Buscar columnas climáticas
    clima_cols = []
    for col in df_extendido.columns:
        if any(keyword in col.lower() for keyword in ['precipitacion', 'temperatura', 'ndvi', 'lst']):
            clima_cols.append(col)
    
    print(f"Columnas climáticas encontradas: {clima_cols[:10]}...")  # Mostrar solo las primeras 10
    
    for col in clima_cols[:5]:  # Verificar las primeras 5
        count = df_2025[col].notna().sum()
        print(f"{col}: {count}/{len(df_2025)} ({count/len(df_2025)*100:.1f}%)")
    
    # Guardar base extendida
    archivo_salida = Path("data/base de datos central/base_master_consolidada_2022_2025_completa.csv")
    
    print(f"\nGuardando base extendida: {archivo_salida}")
    df_extendido.to_csv(archivo_salida, index=False)
    
    # También crear versión Excel
    archivo_excel = Path("data/base de datos central/base_master_consolidada_2022_2025_completa.xlsx")
    
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_extendido.to_excel(writer, sheet_name='Base_Master_2022_2025', index=False)
        
        # Hoja de cobertura temporal
        cobertura = []
        for año in sorted(df_extendido['año'].unique()):
            for mes in sorted(df_extendido[df_extendido['año'] == año]['mes'].unique()):
                subset = df_extendido[(df_extendido['año'] == año) & (df_extendido['mes'] == mes)]
                
                # Contar datos disponibles para variables clave
                datos_fies = subset['Comio_menos'].notna().sum() if 'Comio_menos' in subset.columns else 0
                
                # Buscar primera columna climática disponible
                datos_clima = 0
                for col in clima_cols:
                    if col in subset.columns:
                        datos_clima = subset[col].notna().sum()
                        break
                
                cobertura.append({
                    'año': año,
                    'mes': mes,
                    'fecha': f"{año}-{mes:02d}",
                    'registros_total': len(subset),
                    'datos_fies': datos_fies,
                    'datos_climaticos': datos_clima,
                    'cobertura_fies': f"{datos_fies/32*100:.1f}%",
                    'cobertura_climaticos': f"{datos_clima/32*100:.1f}%"
                })
        
        df_cobertura = pd.DataFrame(cobertura)
        df_cobertura.to_excel(writer, sheet_name='Cobertura_Temporal', index=False)
    
    print(f"Versión Excel guardada: {archivo_excel}")
    
    # Mostrar ejemplo de datos climáticos disponibles
    print(f"\n=== EJEMPLO DATOS CLIMATICOS 2025 ===")
    ejemplo_2025 = df_extendido[(df_extendido['año'] == 2025) & (df_extendido['departamento'] == 'Amazonas')]
    
    # Mostrar algunas columnas climáticas
    cols_ejemplo = ['fecha', 'departamento'] + clima_cols[:3]
    cols_disponibles = [col for col in cols_ejemplo if col in ejemplo_2025.columns]
    
    print("Columnas a mostrar:", cols_disponibles)
    print(ejemplo_2025[cols_disponibles].head(6))
    
    print(f"\nBASE MASTER EXTENDIDA HASTA DICIEMBRE 2025 COMPLETADA!")
    print(f"Archivo principal: {archivo_salida}")
    
    return df_extendido

if __name__ == "__main__":
    df_resultado = extender_base_master_2025_corregido()

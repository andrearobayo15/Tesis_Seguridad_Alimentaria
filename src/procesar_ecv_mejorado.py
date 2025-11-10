"""
Script mejorado para procesar el archivo ECV y convertir cada hoja a CSV con mejor estructura.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Configuración de rutas
data_path = Path('data')
procesado_path = data_path / 'procesado'
archivo_ecv = data_path / 'original' / 'anex-ECV-Series-2024.xlsx'

# Crear carpeta procesado si no existe
procesado_path.mkdir(exist_ok=True)

def procesar_hoja(df, nombre_hoja):
    """
    Procesa una hoja del Excel, detectando celdas combinadas y estructurando los datos.
    """
    try:
        # Eliminar filas vacías al inicio
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Identificar la fila de años (segunda fila)
        if len(df) < 2:
            print(f'La hoja {nombre_hoja} no tiene suficientes filas para procesar.')
            return None
            
        # Encontrar la fila de encabezados (contiene "Total viviendas")
        idx_encabezados = df[df.iloc[:, 1] == 'Total viviendas'].index[0]
        
        # Obtener los nombres de las columnas
        columnas = df.iloc[idx_encabezados].tolist()
        
        # Crear lista para almacenar los datos
        datos = []
        
        # Procesar cada fila de datos (desde la fila después de los encabezados)
        for idx in range(idx_encabezados + 1, len(df)):
            fila = df.iloc[idx]
            departamento = fila[0]
            
            # Obtener los años y medidas
            for i in range(1, len(columnas), 2):
                if i + 1 < len(columnas):
                    año = columnas[i]
                    medida = columnas[i + 1]
                    valor = fila[i + 1]
                    
                    if pd.notna(valor):
                        # Agregar datos a la lista
                        datos.append({
                            'Departamento': departamento,
                            'Año': año,
                            'Medida': medida,
                            'Valor': valor
                        })
        
        # Crear DataFrame final
        df_datos = pd.DataFrame(datos)
        
        # Eliminar filas con valores NaN
        df_datos = df_datos.dropna()
        
        # Guardar como CSV
        nombre_csv = f'{nombre_hoja}_mejorado.csv'
        ruta_csv = procesado_path / nombre_csv
        df_datos.to_csv(ruta_csv, index=False, encoding='utf-8')
        print(f'Archivo guardado: {ruta_csv}')
        
        # Mostrar información básica
        print(f'\nInformación de la hoja {nombre_hoja}:')
        print(f'Filas: {len(df_datos)}')
        print(f'Columnas: {df_datos.columns.tolist()}')
        
        return df_datos
        
    except Exception as e:
        print(f'Error al procesar la hoja {nombre_hoja}: {e}')
        return None

# Leer el archivo Excel
print(f'\nLeyendo archivo: {archivo_ecv.name}')
try:
    with pd.ExcelFile(archivo_ecv) as xls:
        # Obtener lista de hojas
        hojas = xls.sheet_names
        print(f'\nHojas encontradas: {hojas}')
        
        # Procesar cada hoja
        for hoja in hojas:
            print(f'\nProcesando hoja: {hoja}')
            try:
                # Leer la hoja
                df = pd.read_excel(xls, sheet_name=hoja)
                
                # Procesar la hoja
                df_procesado = procesar_hoja(df, hoja)
                
                if df_procesado is not None:
                    # Mostrar información básica
                    print(f'\nInformación de la hoja {hoja}:')
                    print(f'Filas: {len(df_procesado)}')
                    print(f'Columnas: {df_procesado.columns.tolist()}')
                    
            except Exception as e:
                print(f'Error al procesar la hoja {hoja}: {e}')
                
    print('\nProceso completado.')
    
except Exception as e:
    print(f'Error al leer el archivo Excel: {e}')

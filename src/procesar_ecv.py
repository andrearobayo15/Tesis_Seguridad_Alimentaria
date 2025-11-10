"""
Script para procesar el archivo ECV y convertir cada hoja a CSV.
"""

import pandas as pd
from pathlib import Path

# Configuración de rutas
data_path = Path('data')
procesado_path = data_path / 'procesado'
archivo_ecv = data_path / 'original' / 'anex-ECV-Series-2024.xlsx'

# Crear carpeta procesado si no existe
procesado_path.mkdir(exist_ok=True)

# Leer el archivo Excel
print(f'\nLeyendo archivo: {archivo_ecv.name}')
try:
    # Leer todas las hojas del Excel
    with pd.ExcelFile(archivo_ecv) as xls:
        # Obtener lista de hojas
        hojas = xls.sheet_names
        print(f'\nHojas encontradas: {hojas}')
        
        # Procesar cada hoja
        for i, hoja in enumerate(hojas, 1):
            print(f'\nProcesando hoja {i}: {hoja}')
            try:
                # Leer la hoja
                df = pd.read_excel(xls, sheet_name=hoja)
                
                # Limpiar el DataFrame
                # Eliminar filas vacías al inicio
                df = df.dropna(how='all').reset_index(drop=True)
                
                # Guardar como CSV
                nombre_csv = f'{hoja}.csv'
                ruta_csv = procesado_path / nombre_csv
                df.to_csv(ruta_csv, index=False, encoding='utf-8')
                print(f'Archivo guardado: {ruta_csv}')
                
                # Mostrar información básica
                print(f'\nInformación de la hoja {hoja}:')
                print(f'Filas: {len(df)}')
                print(f'Columnas: {len(df.columns)}')
                print(f'Columnas: {df.columns.tolist()}')
                
            except Exception as e:
                print(f'Error al procesar la hoja {hoja}: {e}')
                
    print('\nProceso completado.')
    
except Exception as e:
    print(f'Error al leer el archivo Excel: {e}')

import pandas as pd
import chardet
from pathlib import Path

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

# Rutas de los archivos
input_file = Path(r'data\original\BaseDatos-SIPSA_P-Mensual-2022\mensual 22.csv')

# Detectar la codificación del archivo
print(f"Detectando codificación del archivo: {input_file}")
encoding = detect_encoding(input_file)
print(f"Codificación detectada: {encoding}")

# Leer el archivo CSV con la codificación detectada
print(f"\nLeyendo archivo: {input_file}")
try:
    df = pd.read_csv(input_file, encoding=encoding, low_memory=False, on_bad_lines='warn')
    
    # Mostrar información básica
    print("\nInformación del DataFrame:")
    print(df.info())
    
    # Mostrar las primeras filas
    print("\nPrimeras filas del DataFrame:")
    print(df.head().to_string())
    
    # Mostrar columnas disponibles
    print("\nColumnas disponibles:")
    for col in df.columns:
        print(f"- {col}")
    
    # Verificar si hay columnas relevantes para alimentos
    posibles_columnas_alimentos = ['alimento', 'producto', 'nombre', 'descripcion', 'item']
    columnas_relevantes = [col for col in df.columns if any(x in col.lower() for x in posibles_columnas_alimentos)]
    
    if columnas_relevantes:
        print("\nColumnas que podrían contener información de alimentos:")
        for col in columnas_relevantes:
            print(f"- {col} (valores únicos: {df[col].nunique()})")
    
    # Verificar columnas de precios
    posibles_columnas_precios = ['precio', 'valor', 'costo', 'price']
    columnas_precios = [col for col in df.columns if any(x in col.lower() for x in posibles_columnas_precios)]
    
    if columnas_precios:
        print("\nColumnas que podrían contener precios:")
        for col in columnas_precios:
            print(f"- {col} (tipo: {df[col].dtype})")
    
    # Verificar columnas de fechas
    posibles_columnas_fechas = ['fecha', 'date', 'mes', 'año', 'year', 'month']
    columnas_fechas = [col for col in df.columns if any(x in col.lower() for x in posibles_columnas_fechas)]
    
    if columnas_fechas:
        print("\nColumnas que podrían contener fechas:")
        for col in columnas_fechas:
            print(f"- {col} (valores únicos: {df[col].nunique()})")
    
    # Mostrar categorías o grupos de alimentos si existen
    posibles_columnas_grupos = ['grupo', 'categoria', 'tipo', 'clase']
    columnas_grupos = [col for col in df.columns if any(x in col.lower() for x in posibles_columnas_grupos)]
    
    if columnas_grupos:
        print("\nPosibles grupos o categorías de alimentos:")
        for col in columnas_grupos:
            print(f"\nValores únicos en '{col}':")
            print(df[col].value_counts().head(10))
            
except Exception as e:
    print(f"\nError al leer el archivo: {e}")

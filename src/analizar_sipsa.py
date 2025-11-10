import pandas as pd
from pathlib import Path
import re

def limpiar_nombres_columnas(df):
    """Limpia los nombres de las columnas del DataFrame"""
    df.columns = [str(col).strip() for col in df.columns]
    df.columns = [re.sub(r'\s+', ' ', col).strip() for col in df.columns]
    return df

def extraer_departamento(mercado):
    """Extrae el departamento del nombre del mercado"""
    if not isinstance(mercado, str):
        return None
    
    # Lista de departamentos de Colombia
    departamentos = [
        'AMAZONAS', 'ANTIOQUIA', 'ARAUCA', 'ATLÁNTICO', 'BOGOTÁ', 'BOLÍVAR',
        'BOYACÁ', 'CALDAS', 'CAQUETÁ', 'CASANARE', 'CAUCA', 'CESAR', 'CHOCÓ',
        'CÓRDOBA', 'CUNDINAMARCA', 'GUAINÍA', 'GUAVIARE', 'HUILA', 'LA GUAJIRA',
        'MAGDALENA', 'META', 'NARIÑO', 'NORTE DE SANTANDER', 'PUTUMAYO',
        'QUINDÍO', 'RISARALDA', 'SAN ANDRÉS', 'SANTANDER', 'SUCRE', 'TOLIMA',
        'VALLE DEL CAUCA', 'VAUPÉS', 'VICHADA'
    ]
    
    mercado = mercado.upper()
    
    # Buscar coincidencias exactas primero
    for depto in departamentos:
        if depto in mercado:
            return depto
    
    # Si no hay coincidencia exacta, buscar coincidencias parciales
    for depto in departamentos:
        # Eliminar acentos y signos de puntuación para una mejor coincidencia
        depto_limpio = depto.replace('Á', 'A').replace('É', 'E').replace('Í', 'I')\
                        .replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N')
        mercado_limpio = mercado.replace('Á', 'A').replace('É', 'E').replace('Í', 'I')\
                          .replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N')
        
        if depto_limpio in mercado_limpio:
            return depto
    
    # Si no se encuentra coincidencia, devolver el nombre del mercado
    return mercado.strip()

def leer_archivo_sipsa(archivo):
    """Lee un archivo de SIPSA y devuelve un DataFrame limpio"""
    try:
        # Leer el archivo Excel sin asumir encabezados
        df = pd.read_excel(archivo, header=None)
        
        # Encontrar la fila que contiene los encabezados
        header_row = None
        for i in range(min(20, len(df))):  # Buscar en las primeras 20 filas
            row_vals = df.iloc[i].dropna().astype(str).str.lower().values
            if 'fecha' in ' '.join(row_vals) and 'producto' in ' '.join(row_vals) and 'precio' in ' '.join(row_vals):
                header_row = i
                break
        
        if header_row is None:
            print("No se pudo encontrar la fila de encabezados")
            return None
        
        # Leer el archivo con los encabezados correctos
        df = pd.read_excel(archivo, header=header_row)
        
        # Limpiar nombres de columnas
        df.columns = [str(col).strip() for col in df.columns]
        
        # Mapear nombres de columnas a un formato estándar
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'fecha' in col_lower:
                column_mapping[col] = 'fecha'
            elif 'producto' in col_lower:
                column_mapping[col] = 'producto'
            elif 'mercado' in col_lower or 'ciudad' in col_lower:
                column_mapping[col] = 'mercado'
            elif 'precio' in col_lower and 'promedio' in col_lower:
                column_mapping[col] = 'precio_kg'
            elif 'grupo' in col_lower:
                column_mapping[col] = 'grupo'
        
        # Renombrar columnas
        df = df.rename(columns=column_mapping)
        
        # Mantener solo las columnas necesarias
        columnas_necesarias = ['fecha', 'grupo', 'producto', 'mercado', 'precio_kg']
        df = df[[col for col in columnas_necesarias if col in df.columns]].copy()
        
        # Limpiar los datos
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        
        if 'precio_kg' in df.columns:
            if df['precio_kg'].dtype == 'object':
                df['precio_kg'] = df['precio_kg'].astype(str).str.replace('[^\d.,]', '', regex=True)
                df['precio_kg'] = pd.to_numeric(df['precio_kg'].str.replace(',', '.'), errors='coerce')
        
        # Filtrar filas con datos faltantes
        df = df.dropna(subset=['mercado', 'precio_kg'])
        
        return df
    
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        import traceback
        traceback.print_exc()
        return None

def analizar_archivo_sipsa(archivo):
    """Analiza un archivo de SIPSA y devuelve estadísticas por departamento"""
    print(f"\nAnalizando archivo: {archivo.name}")
    
    try:
        # Leer el archivo
        df = leer_archivo_sipsa(archivo)
        
        if df is None or df.empty:
            print("No se pudo leer el archivo o está vacío")
            return None
        
        # Mostrar información básica
        print(f"\nTotal de registros: {len(df):,}")
        print("\nPrimeras filas de datos:")
        print(df.head())
        
        # Extraer departamento del nombre del mercado
        df['departamento'] = df['mercado'].apply(extraer_departamento)
        
        # Análisis por departamento
        print(f"\nDepartamentos únicos encontrados: {df['departamento'].nunique()}")
        
        # Conteo por departamento
        conteo_deptos = df['departamento'].value_counts().reset_index()
        conteo_deptos.columns = ['Departamento', 'Registros']
        conteo_deptos['Porcentaje'] = (conteo_deptos['Registros'] / len(df) * 100).round(2)
        
        print("\nDistribución por departamento:")
        print(conteo_deptos.to_string(index=False))
        
        # Estadísticas de precios por departamento
        if 'precio_kg' in df.columns:
            print("\nEstadísticas de precios por departamento:")
            stats = df.groupby('departamento')['precio_kg'].agg(['count', 'min', 'mean', 'max', 'std']).round(2)
            stats = stats.rename(columns={'count': 'Registros', 'mean': 'Precio Promedio', 
                                        'min': 'Mínimo', 'max': 'Máximo', 'std': 'Desv. Estándar'})
            print(stats.sort_values('Registros', ascending=False).to_string())
        
        return df
    
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return None

def analizar_todos_archivos_sipsa():
    """Analiza todos los archivos SIPSA en el directorio de datos"""
    directorio = Path('data/original')
    archivos_sipsa = list(directorio.glob('*SIPSA*.xlsx')) + list(directorio.glob('*SIPSA*.xls'))
    
    if not archivos_sipsa:
        print("No se encontraron archivos SIPSA para analizar")
        return
    
    print(f"Se encontraron {len(archivos_sipsa)} archivos SIPSA")
    
    resultados = []
    for archivo in sorted(archivos_sipsa, key=lambda x: x.name):
        df = analizar_archivo_sipsa(archivo)
        if df is not None and not df.empty:
            resultados.append(df)
    
    # Combinar todos los DataFrames si hay más de uno
    if resultados:
        df_completo = pd.concat(resultados, ignore_index=True)
        print("\n=== ANÁLISIS COMBINADO DE TODOS LOS ARCHIVOS ===")
        print(f"Total de registros combinados: {len(df_completo):,}")
        print(f"Total de departamentos únicos: {df_completo['departamento'].nunique()}")
        
        # Guardar resultados
        output_dir = Path('data/procesado')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'sipsa_departamentos_analisis.csv'
        df_completo.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResultados guardados en: {output_file}")
        
        # Mostrar todos los departamentos encontrados
        print("\nTodos los departamentos encontrados:")
        deptos_unicos = sorted(df_completo['departamento'].unique())
        for i, depto in enumerate(deptos_unicos, 1):
            print(f"{i:2d}. {depto}")

if __name__ == "__main__":
    print("=== ANÁLISIS DE DATOS SIPSA POR DEPARTAMENTO ===")
    analizar_todos_archivos_sipsa()

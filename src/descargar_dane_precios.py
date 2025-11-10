import pandas as pd
import requests
from pathlib import Path
import json
from datetime import datetime

def descargar_datos_dane():
    """
    Descarga datos de precios del DANE
    """
    # URL del servicio de datos abiertos del DANE (IPC por ciudades)
    url = "https://www.datos.gov.co/resource/ceyp-9c7c.json?$limit=50000"
    
    try:
        print("Descargando datos del DANE...")
        response = requests.get(url)
        response.raise_for_status()
        
        datos = response.json()
        df = pd.DataFrame(datos)
        
        # Guardar datos crudos
        output_dir = Path('data/original/dane')
        output_dir.mkdir(parents=True, exist_ok=True)
        fecha_actual = datetime.now().strftime('%Y%m%d')
        output_file = output_dir / f'precios_dane_{fecha_actual}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Datos guardados en: {output_file}")
        return df
    except Exception as e:
        print(f"Error al descargar datos del DANE: {e}")
        return None

def verificar_departamentos(df):
    """
    Verifica la cobertura de departamentos en los datos
    """
    # Lista de los 32 departamentos de Colombia + Bogotá D.C.
    departamentos_colombia = [
        'Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá D.C.', 'Bolívar',
        'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
        'Córdoba', 'Cundinamarca', 'Guainía', 'Guaviare', 'Huila', 'La Guajira',
        'Magdalena', 'Meta', 'Nariño', 'Norte de Santander', 'Putumayo',
        'Quindío', 'Risaralda', 'San Andrés y Providencia', 'Santander', 
        'Sucre', 'Tolima', 'Valle del Cauca', 'Vaupés', 'Vichada'
    ]
    
    # Columnas que podrían contener departamentos
    posibles_columnas = ['dpto', 'departamento', 'nombre_departamento', 'ciudad', 'municipio']
    
    print("\n=== Análisis de cobertura por departamento ===")
    
    # Buscar columnas que contengan información de departamentos
    columnas_encontradas = []
    for col in df.columns:
        if any(x in col.lower() for x in ['dpto', 'departamento', 'ciudad', 'municipio']):
            columnas_encontradas.append(col)
    
    if not columnas_encontradas:
        print("No se encontraron columnas con información de departamentos")
        return
    
    print(f"\nColumnas con información geográfica encontradas: {', '.join(columnas_encontradas)}")
    
    # Analizar cada columna encontrada
    for col in columnas_encontradas:
        print(f"\nAnálisis de la columna: {col}")
        print("-" * 50)
        
        # Contar valores únicos
        valores_unicos = df[col].nunique()
        print(f"Valores únicos: {valores_unicos}")
        
        # Mostrar los valores más comunes
        print("\nValores más comunes:")
        print(df[col].value_counts().head(10))
        
        # Verificar cobertura de departamentos
        if valores_unicos > 10:  # Solo si hay suficientes valores únicos
            print("\nBuscando coincidencias con departamentos...")
            departamentos_encontrados = []
            
            for depto in departamentos_colombia:
                if df[col].str.contains(depto, case=False, na=False).any():
                    departamentos_encontrados.append(depto)
            
            print(f"\nDepartamentos encontrados: {len(departamentos_encontrados)}/32")
            if departamentos_encontrados:
                print("\nLista de departamentos encontrados:")
                for depto in sorted(departamentos_encontrados):
                    print(f"- {depto}")
                
                # Mostrar departamentos faltantes
                faltantes = set(departamentos_colombia) - set(departamentos_encontrados)
                if faltantes:
                    print("\nDepartamentos faltantes:")
                    for depto in sorted(faltantes):
                        print(f"- {depto}")

def main():
    print("=== Obteniendo datos de precios del DANE ===")
    
    # 1. Descargar datos
    df = descargar_datos_dane()
    
    if df is not None and not df.empty:
        print(f"\nTotal de registros descargados: {len(df)}")
        print("\nPrimeras filas de los datos:")
        print(df.head())
        
        # 2. Verificar cobertura por departamento
        verificar_departamentos(df)
    else:
        print("No se pudieron obtener datos del DANE")
        
        # Cargar datos de ejemplo si la descarga falla
        print("\nCargando datos de ejemplo...")
        try:
            # Verificar si hay datos de ejemplo anteriores
            archivos_ejemplo = list(Path('data/original').rglob('precios_*.csv'))
            if archivos_ejemplo:
                archivo_ejemplo = archivos_ejemplo[0]
                print(f"Cargando datos de ejemplo de: {archivo_ejemplo}")
                df_ejemplo = pd.read_csv(archivo_ejemplo, encoding='latin1')
                verificar_departamentos(df_ejemplo)
            else:
                print("No se encontraron archivos de ejemplo")
        except Exception as e:
            print(f"Error al cargar datos de ejemplo: {e}")

if __name__ == "__main__":
    main()

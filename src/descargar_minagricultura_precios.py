import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

def descargar_datos_minagricultura():
    """
    Descarga datos de precios del Ministerio de Agricultura
    """
    # URL del portal de datos abiertos del MinAgricultura
    url = "https://www.datos.gov.co/resource/2pnw-mmge.json"  # Precios de productos agropecuarios
    
    try:
        print("Descargando datos del Ministerio de Agricultura...")
        response = requests.get(url, params={'$limit': 50000})
        response.raise_for_status()
        
        datos = response.json()
        df = pd.DataFrame(datos)
        
        # Guardar datos crudos
        output_dir = Path('data/original/minagricultura')
        output_dir.mkdir(parents=True, exist_ok=True)
        fecha_actual = datetime.now().strftime('%Y%m%d')
        output_file = output_dir / f'precios_agro_{fecha_actual}.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Datos guardados en: {output_file}")
        return df
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        return None

def verificar_cobertura_departamental(df):
    """
    Verifica la cobertura de departamentos en los datos
    """
    # Lista de los 32 departamentos de Colombia + Bogotá D.C.
    departamentos_colombia = [
        'Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá', 'Bolívar',
        'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
        'Córdoba', 'Cundinamarca', 'Guainía', 'Guaviare', 'Huila', 'La Guajira',
        'Magdalena', 'Meta', 'Nariño', 'Norte de Santander', 'Putumayo',
        'Quindío', 'Risaralda', 'San Andrés', 'Santander', 
        'Sucre', 'Tolima', 'Valle del Cauca', 'Vaupés', 'Vichada'
    ]
    
    print("\n=== Análisis de cobertura por departamento ===")
    
    # Buscar columnas que podrían contener departamentos
    columnas_posibles = [col for col in df.columns if any(x in col.lower() for x in ['dpto', 'departamento', 'ciudad', 'municipio'])]
    
    if not columnas_posibles:
        print("No se encontraron columnas con información de departamentos")
        print("\nColumnas disponibles:", list(df.columns))
        return
    
    print(f"\nAnalizando columnas: {', '.join(columnas_posibles)}")
    
    for col in columnas_posibles:
        print(f"\n--- Análisis de: {col} ---")
        print(f"Valores únicos: {df[col].nunique()}")
        
        # Mostrar los valores más comunes
        print("\nValores más comunes:")
        print(df[col].value_counts().head(10))
        
        # Verificar coincidencias con departamentos
        print("\nBuscando coincidencias con departamentos...")
        coincidencias = []
        
        for depto in departamentos_colombia:
            # Buscar coincidencias parciales (por si el nombre está en una cadena más larga)
            if df[col].astype(str).str.contains(depto, case=False, na=False).any():
                coincidencias.append(depto)
        
        if coincidencias:
            print(f"\nDepartamentos encontrados: {len(coincidencias)}/32")
            for depto in sorted(coincidencias):
                print(f"- {depto}")
            
            # Mostrar departamentos faltantes
            faltantes = set(departamentos_colombia) - set(coincidencias)
            if faltantes:
                print("\nDepartamentos faltantes:")
                for depto in sorted(faltantes):
                    print(f"- {depto}")
            
            return  # Terminar si encontramos departamentos
    
    print("\nNo se encontraron coincidencias con nombres de departamentos")

def main():
    print("=== Obteniendo datos de precios del Ministerio de Agricultura ===")
    
    # 1. Descargar datos
    df = descargar_datos_minagricultura()
    
    if df is not None and not df.empty:
        print(f"\nTotal de registros descargados: {len(df)}")
        print("\nPrimeras filas de los datos:")
        print(df.head())
        
        # 2. Verificar cobertura por departamento
        verificar_cobertura_departamental(df)
    else:
        print("No se pudieron obtener datos. Cargando datos de ejemplo...")
        
        # Cargar datos de ejemplo si la descarga falla
        try:
            # Verificar si hay datos de ejemplo anteriores
            archivos_ejemplo = list(Path('data/original').rglob('precios_*.csv'))
            if archivos_ejemplo:
                archivo_ejemplo = archivos_ejemplo[0]
                print(f"Cargando datos de ejemplo de: {archivo_ejemplo}")
                df_ejemplo = pd.read_csv(archivo_ejemplo, encoding='latin1')
                verificar_cobertura_departamental(df_ejemplo)
            else:
                print("No se encontraron archivos de ejemplo")
        except Exception as e:
            print(f"Error al cargar datos de ejemplo: {e}")

if __name__ == "__main__":
    main()

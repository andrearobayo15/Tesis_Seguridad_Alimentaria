import pandas as pd
import requests
from pathlib import Path
import json
from datetime import datetime

def obtener_datos_upra():
    """
    Descarga datos de precios de la UPRA para todos los departamentos
    """
    # URL de la API de UPRA (ejemplo, puede necesitar actualización)
    url = "https://www.datos.gov.co/resource/ceyp-9c7c.json"
    
    try:
        # Hacer la petición a la API
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error para respuestas no exitosas
        
        # Convertir la respuesta a DataFrame
        datos = response.json()
        df = pd.DataFrame(datos)
        
        # Guardar datos crudos
        output_dir = Path('data/original/upra')
        output_dir.mkdir(parents=True, exist_ok=True)
        fecha_actual = datetime.now().strftime('%Y%m%d')
        df.to_csv(output_dir / f'precios_upra_{fecha_actual}.csv', index=False, encoding='utf-8')
        
        return df
    except Exception as e:
        print(f"Error al descargar datos de UPRA: {e}")
        return None

def procesar_datos_upra(df):
    """
    Procesa los datos de la UPRA para asegurar cobertura de los 32 departamentos
    """
    # Lista de los 32 departamentos de Colombia
    departamentos_colombia = [
        'Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá D.C.', 'Bolívar',
        'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
        'Córdoba', 'Cundinamarca', 'Guainía', 'Guaviare', 'Huila', 'La Guajira',
        'Magdalena', 'Meta', 'Nariño', 'Norte de Santander', 'Putumayo',
        'Quindío', 'Risaralda', 'San Andrés y Providencia', 'Santander', 
        'Sucre', 'Tolima', 'Valle del Cauca', 'Vaupés', 'Vichada'
    ]
    
    # Verificar cobertura de departamentos
    if 'departamento' in df.columns:
        departamentos_presentes = df['departamento'].unique()
        print(f"Departamentos en los datos: {len(departamentos_presentes)}/32")
        
        # Identificar departamentos faltantes
        faltantes = set(departamentos_colombia) - set(departamentos_presentes)
        if faltantes:
            print(f"Departamentos faltantes: {', '.join(faltantes)}")
    
    # Procesamiento estándar (ajustar según estructura real de los datos)
    columnas_requeridas = ['producto', 'departamento', 'precio', 'fecha', 'unidad_medida']
    
    # Renombrar columnas si es necesario
    mapeo_columnas = {
        'nom_producto': 'producto',
        'departamento_origen': 'departamento',
        'precio_promedio': 'precio',
        'fecha_corte': 'fecha',
        'unidad': 'unidad_medida'
    }
    
    df = df.rename(columns=mapeo_columnas)
    
    # Convertir tipos de datos
    if 'precio' in df.columns:
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    
    return df

def generar_estructura_completa():
    """
    Genera una estructura de datos con todos los departamentos
    para asegurar cobertura completa
    """
    # Lista de departamentos de Colombia
    departamentos_colombia = [
        'Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá D.C.', 'Bolívar',
        'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
        'Córdoba', 'Cundinamarca', 'Guainía', 'Guaviare', 'Huila', 'La Guajira',
        'Magdalena', 'Meta', 'Nariño', 'Norte de Santander', 'Putumayo',
        'Quindío', 'Risaralda', 'San Andrés y Providencia', 'Santander', 
        'Sucre', 'Tolima', 'Valle del Cauca', 'Vaupés', 'Vichada'
    ]
    # Crear datos de ejemplo (reemplazar con datos reales)
    fechas = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
    productos = ['Arroz', 'Fríjol', 'Papa', 'Plátano', 'Pollo']
    
    datos = []
    for fecha in fechas:
        for depto in departamentos_colombia:
            for producto in productos:
                datos.append({
                    'fecha': fecha,
                    'departamento': depto,
                    'producto': producto,
                    'precio': None,  # Valor por defecto
                    'unidad_medida': 'kg',
                    'fuente': 'SIN DATOS'
                })
    
    return pd.DataFrame(datos)

def main():
    print("=== Descargando datos de precios por departamento ===")
    
    # 1. Intentar descargar datos de UPRA
    print("\n1. Descargando datos de UPRA...")
    df_upra = obtener_datos_upra()
    
    if df_upra is not None and not df_upra.empty:
        print("Datos descargados exitosamente de UPRA")
        print(f"Registros: {len(df_upra)}")
        print("\nPrimeras filas:")
        print(df_upra.head())
        
        # 2. Procesar datos
        print("\n2. Procesando datos...")
        df_procesado = procesar_datos_upra(df_upra)
        
        # 3. Guardar datos procesados
        output_dir = Path('data/procesado')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'precios_departamentos_completo.csv'
        df_procesado.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nDatos procesados guardados en: {output_file}")
        
        # 4. Generar reporte de cobertura
        if 'departamento' in df_procesado.columns:
            cobertura = df_procesado['departamento'].value_counts().reset_index()
            cobertura.columns = ['departamento', 'registros']
            
            reporte_file = output_dir / 'reporte_cobertura_departamentos.csv'
            cobertura.to_csv(reporte_file, index=False, encoding='utf-8')
            print(f"Reporte de cobertura guardado en: {reporte_file}")
    else:
        print("No se pudieron obtener datos de UPRA. Usando datos de ejemplo.")
        # Crear estructura de datos de ejemplo
        df_ejemplo = generar_estructura_completa()
        
        output_file = output_dir / 'precios_departamentos_ejemplo.csv'
        df_ejemplo.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nEstructura de datos de ejemplo guardada en: {output_file}")
        print("\nNota: Los precios son nulos. Deberás completar con datos reales.")

if __name__ == "__main__":
    main()

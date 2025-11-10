import pandas as pd
import os
from pathlib import Path

def extraer_departamento(mercado):
    """Extrae el departamento del nombre del mercado"""
    # Lista de departamentos de Colombia
    departamentos = [
        'Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá', 'Bolívar',
        'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
        'Córdoba', 'Cundinamarca', 'Guainía', 'Guaviare', 'Huila', 'La Guajira',
        'Magdalena', 'Meta', 'Nariño', 'Norte de Santander', 'Putumayo',
        'Quindío', 'Risaralda', 'San Andrés', 'Santander', 'Sucre', 'Tolima',
        'Valle del Cauca', 'Vaupés', 'Vichada'
    ]
    
    # Buscar departamento en el nombre del mercado
    for depto in departamentos:
        if depto.lower() in mercado.lower():
            return depto
    
    # Si no se encuentra, devolver 'Desconocido' o el nombre del mercado
    return 'Desconocido'

def procesar_archivo_precios(ruta_archivo):
    """
    Procesa el archivo de precios de alimentos y devuelve un DataFrame estructurado
    """
    # Leer el archivo con la codificación correcta
    with open(ruta_archivo, 'r', encoding='ISO-8859-1') as f:
        lineas = f.readlines()
    
    # Procesar cada línea y dividir por punto y coma
    datos = []
    for i, linea in enumerate(lineas):
        # Saltar líneas vacías
        if not linea.strip():
            continue
            
        # Dividir la línea por punto y coma
        partes = [p.strip() for p in linea.split(';')]
        
        # Tomar los primeros 5 elementos (fecha, grupo, producto, mercado, precio)
        # y descartar el resto
        if len(partes) >= 5:
            fecha = partes[0]
            grupo = partes[1] if len(partes) > 1 else ''
            producto = partes[2] if len(partes) > 2 else ''
            mercado = partes[3] if len(partes) > 3 else ''
            # Extraer departamento
            departamento = extraer_departamento(mercado)
            
            # El precio puede tener el símbolo de moneda y otros caracteres, los limpiamos
            precio_str = partes[4].replace('$', '').replace(',', '.').strip()
            
            try:
                precio = float(precio_str) if precio_str else None
            except ValueError:
                precio = None
                
            datos.append({
                'fecha': fecha,
                'grupo': grupo,
                'producto': producto,
                'mercado': mercado,
                'departamento': departamento,
                'precio_kg': precio
            })
    
    # Crear DataFrame con los datos procesados
    df = pd.DataFrame(datos)
    
    # Convertir la columna de fecha a datetime
    # Primero, mapear los nombres de los meses en español a números
    meses_es = {
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }
    
    def parse_fecha(fecha_str):
        try:
            if '-' in fecha_str:
                mes, anio = fecha_str.split('-')
                mes = meses_es.get(mes.lower()[:3], '01')
                return pd.to_datetime(f"{anio}-{mes}-01")
        except:
            pass
        return None
    
    df['fecha_dt'] = df['fecha'].apply(parse_fecha)
    
    return df

def analizar_datos(df):
    """
    Realiza un análisis exploratorio de los datos de precios
    """
    print("\n" + "="*50)
    print("ANÁLISIS DE DATOS DE PRECIOS")
    print("="*50)
    
    # Información general
    print(f"\nTotal de registros: {len(df):,}")
    print(f"Período: {df['fecha_dt'].min().strftime('%Y-%m')} a {df['fecha_dt'].max().strftime('%Y-%m')}")
    
    # Grupos de alimentos únicos
    grupos = df['grupo'].unique()
    print(f"\nGrupos de alimentos ({len(grupos)}):")
    for grupo in sorted(grupos):
        print(f"- {grupo}")
    
    # Productos por grupo
    print("\nNúmero de productos por grupo:")
    print(df.groupby('grupo')['producto'].nunique().sort_values(ascending=False))
    
    # Departamentos únicos
    print("\nDepartamentos con datos:")
    print(df['departamento'].value_counts())
    
    # Mercados únicos por departamento
    print("\nMercados únicos por departamento:")
    print(df.groupby('departamento')['mercado'].nunique().sort_values(ascending=False))
    
    # Estadísticas de precios
    print("\nEstadísticas generales de precios por kg:")
    print(df['precio_kg'].describe())
    
    # Estadísticas de precios por departamento
    if 'departamento' in df.columns and 'precio_kg' in df.columns:
        print("\nPrecio promedio por kg por departamento:")
        print(df.groupby('departamento')['precio_kg'].agg(['mean', 'std', 'count']).round(2).sort_values('mean', ascending=False))
    
    return df

def calcular_promedios_por_departamento(df, grupos_seleccionados=None, productos_seleccionados=None, top_n=3):
    """
    Calcula promedios mensuales de precios por departamento para los grupos y productos seleccionados
    """
    if grupos_seleccionados is None:
        # Si no se especifican grupos, usar todos
        grupos_seleccionados = df['grupo'].unique()
    
    # Filtrar por grupos seleccionados
    df_filtrado = df[df['grupo'].isin(grupos_seleccionados)]
    
    # Si no se especifican productos, seleccionar los más comunes
    if productos_seleccionados is None:
        productos_seleccionados = []
        for grupo in grupos_seleccionados:
            productos_grupo = df_filtrado[df_filtrado['grupo'] == grupo]
            top_productos = productos_grupo['producto'].value_counts().head(top_n).index.tolist()
            productos_seleccionados.extend([(grupo, p) for p in top_productos])
    
    # Filtrar solo los productos seleccionados
    df_top = pd.DataFrame()
    for grupo, producto in productos_seleccionados:
        df_temp = df_filtrado[(df_filtrado['grupo'] == grupo) & 
                             (df_filtrado['producto'] == producto)]
        df_top = pd.concat([df_top, df_temp])
    
    # Calcular promedios mensuales por departamento, grupo y producto
    if not df_top.empty:
        promedios = df_top.groupby(['departamento', 'grupo', 'producto', 'fecha_dt'])['precio_kg'].agg(
            ['mean', 'count', 'std']
        ).reset_index()
        promedios = promedios.rename(columns={
            'mean': 'precio_promedio_kg',
            'count': 'num_registros',
            'std': 'desviacion_estandar'
        })
    else:
        promedios = pd.DataFrame()
    
    return promedios

def main():
    # Rutas de los archivos
    ruta_archivo = Path(r'data\\original\\BaseDatos-SIPSA_P-Mensual-2022\\mensual 22.csv')
    
    # Procesar el archivo
    print(f"Procesando archivo: {ruta_archivo}")
    df = procesar_archivo_precios(ruta_archivo)
    
    # Analizar los datos
    df_analizado = analizar_datos(df)
    
    # Guardar resultados
    output_dir = Path('data/procesado')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos completos
    output_file = output_dir / 'precios_alimentos_2022_completo.csv'
    df_analizado.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nDatos completos guardados en: {output_file}")
    
    # Ejemplo: Calcular promedios por departamento para algunos grupos
    grupos_ejemplo = ['FRUTAS', 'VERDURAS Y HORTALIZAS', 'CARNES']
    
    # Seleccionar algunos productos representativos de cada grupo
    productos_ejemplo = [
        ('FRUTAS', 'Banano'),
        ('FRUTAS', 'Naranja'),
        ('FRUTAS', 'Mango'),
        ('VERDURAS Y HORTALIZAS', 'Tomate'),
        ('VERDURAS Y HORTALIZAS', 'Cebolla cabezona'),
        ('VERDURAS Y HORTALIZAS', 'Zanahoria'),
        ('CARNES', 'Carne de res'),
        ('CARNES', 'Pechuga de pollo'),
        ('CARNES', 'Carne de cerdo')
    ]
    
    # Calcular promedios por departamento
    promedios_deptos = calcular_promedios_por_departamento(
        df_analizado, 
        grupos_seleccionados=grupos_ejemplo,
        productos_seleccionados=productos_ejemplo
    )
    
    if not promedios_deptos.empty:
        # Guardar promedios por departamento
        output_promedios_deptos = output_dir / 'promedios_mensuales_por_departamento_2022.csv'
        promedios_deptos.to_csv(output_promedios_deptos, index=False, encoding='utf-8')
        print(f"Promedios mensuales por departamento guardados en: {output_promedios_deptos}")
        
        # Mostrar resumen de precios por departamento
        print("\nResumen de precios promedio por departamento (último mes):")
        ultimo_mes = promedios_deptos['fecha_dt'].max()
        resumen_ultimo_mes = promedios_deptos[promedios_deptos['fecha_dt'] == ultimo_mes]
        
        # Pivotar para mejor visualización
        resumen_pivot = resumen_ultimo_mes.pivot_table(
            index=['departamento'],
            columns=['grupo', 'producto'],
            values='precio_promedio_kg',
            aggfunc='mean'
        ).round(2)
        
        print("\nPrecios promedio por departamento (último mes):")
        print(resumen_pivot.to_string())
        
        # Guardar resumen
        resumen_pivot.to_csv(output_dir / 'resumen_precios_ultimo_mes.csv', encoding='utf-8')
        
        # Análisis de variabilidad de precios
        print("\nVariabilidad de precios por departamento (desviación estándar):")
        variabilidad = promedios_deptos.groupby(['departamento', 'grupo', 'producto'])['precio_promedio_kg'].std().reset_index()
        print(variabilidad.sort_values('precio_promedio_kg', ascending=False).head(10).to_string())
    else:
        print("No se pudieron calcular los promedios por departamento.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analizar y visualizar datos de NDVI y temperatura (ERA5_LST).
Genera histogramas, diagramas de cajas y gráficos de evolución temporal.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, YearLocator
import matplotlib.ticker as ticker
from datetime import datetime
import calendar

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos(archivo):
    """
    Carga los datos desde un archivo CSV y prepara las fechas.
    
    Args:
        archivo: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados y fechas procesadas
    """
    print(f"Cargando datos desde: {archivo}")
    
    try:
        # Cargar el archivo
        df = pd.read_csv(archivo, encoding='utf-8')
        
        # Convertir fecha a datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Extraer año y mes
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['mes_nombre'] = df['fecha'].dt.month_name()
        
        # Crear una fecha continua para gráficos de evolución temporal
        df['fecha_continua'] = df['fecha']
        
        print(f"Datos cargados. Dimensiones: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def crear_histogramas_general(df, variable, nombre_variable, ruta_salida):
    """
    Crea histogramas generales para la variable especificada.
    
    Args:
        df: DataFrame con los datos
        variable: Nombre de la columna con los valores a graficar
        nombre_variable: Nombre para mostrar en los títulos
        ruta_salida: Ruta donde guardar los gráficos
    """
    print(f"Creando histogramas generales para {nombre_variable}...")
    
    # Crear carpeta si no existe
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Histograma general
    plt.figure(figsize=(14, 9))
    
    # Calcular límites para el eje x para evitar que se vea apretado
    valor_min = df[variable].min()
    valor_max = df[variable].max()
    margen = (valor_max - valor_min) * 0.1  # 10% de margen adicional
    
    # Crear el histograma con más bins para mejor visualización
    sns.histplot(df[variable], kde=True, bins=35)
    
    # Establecer límites del eje x con margen adicional
    plt.xlim(valor_min - margen, valor_max + margen)
    
    # Ajustar el número de ticks en el eje x
    plt.locator_params(axis='x', nbins=12)
    
    plt.title(f'Distribución de {nombre_variable} (2022-2025)', fontsize=16)
    plt.xlabel(nombre_variable, fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'histograma_{variable}_general.png'), dpi=300)
    plt.close()
    
    # Histograma por año
    plt.figure(figsize=(16, 10))
    for año in sorted(df['año'].unique()):
        sns.histplot(df[df['año'] == año][variable], kde=True, bins=20, alpha=0.7, label=f'Año {año}')
    
    plt.title(f'Distribución de {nombre_variable} por Año', fontsize=16)
    plt.xlabel(nombre_variable, fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'histograma_{variable}_por_año.png'), dpi=300)
    plt.close()
    
    # Histograma por mes (todos los años combinados)
    plt.figure(figsize=(16, 10))
    
    # Ordenar meses por número
    meses_orden = sorted(df['mes'].unique())
    meses_nombres = [calendar.month_name[mes] for mes in meses_orden]
    
    for i, mes in enumerate(meses_orden):
        sns.histplot(df[df['mes'] == mes][variable], kde=True, bins=20, alpha=0.7, label=meses_nombres[i])
    
    plt.title(f'Distribución de {nombre_variable} por Mes', fontsize=16)
    plt.xlabel(nombre_variable, fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.legend(fontsize=12, ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'histograma_{variable}_por_mes.png'), dpi=300)
    plt.close()

def crear_boxplots_general(df, variable, nombre_variable, ruta_salida):
    """
    Crea diagramas de cajas generales para la variable especificada.
    
    Args:
        df: DataFrame con los datos
        variable: Nombre de la columna con los valores a graficar
        nombre_variable: Nombre para mostrar en los títulos
        ruta_salida: Ruta donde guardar los gráficos
    """
    print(f"Creando diagramas de cajas generales para {nombre_variable}...")
    
    # Crear carpeta si no existe
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Boxplot por año
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='año', y=variable, data=df)
    plt.title(f'Distribución de {nombre_variable} por Año', fontsize=16)
    plt.xlabel('Año', fontsize=14)
    plt.ylabel(nombre_variable, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'boxplot_{variable}_por_año.png'), dpi=300)
    plt.close()
    
    # Boxplot por mes
    plt.figure(figsize=(14, 8))
    
    # Crear una columna para ordenar los meses
    month_order = [calendar.month_name[i] for i in range(1, 13)]
    
    # Filtrar solo los meses presentes en los datos
    available_months = sorted(df['mes'].unique())
    month_order = [calendar.month_name[i] for i in available_months]
    
    # Crear el boxplot
    sns.boxplot(x='mes_nombre', y=variable, data=df, order=month_order)
    plt.title(f'Distribución de {nombre_variable} por Mes', fontsize=16)
    plt.xlabel('Mes', fontsize=14)
    plt.ylabel(nombre_variable, fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'boxplot_{variable}_por_mes.png'), dpi=300)
    plt.close()
    
    # Boxplot por departamento
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='departamento', y=variable, data=df)
    plt.title(f'Distribución de {nombre_variable} por Departamento', fontsize=16)
    plt.xlabel('Departamento', fontsize=14)
    plt.ylabel(nombre_variable, fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'boxplot_{variable}_por_departamento.png'), dpi=300)
    plt.close()

def grafico_evolucion_temporal_general(df, variable, nombre_variable, ruta_salida):
    """
    Crea gráficos de evolución temporal para la variable especificada.
    
    Args:
        df: DataFrame con los datos
        variable: Nombre de la columna con los valores a graficar
        nombre_variable: Nombre para mostrar en los títulos
        ruta_salida: Ruta donde guardar los gráficos
    """
    print(f"Creando gráficos de evolución temporal para {nombre_variable}...")
    
    # Crear carpeta si no existe
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Calcular promedios mensuales
    df_mensual = df.groupby(['fecha_continua', 'año', 'mes'])[variable].mean().reset_index()
    df_mensual = df_mensual.sort_values(['año', 'mes'])
    
    # Gráfico de evolución temporal
    plt.figure(figsize=(16, 8))
    
    # Línea principal
    sns.lineplot(
        data=df_mensual,
        x='fecha_continua',
        y=variable,
        marker='o',
        markersize=5,
        linewidth=1.5,
        color='royalblue'
    )
    
    # Línea de tendencia
    sns.regplot(
        x=np.array(range(len(df_mensual))),
        y=df_mensual[variable],
        scatter=False,
        color='red',
        line_kws={'linestyle': '--', 'linewidth': 1.5}
    )
    
    # Calcular y graficar media móvil (3 meses)
    df_mensual['media_movil'] = df_mensual[variable].rolling(window=3).mean()
    plt.plot(df_mensual['fecha_continua'], df_mensual['media_movil'], 
             color='green', linestyle='-', linewidth=2, alpha=0.7,
             label='Media Móvil (3 meses)')
    
    # Configurar el gráfico
    plt.title(f'Evolución Temporal de {nombre_variable} Promedio (2022-2025)', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel(nombre_variable, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Configurar el eje X para mostrar años
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # Añadir leyenda
    plt.legend(['Valor Mensual', 'Tendencia', 'Media Móvil (3 meses)'], fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'evolucion_{variable}_general.png'), dpi=300)
    plt.close()
    
    # Gráfico de evolución temporal por año
    años = sorted(df['año'].unique())
    
    plt.figure(figsize=(16, 10))
    
    for año in años:
        # Filtrar datos por año
        df_año = df_mensual[df_mensual['año'] == año]
        
        # Graficar línea para cada año
        plt.plot(df_año['mes'], df_año[variable], marker='o', linewidth=2, label=f'Año {año}')
    
    plt.title(f'Evolución Mensual de {nombre_variable} por Año', fontsize=16)
    plt.xlabel('Mes', fontsize=14)
    plt.ylabel(nombre_variable, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Configurar el eje X para mostrar meses
    plt.xticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_salida, f'evolucion_{variable}_por_año.png'), dpi=300)
    plt.close()

def graficos_por_departamento(df, variable, nombre_variable, ruta_base):
    """
    Crea gráficos por departamento para la variable especificada.
    
    Args:
        df: DataFrame con los datos
        variable: Nombre de la columna con los valores a graficar
        nombre_variable: Nombre para mostrar en los títulos
        ruta_base: Ruta base donde guardar los gráficos
    """
    print(f"Creando gráficos por departamento para {nombre_variable}...")
    
    # Obtener lista de departamentos
    departamentos = sorted(df['departamento'].unique())
    
    for departamento in departamentos:
        print(f"Procesando departamento: {departamento}")
        
        # Crear carpeta para el departamento
        ruta_depto = os.path.join(ruta_base, departamento.replace(' ', '_'))
        os.makedirs(ruta_depto, exist_ok=True)
        
        # Filtrar datos para el departamento
        df_depto = df[df['departamento'] == departamento].copy()
        
        # Histograma con más espacio y mejor visualización
        plt.figure(figsize=(14, 9))
        
        # Calcular límites para el eje x para evitar que se vea apretado
        valor_min = df_depto[variable].min()
        valor_max = df_depto[variable].max()
        margen = (valor_max - valor_min) * 0.1  # 10% de margen adicional
        
        # Crear el histograma con más bins para mejor visualización
        ax = sns.histplot(df_depto[variable], kde=True, bins=25)
        
        # Establecer límites del eje x con margen adicional
        plt.xlim(valor_min - margen, valor_max + margen)
        
        # Ajustar el número de ticks en el eje x
        plt.locator_params(axis='x', nbins=10)
        
        plt.title(f'Distribución de {nombre_variable} en {departamento} (2022-2025)', fontsize=16)
        plt.xlabel(nombre_variable, fontsize=14)
        plt.ylabel('Frecuencia', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_depto, f'histograma_{variable}.png'), dpi=300)
        plt.close()
        
        # Boxplot por año
        if len(df_depto['año'].unique()) > 1:
            plt.figure(figsize=(10, 7))
            sns.boxplot(x='año', y=variable, data=df_depto)
            plt.title(f'Distribución de {nombre_variable} por Año en {departamento}', fontsize=16)
            plt.xlabel('Año', fontsize=14)
            plt.ylabel(nombre_variable, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ruta_depto, f'boxplot_{variable}_por_año.png'), dpi=300)
            plt.close()
        
        # Boxplot por mes
        plt.figure(figsize=(12, 7))
        
        # Crear una columna para ordenar los meses
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        
        # Filtrar solo los meses presentes en los datos
        available_months = sorted(df_depto['mes'].unique())
        month_order = [calendar.month_name[i] for i in available_months]
        
        # Crear el boxplot
        sns.boxplot(x='mes_nombre', y=variable, data=df_depto, order=month_order)
        plt.title(f'Distribución de {nombre_variable} por Mes en {departamento}', fontsize=16)
        plt.xlabel('Mes', fontsize=14)
        plt.ylabel(nombre_variable, fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_depto, f'boxplot_{variable}_por_mes.png'), dpi=300)
        plt.close()
        
        # Evolución temporal
        df_mensual = df_depto.groupby(['fecha_continua', 'año', 'mes'])[variable].mean().reset_index()
        df_mensual = df_mensual.sort_values(['año', 'mes'])
        
        plt.figure(figsize=(16, 9))
        
        # Línea principal
        sns.lineplot(
            data=df_mensual,
            x='fecha_continua',
            y=variable,
            marker='o',
            markersize=5,
            linewidth=1.5,
            color='royalblue'
        )
        
        # Línea de tendencia
        if len(df_mensual) > 1:
            sns.regplot(
                x=np.array(range(len(df_mensual))),
                y=df_mensual[variable],
                scatter=False,
                color='red',
                line_kws={'linestyle': '--', 'linewidth': 1.5}
            )
        
        # Calcular y graficar media móvil (3 meses) si hay suficientes datos
        if len(df_mensual) >= 3:
            df_mensual['media_movil'] = df_mensual[variable].rolling(window=3).mean()
            plt.plot(df_mensual['fecha_continua'], df_mensual['media_movil'], 
                    color='green', linestyle='-', linewidth=2, alpha=0.7,
                    label='Media Móvil (3 meses)')
        
        # Calcular límites para el eje y para evitar que se vea apretado
        valor_min = df_mensual[variable].min()
        valor_max = df_mensual[variable].max()
        margen = (valor_max - valor_min) * 0.15  # 15% de margen adicional
        plt.ylim(max(0, valor_min - margen), valor_max + margen)
        
        # Configurar el gráfico
        plt.title(f'Evolución Temporal de {nombre_variable} en {departamento} (2022-2025)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel(nombre_variable, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Configurar el eje X para mostrar fechas de manera más clara
        ax = plt.gca()
        
        # Usar localizadores de fecha más adecuados para el rango 2022-2025
        from matplotlib.dates import MonthLocator, YearLocator
        
        # Localizador principal para años
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        
        # Localizador secundario para trimestres
        ax.xaxis.set_minor_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
        
        # Asegurar que se muestren todas las fechas necesarias
        fecha_min = df_mensual['fecha_continua'].min()
        fecha_max = df_mensual['fecha_continua'].max()
        plt.xlim(fecha_min, fecha_max)
        
        plt.xticks(rotation=45)
        
        # Añadir leyenda
        if len(df_mensual) >= 3:
            plt.legend(['Valor Mensual', 'Tendencia', 'Media Móvil (3 meses)'], fontsize=12)
        else:
            plt.legend(['Valor Mensual', 'Tendencia'], fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_depto, f'evolucion_{variable}.png'), dpi=300)
        plt.close()

def analizar_variable(archivo, variable, nombre_variable, tipo_variable):
    """
    Analiza una variable específica y genera todos los gráficos.
    
    Args:
        archivo: Ruta al archivo CSV con los datos
        variable: Nombre de la columna con los valores a graficar
        nombre_variable: Nombre para mostrar en los títulos
        tipo_variable: Tipo de variable (ndvi, lst, precipitacion)
    """
    print(f"\n=== ANÁLISIS DE {nombre_variable.upper()} ===")
    
    # Cargar datos
    df = cargar_datos(archivo)
    if df is None:
        print(f"No se pudieron cargar los datos de {nombre_variable}.")
        return
    
    # Definir rutas para guardar gráficos
    ruta_general = os.path.join('graficos', 'general', tipo_variable)
    ruta_departamentos = os.path.join('graficos', 'departamentos')
    
    # Crear gráficos generales
    crear_histogramas_general(df, variable, nombre_variable, ruta_general)
    crear_boxplots_general(df, variable, nombre_variable, ruta_general)
    grafico_evolucion_temporal_general(df, variable, nombre_variable, ruta_general)
    
    # Crear gráficos por departamento
    graficos_por_departamento(df, variable, nombre_variable, ruta_departamentos)
    
    print(f"Análisis de {nombre_variable} completado.")

def main():
    """
    Función principal que ejecuta el análisis para todas las variables.
    """
    # Definir rutas de los archivos
    archivo_ndvi = os.path.join('data', 'procesado', 'ndvi_final.csv')
    archivo_lst = os.path.join('data', 'procesado', 'lst_completo.csv')
    archivo_precipitacion = os.path.join('data', 'procesado', 'precipitacion_filtrada.csv')
    
    # Analizar NDVI
    analizar_variable(archivo_ndvi, 'ndvi_medio', 'NDVI', 'ndvi')
    
    # Analizar temperatura (LST)
    analizar_variable(archivo_lst, 'temperatura_media', 'Temperatura', 'lst')
    
    # Analizar precipitación
    analizar_variable(archivo_precipitacion, 'precipitacion_media', 'Precipitación', 'precipitacion')
    
    print("\nAnálisis completo. Los gráficos han sido guardados en las carpetas correspondientes.")

if __name__ == "__main__":
    main()

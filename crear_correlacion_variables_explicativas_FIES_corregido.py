#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creación de gráfico de correlación entre variables explicativas y FIES
Para Figura 1: Correlación entre Variables Explicativas y FIES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

def cargar_datos():
    """Carga los datos de la base master"""
    print("Cargando datos de la base master...")
    
    # Cargar base master
    df = pd.read_csv('d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv')
    
    print(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    
    # Filtrar solo registros con datos FIES
    df_fies = df[df['FIES_moderado_grave'].notna()].copy()
    print(f"Registros con FIES: {len(df_fies)}")
    
    return df_fies

def seleccionar_variables_explicativas(df):
    """Selecciona las principales variables explicativas"""
    
    # Variables explicativas principales
    variables_explicativas = [
        # Variables socioeconómicas
        'IPM_Total',
        'IPC_Total',
        
        # Variables de servicios públicos
        'Energia',
        'Gas_natural', 
        'Acueducto',
        'Alcantarillado',
        'Telefono_fijo',
        'Internet',
        
        # Variables climáticas
        'precipitacion_mm',
        'temperatura_promedio_c',
        'NDVI_promedio'
    ]
    
    # Verificar qué variables están disponibles
    variables_disponibles = []
    for var in variables_explicativas:
        if var in df.columns:
            # Verificar que tenga datos suficientes
            datos_validos = df[var].notna().sum()
            if datos_validos > 100:  # Al menos 100 observaciones
                variables_disponibles.append(var)
                print(f"OK {var}: {datos_validos} observaciones validas")
            else:
                print(f"NO {var}: Solo {datos_validos} observaciones validas")
        else:
            print(f"NO {var}: No encontrada en los datos")
    
    print(f"\nVariables explicativas seleccionadas: {len(variables_disponibles)}")
    return variables_disponibles

def crear_matriz_correlacion_fies(df, variables_explicativas):
    """Crea matriz de correlación entre variables explicativas y FIES"""
    print("Calculando correlaciones con variables FIES...")
    
    # Variables FIES objetivo
    variables_fies = ['FIES_moderado_grave', 'FIES_grave']
    
    # Crear dataset para correlaciones
    variables_analisis = variables_explicativas + variables_fies
    df_corr = df[variables_analisis].copy()
    
    # Calcular matriz de correlación
    matriz_corr = df_corr.corr()
    
    # Extraer solo correlaciones con variables FIES
    corr_fies = matriz_corr[variables_fies].drop(variables_fies, axis=0)
    
    print(f"Matriz de correlacion: {corr_fies.shape}")
    return corr_fies, df_corr

def crear_grafico_correlacion_fies(corr_fies, variables_explicativas):
    """Crea gráfico de correlación entre variables explicativas y FIES"""
    print("Creando grafico de correlacion...")
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Gráfico 1: Heatmap de correlaciones
    sns.heatmap(corr_fies, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.3f',
                cbar_kws={'label': 'Coeficiente de Correlacion'},
                ax=ax1)
    
    ax1.set_title('Correlacion: Variables Explicativas vs FIES\n(Heatmap)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Variables FIES', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variables Explicativas', fontsize=12, fontweight='bold')
    
    # Rotar etiquetas para mejor legibilidad
    ax1.tick_params(axis='y', rotation=0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Barras de correlación con FIES_moderado_grave
    corr_moderado_grave = corr_fies['FIES_moderado_grave'].sort_values(key=abs, ascending=False)
    
    # Colores según signo de correlación
    colores = ['red' if x < 0 else 'blue' for x in corr_moderado_grave.values]
    
    bars = ax2.barh(range(len(corr_moderado_grave)), corr_moderado_grave.values, color=colores, alpha=0.7)
    
    ax2.set_yticks(range(len(corr_moderado_grave)))
    ax2.set_yticklabels(corr_moderado_grave.index, fontsize=10)
    ax2.set_xlabel('Coeficiente de Correlacion', fontsize=12, fontweight='bold')
    ax2.set_title('Correlacion con FIES Moderado-Grave\n(Ordenado por magnitud)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Línea vertical en x=0
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Agregar valores en las barras
    for i, (bar, valor) in enumerate(zip(bars, corr_moderado_grave.values)):
        ax2.text(valor + (0.01 if valor >= 0 else -0.01), i, f'{valor:.3f}', 
                va='center', ha='left' if valor >= 0 else 'right', fontsize=9)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Correlacion Positiva'),
                      Patch(facecolor='red', alpha=0.7, label='Correlacion Negativa')]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Guardar
    plt.savefig('d:/Tesis maestria/Tesis codigo/resultados/01_correlacion_variables_explicativas_FIES.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('d:/Tesis maestria/Tesis codigo/resultados/01_correlacion_variables_explicativas_FIES.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("Grafico guardado: 01_correlacion_variables_explicativas_FIES.png/pdf")
    
    return fig

def crear_grafico_scatter_principales(df, variables_explicativas):
    """Crea gráficos de dispersión para las variables más correlacionadas"""
    print("Creando graficos de dispersion...")
    
    # Calcular correlaciones para seleccionar las más importantes
    corr_moderado = df[variables_explicativas + ['FIES_moderado_grave']].corr()['FIES_moderado_grave'].drop('FIES_moderado_grave')
    top_variables = corr_moderado.abs().sort_values(ascending=False).head(4).index.tolist()
    
    print(f"Variables mas correlacionadas: {top_variables}")
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(top_variables):
        ax = axes[i]
        
        # Filtrar datos válidos
        mask = (df[var].notna()) & (df['FIES_moderado_grave'].notna())
        x_data = df.loc[mask, var]
        y_data = df.loc[mask, 'FIES_moderado_grave']
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=30, color='steelblue')
        
        # Línea de tendencia
        if len(x_data) > 10:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data.sort_values(), p(x_data.sort_values()), "r--", alpha=0.8, linewidth=2)
        
        # Correlación
        corr_val = corr_moderado[var]
        
        ax.set_xlabel(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('FIES Moderado-Grave (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{var.replace("_", " ").title()}\nCorrelacion: {corr_val:.3f}', 
                    fontsize=12, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Estadísticas
        ax.text(0.05, 0.95, f'N = {len(x_data)}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    plt.suptitle('Relacion entre Variables Explicativas Principales y FIES Moderado-Grave', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Guardar
    plt.savefig('d:/Tesis maestria/Tesis codigo/resultados/02_scatter_variables_explicativas_FIES.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('d:/Tesis maestria/Tesis codigo/resultados/02_scatter_variables_explicativas_FIES.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("Grafico guardado: 02_scatter_variables_explicativas_FIES.png/pdf")
    
    return fig

def generar_resumen_correlaciones(corr_fies, df_corr):
    """Genera resumen estadístico de las correlaciones"""
    print("\n=== RESUMEN DE CORRELACIONES ===")
    
    # Correlaciones con FIES_moderado_grave
    print("\nTOP 10 CORRELACIONES CON FIES MODERADO-GRAVE:")
    corr_mod = corr_fies['FIES_moderado_grave'].sort_values(key=abs, ascending=False)
    for i, (var, corr) in enumerate(corr_mod.head(10).items(), 1):
        direccion = "POSITIVA" if corr > 0 else "NEGATIVA"
        print(f"{i:2d}. {var:25s}: {corr:+.3f} {direccion}")
    
    # Correlaciones con FIES_grave
    print("\nTOP 10 CORRELACIONES CON FIES GRAVE:")
    corr_grave = corr_fies['FIES_grave'].sort_values(key=abs, ascending=False)
    for i, (var, corr) in enumerate(corr_grave.head(10).items(), 1):
        direccion = "POSITIVA" if corr > 0 else "NEGATIVA"
        print(f"{i:2d}. {var:25s}: {corr:+.3f} {direccion}")
    
    # Estadísticas generales
    print(f"\nESTADISTICAS GENERALES:")
    print(f"Variables analizadas: {len(corr_fies)}")
    print(f"Observaciones promedio: {df_corr.count().mean():.0f}")
    print(f"Correlacion promedio (abs) con FIES_moderado_grave: {corr_fies['FIES_moderado_grave'].abs().mean():.3f}")
    print(f"Correlacion promedio (abs) con FIES_grave: {corr_fies['FIES_grave'].abs().mean():.3f}")

def main():
    """Función principal"""
    print("=== CREACION DE GRAFICO: CORRELACION VARIABLES EXPLICATIVAS vs FIES ===")
    
    # Cargar datos
    df = cargar_datos()
    
    # Seleccionar variables explicativas
    variables_explicativas = seleccionar_variables_explicativas(df)
    
    if len(variables_explicativas) < 3:
        print("ERROR: No hay suficientes variables explicativas disponibles")
        return
    
    # Crear matriz de correlación
    corr_fies, df_corr = crear_matriz_correlacion_fies(df, variables_explicativas)
    
    # Crear gráficos
    fig1 = crear_grafico_correlacion_fies(corr_fies, variables_explicativas)
    fig2 = crear_grafico_scatter_principales(df, variables_explicativas)
    
    # Generar resumen
    generar_resumen_correlaciones(corr_fies, df_corr)
    
    print("\n=== ARCHIVOS CREADOS ===")
    print("OK 01_correlacion_variables_explicativas_FIES.png/pdf")
    print("OK 02_scatter_variables_explicativas_FIES.png/pdf")
    print("\nESTOS SON LOS ARCHIVOS CORRECTOS PARA:")
    print("- Figura 1: Correlacion entre Variables Explicativas y FIES")
    print("- Figura 2: Relacion entre Variables Principales y FIES")
    
    plt.close('all')

if __name__ == "__main__":
    main()

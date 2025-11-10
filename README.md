# Procesamiento de Datos para Tesis de Maestría

Este repositorio contiene scripts para el procesamiento de datos socioeconómicos y de vivienda de Colombia, utilizados para una tesis de maestría. Los scripts procesan datos de la Encuesta Nacional de Calidad de Vida (ECV) del DANE, transformándolos en series temporales mensuales para análisis estadístico.

## Estructura del Proyecto

```
Tesis codigo/
│
├── data/
│   ├── original/         # Archivos CSV originales exportados de Excel
│   └── procesado/        # Archivos CSV procesados con datos mensuales
│
├── procesar_ecv_*.py     # Scripts para procesar diferentes aspectos de la ECV
└── README.md             # Este archivo
```

## Archivos de Datos Procesados

Los siguientes archivos CSV procesados están disponibles en el directorio `data/procesado/`:

1. **ecv_servicios_2022_2024_mensual.csv**: Datos de acceso a servicios públicos
2. **ecv_pobreza_2022_2024_mensual.csv**: Datos de percepción de pobreza
3. **ecv_deficit_habitacional_total_2022_2024_mensual.csv**: Datos de déficit habitacional
4. **ecv_casa_2022_2024_mensual.csv**: Datos de tenencia de vivienda
5. **ecv_calidadvida_2022_2024_mensual.csv**: Datos de satisfacción con diferentes aspectos de la vida
6. **fies_2023_mensual.csv**: Datos de inseguridad alimentaria FIES para 2023
7. **fies_2024_mensual.csv**: Datos de inseguridad alimentaria FIES para 2024

## Scripts de Procesamiento

Los siguientes scripts están disponibles para procesar los diferentes aspectos de la ECV:

1. **transformar_ecv_servicios.py**: Procesa datos de acceso a servicios públicos
2. **transformar_ecv_pobreza.py**: Procesa datos de percepción de pobreza
3. **transformar_ecv_deficithabitacional.py**: Procesa datos de déficit habitacional
4. **transformar_ecv_casa.py**: Procesa datos de tenencia de vivienda
5. **transformar_ecv_calidadvida.py**: Procesa datos de satisfacción con diferentes aspectos de la vida
6. **procesar_fies_2023.py**: Procesa datos de inseguridad alimentaria FIES para 2023
7. **procesar_fies_2024.py**: Procesa datos de inseguridad alimentaria FIES para 2024

Además, se incluyen scripts de análisis para entender la estructura de los archivos originales:

1. **analizar_estructura_deficithabitacional.py**: Analiza la estructura del archivo de déficit habitacional
2. **analizar_estructura_casa.py**: Analiza la estructura del archivo de tenencia de vivienda
3. **analizar_estructura_calidadvida.py**: Analiza la estructura del archivo de calidad de vida

## Metodología de Procesamiento

Los scripts de procesamiento siguen metodologías similares:

### Para datos ECV:

1. **Lectura de datos**: Se leen los archivos CSV exportados desde Excel, utilizando codificación latin1, utf-8 o ISO-8859-1 según sea necesario.
2. **Análisis de estructura**: Se analizan las primeras filas para identificar la estructura de encabezados múltiples y las columnas relevantes.
3. **Filtrado de datos**: Se filtran las filas para incluir solo los 32 departamentos de Colombia (excluyendo totales nacionales, cabeceras, etc.).
4. **Identificación de columnas**: Se identifican las columnas de porcentaje para cada indicador y año (2022-2024).
5. **Extracción de variables**: Se extraen los valores de las columnas identificadas para cada departamento.
6. **Expansión temporal**: Los datos anuales se replican para generar series mensuales para los años 2022-2024, excluyendo meses futuros en 2024.
7. **Normalización**: Se normalizan los nombres de departamentos y variables para mantener consistencia entre todos los archivos.
8. **Exportación**: Los datos procesados se guardan en archivos CSV con estructura uniforme.

### Para datos FIES:

1. **Lectura de datos**: Se leen los archivos CSV limpios, utilizando codificación latin1 y separador de punto y coma.
2. **Identificación de columnas**: Se identifican las columnas que contienen porcentajes (marcadas con el símbolo "%").
3. **Extracción de valores**: Se extraen los valores porcentuales anuales para cada variable de inseguridad alimentaria.
4. **Conversión de formatos**: Se convierten los valores de string con coma decimal a float.
5. **Expansión temporal**: Los datos anuales se replican para generar series mensuales, manteniendo el mismo valor anual para cada mes (sin dividir por 12).
6. **Normalización**: Se normalizan los nombres de departamentos para mantener consistencia.
7. **Exportación**: Los datos procesados se guardan en archivos CSV con estructura uniforme.

## Estructura de los Datos Procesados

Todos los archivos CSV procesados siguen la misma estructura:

- **departamento**: Nombre normalizado del departamento
- **año**: Año del registro (2022-2024)
- **mes**: Mes del registro (1-12)
- **fecha**: Fecha en formato YYYY-MM-DD (primer día de cada mes)
- **[variables]**: Columnas específicas para cada conjunto de datos, representando porcentajes

## Variables por Conjunto de Datos

### Servicios Públicos (ecv_servicios_2022_2024_mensual.csv)
- **Energia**: Porcentaje de hogares con acceso a energía eléctrica
- **Acueducto**: Porcentaje de hogares con acceso a acueducto
- **Alcantarillado**: Porcentaje de hogares con acceso a alcantarillado
- **Gas_natural**: Porcentaje de hogares con acceso a gas natural
- **Recoleccion_basuras**: Porcentaje de hogares con acceso a recolección de basuras
- **Internet**: Porcentaje de hogares con acceso a internet

### Tenencia de Vivienda (ecv_casa_2022_2024_mensual.csv)
- **Propia_totalmente_pagada**: Porcentaje de hogares con vivienda propia totalmente pagada
- **Propia_la_estan_pagando**: Porcentaje de hogares con vivienda propia que están pagando
- **En_arriendo_o_subarriendo**: Porcentaje de hogares con vivienda en arriendo o subarriendo
- **Con_permiso_sin_pago**: Porcentaje de hogares con vivienda con permiso del propietario sin pago
- **Posesion_sin_titulo**: Porcentaje de hogares con vivienda en posesión sin título
- **Propiedad_colectiva**: Porcentaje de hogares con vivienda en propiedad colectiva

### Déficit Habitacional (ecv_deficit_habitacional_total_2022_2024_mensual.csv)
- **Deficit_cuantitativo**: Porcentaje de hogares en déficit cuantitativo
- **Deficit_cualitativo**: Porcentaje de hogares en déficit cualitativo
- **Deficit_habitacional**: Porcentaje de hogares en déficit habitacional total

### Percepción de Pobreza (ecv_pobreza_2022_2024_mensual.csv)
- **Pobres**: Porcentaje de hogares que se consideran pobres
- **No_pobres**: Porcentaje de hogares que no se consideran pobres

### Satisfacción con Aspectos de la Vida (ecv_calidadvida_2022_2024_mensual.csv)
- **Vida_general**: Promedio de satisfacción con la vida en general (escala 0-10)
- **Salud**: Promedio de satisfacción con la salud (escala 0-10)
- **Seguridad**: Promedio de satisfacción con la seguridad (escala 0-10)
- **Trabajo_actividad**: Promedio de satisfacción con el trabajo o actividad (escala 0-10)
- **Tiempo_libre**: Promedio de satisfacción con el tiempo libre (escala 0-10)
- **Ingreso**: Promedio de satisfacción con el ingreso (escala 0-10)

### Inseguridad Alimentaria (FIES)
- **Preocupacion_alimentos**: Porcentaje de hogares preocupados por no tener suficientes alimentos para comer
- **No_alimentos_saludables**: Porcentaje de hogares que no pudieron comer alimentos saludables y nutritivos
- **Poca_variedad_alimentos**: Porcentaje de hogares que consumieron poca variedad de alimentos
- **Saltar_comida**: Porcentaje de hogares donde al menos un integrante tuvo que saltar una comida
- **Comio_menos**: Porcentaje de hogares donde al menos un integrante comió menos de lo que pensaba que debía comer
- **Sin_alimentos**: Porcentaje de hogares que se quedaron sin alimentos
- **Hambre_no_comio**: Porcentaje de hogares donde al menos un integrante tuvo hambre pero no comió
- **No_comio_dia_entero**: Porcentaje de hogares donde al menos un integrante no comió en un día entero

## Notas Importantes

- Los datos cubren los 32 departamentos de Colombia, incluyendo Bogotá y excluyendo San Andrés.
- Para algunos departamentos y variables, especialmente en 2024, se utilizaron datos de respaldo del año 2021 cuando no estaban disponibles datos más recientes.
- La expansión a datos mensuales asume que los porcentajes se mantienen constantes durante todo el año.
- Los nombres de departamentos han sido normalizados para mantener consistencia entre todos los archivos.

## Fuente de Datos

Los datos originales provienen de las siguientes fuentes del Departamento Administrativo Nacional de Estadística (DANE) de Colombia:

- **ECV**: Encuesta Nacional de Calidad de Vida
- **FIES**: Escala de Experiencia de Inseguridad Alimentaria (Food Insecurity Experience Scale)

## Requisitos

- Python 3.6+
- pandas
- numpy

## Uso

Para procesar un archivo específico, ejecute el script correspondiente:

```bash
python transformar_ecv_servicios.py
python transformar_ecv_pobreza.py
python transformar_ecv_deficithabitacional.py
python transformar_ecv_casa.py
python transformar_ecv_calidadvida.py
python procesar_fies_2023.py
python procesar_fies_2024.py
```

Para analizar la estructura de un archivo original antes de procesarlo:

```bash
python analizar_estructura_deficithabitacional.py
python analizar_estructura_casa.py
python analizar_estructura_calidadvida.py
```

Cada script de transformación leerá el archivo CSV original correspondiente y generará un archivo CSV procesado en el directorio `data/procesado/`.

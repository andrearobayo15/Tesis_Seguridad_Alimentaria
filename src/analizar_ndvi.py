import pandas as pd
from pathlib import Path

# Rutas de los archivos
input_file = Path(r'data\original\NDVI_departamento_frontera_fecha_mod.csv')

# Leer el archivo CSV
print(f"Leyendo archivo: {input_file}")
df = pd.read_csv(input_file)

# Mostrar información básica
print("\nInformación del DataFrame:")
print(df.info())

# Mostrar las primeras filas
print("\nPrimeras filas del DataFrame:")
print(df.head())

# Verificar fechas únicas
print("\nFechas únicas en los datos:")
print(df['fecha'].nunique(), "fechas únicas")
print(df['fecha'].min(), "a", df['fecha'].max())

# Contar registros por departamento
print("\nRegistros por departamento:")
print(df['ADM1_NAME'].value_counts())

# Verificar frecuencia de fechas por departamento
print("\nFechas por departamento:")
for depto in df['ADM1_NAME'].unique():
    depto_dates = df[df['ADM1_NAME'] == depto]['fecha'].nunique()
    print(f"{depto}: {depto_dates} fechas únicas")

# Verificar si hay exactamente 2 registros por mes (cada 15 días)
print("\nVerificación de frecuencia quincenal:")
df['fecha'] = pd.to_datetime(df['fecha'])
df['año_mes'] = df['fecha'].dt.to_period('M')

for depto in df['ADM1_NAME'].unique():
    depto_data = df[df['ADM1_NAME'] == depto]
    counts = depto_data.groupby('año_mes').size()
    print(f"\n{depto} - Registros por mes:")
    print(counts.value_counts().sort_index())
    print(f"Total de meses: {len(counts)}")
    print(f"Promedio de registros por mes: {counts.mean():.2f}")

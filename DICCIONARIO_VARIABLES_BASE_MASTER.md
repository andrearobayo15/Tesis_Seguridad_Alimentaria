# DICCIONARIO DE VARIABLES - BASE MASTER 2022-2025

## Información General

**Archivo:** `BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.xlsx`  
**Período:** 2022-2025  
**Unidad de Análisis:** Departamentos de Colombia (32)  
**Frecuencia:** Mensual  
**Total Registros:** 1,536 (32 departamentos × 48 meses)  
**Total Variables:** 56

---

## 1. VARIABLES DE IDENTIFICACIÓN

### 1.1 Variables Espaciales

| Variable | Descripción | Tipo | Valores |
|----------|-------------|------|---------|
| `departamento` | Nombre del departamento colombiano | Categórica | 32 departamentos únicos |
| `codigo_departamento` | Código DANE del departamento | Numérica | Códigos oficiales DANE |

### 1.2 Variables Temporales

| Variable | Descripción | Tipo | Valores | Formato |
|----------|-------------|------|---------|---------|
| `año` | Año de observación | Numérica | 2022, 2023, 2024, 2025 | YYYY |
| `mes` | Mes de observación | Categórica | enero, febrero, ..., diciembre | Texto español |
| `fecha` | Fecha completa | Fecha | 2022-01-01 a 2025-12-01 | YYYY-MM-DD |

---

## 2. VARIABLES SOCIOECONOMICAS PRINCIPALES

### 2.1 Índice de Pobreza Multidimensional (IPM)

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `IPM_Total` | Índice de Pobreza Multidimensional total departamental | Porcentaje (%) | 75.0% (2022-2024) | DANE - ECV |

**Definición:** Mide la pobreza desde múltiples dimensiones: educación, niñez y juventud, trabajo, salud, servicios públicos domiciliarios y condiciones de la vivienda. Valores más altos indican mayor pobreza multidimensional.

**Interpretación:**
- 0-20%: Pobreza multidimensional baja
- 20-40%: Pobreza multidimensional moderada  
- 40-60%: Pobreza multidimensional alta
- >60%: Pobreza multidimensional muy alta

### 2.2 Índice de Pobreza de Consumo - Alimentación (IPC)

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `IPC_Total` | Índice de Pobreza de Consumo - componente alimentación | Porcentaje (%) | 87.5% (2022-2025) | DANE - GEIH |

**Definición:** Mide la proporción de población que no puede costear una canasta básica de alimentos. Se enfoca específicamente en el componente alimentario del IPC, evaluando el acceso económico a alimentos básicos necesarios para una nutrición adecuada.

**Interpretación:**
- 0-10%: Acceso alimentario bueno
- 10-20%: Acceso alimentario moderado
- 20-35%: Acceso alimentario limitado
- >35%: Acceso alimentario crítico

---

## 3. VARIABLES DE SEGURIDAD ALIMENTARIA (FIES)

### 3.1 Variables de Prevalencia (Clasificación General)

| Variable | Descripción | Unidad | Cobertura | Período |
|----------|-------------|--------|-----------|---------|
| `FIES_inseguridad_alimentaria` | Prevalencia total de inseguridad alimentaria | Porcentaje (%) | 75.0% | 2022-2024 |
| `FIES_leve_moderado` | Prevalencia de inseguridad alimentaria moderada o grave | Porcentaje (%) | 75.0% | 2022-2024 |
| `FIES_moderada` | Prevalencia de inseguridad alimentaria moderada (calculada) | Porcentaje (%) | 75.0% | 2022-2024 |
| `FIES_grave` | Prevalencia de inseguridad alimentaria grave | Porcentaje (%) | 75.0% | 2022-2024 |

**Fuente:** DANE - Encuesta de Calidad de Vida (ECV)

**Definiciones:**
- **Inseguridad Alimentaria:** Acceso limitado o incierto a alimentos nutritivos y culturalmente apropiados
- **Moderada o Grave:** Combinación de niveles moderado y grave de inseguridad alimentaria (dato original DANE)
- **Moderada (calculada):** Variable derivada calculada manualmente como la diferencia entre "moderada o grave" y "grave" (FIES_leve_moderado - FIES_grave). Representa específicamente el nivel moderado de inseguridad alimentaria
- **Grave:** Reducción severa en ingesta de alimentos, hambre, no comer por días enteros (dato original DANE)

### 3.2 Variables Detalladas (Experiencias Específicas)

#### 3.2.1 Nivel de Acceso (Leve)

| Variable | Descripción Completa | Unidad | Cobertura | Período |
|----------|---------------------|--------|-----------|---------|
| `FIES_preocupacion_alimentos` | Porcentaje de hogares que experimentaron preocupación o ansiedad por no tener suficientes alimentos para comer debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |

**Pregunta FIES:** "Durante los últimos 12 meses, ¿hubo algún momento en que usted se preocupó porque su hogar no tuviera suficientes alimentos para comer?"

**Interpretación:** Indicador temprano de inseguridad alimentaria. Refleja ansiedad y preocupación sobre acceso futuro a alimentos.

**Relevancia:** Primera señal de vulnerabilidad alimentaria, útil para intervención preventiva.

#### 3.2.2 Nivel de Calidad (Leve-Moderado)

| Variable | Descripción Completa | Unidad | Cobertura | Período |
|----------|---------------------|--------|-----------|---------|
| `FIES_no_alimentos_saludables` | Porcentaje de hogares que no pudieron comer alimentos saludables y nutritivos debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |
| `FIES_poca_variedad_alimentos` | Porcentaje de hogares que consumieron solo unos pocos tipos de alimentos debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |

**Preguntas FIES:**
- "¿No pudieron comer alimentos saludables y nutritivos?"
- "¿Consumieron solo unos pocos tipos de alimentos?"

**Interpretación:** Compromiso en calidad nutricional. Los hogares acceden a alimentos pero de menor calidad o variedad.

**Relevancia:** Indica riesgo nutricional y monotonía dietética que puede llevar a deficiencias.

#### 3.2.3 Nivel de Cantidad (Moderado)

| Variable | Descripción Completa | Unidad | Cobertura | Período |
|----------|---------------------|--------|-----------|---------|
| `FIES_saltar_comida` | Porcentaje de hogares donde al menos un integrante tuvo que saltar una comida porque no había suficiente dinero u otros recursos para obtener alimentos | Porcentaje (%) | 50.0% | 2023-2024 |
| `FIES_comio_menos` | Porcentaje de hogares donde al menos un integrante comió menos de lo que pensaba que debía comer debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |
| `FIES_sin_alimentos` | Porcentaje de hogares que se quedaron sin alimentos debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |

**Preguntas FIES:**
- "¿Algún integrante tuvo que saltar una comida?"
- "¿Algún integrante comió menos de lo que pensaba que debía?"
- "¿Su hogar se quedó sin alimentos?"

**Interpretación:** Reducción en cantidad de alimentos. Compromiso en frecuencia y volumen de comidas.

**Relevancia:** Indica inseguridad alimentaria moderada con impacto directo en ingesta calórica.

#### 3.2.4 Nivel Severo (Grave)

| Variable | Descripción Completa | Unidad | Cobertura | Período |
|----------|---------------------|--------|-----------|---------|
| `FIES_hambre_sin_comer` | Porcentaje de hogares donde al menos un integrante tuvo hambre pero no comió porque no había suficiente dinero u otros recursos para obtener alimentos | Porcentaje (%) | 50.0% | 2023-2024 |
| `FIES_no_comio_dia_entero` | Porcentaje de hogares donde al menos un integrante no comió en todo un día debido a falta de dinero u otros recursos | Porcentaje (%) | 50.0% | 2023-2024 |

**Preguntas FIES:**
- "¿Algún integrante tuvo hambre pero no comió?"
- "¿Algún integrante no comió en todo un día?"

**Interpretación:** Experiencias extremas de privación alimentaria. Hambre física y ayuno involuntario.

**Relevancia:** Indicadores de inseguridad alimentaria grave que requieren intervención urgente.

**Fuente:** DANE - Encuesta de Calidad de Vida (ECV)

**Marco Conceptual FIES:** La Escala de Experiencia de Inseguridad Alimentaria (FIES) es un instrumento validado internacionalmente por la FAO que mide la severidad de inseguridad alimentaria basada en experiencias reportadas por los hogares. Las 8 preguntas están ordenadas por severidad creciente, desde preocupación hasta privación extrema.

---

## 4. VARIABLES CLIMATICAS

### 4.1 Temperatura

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `temperatura_promedio` | Temperatura promedio mensual departamental | Grados Celsius (°C) | Variable por año | ERA5 |

**Definición:** Promedio mensual de temperatura del aire a 2 metros de altura, agregado espacialmente por departamento.

**Interpretación:**
- <20°C: Clima frío (zonas andinas)
- 20-25°C: Clima templado
- 25-30°C: Clima cálido
- >30°C: Clima muy cálido (zonas costeras/llanos)

**Relevancia:** La temperatura afecta la productividad agrícola, disponibilidad de alimentos y patrones de consumo, influyendo directamente en la seguridad alimentaria.

### 4.2 Precipitación

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `precipitacion_promedio` | Precipitación promedio mensual departamental | Milímetros (mm) | Variable por año | CHIRPS |

**Definición:** Promedio mensual de precipitación acumulada, derivado de datos satelitales y estaciones meteorológicas.

**Interpretación:**
- 0-50mm: Muy seco
- 50-100mm: Seco
- 100-200mm: Moderado
- 200-400mm: Húmedo
- >400mm: Muy húmedo

**Relevancia:** La precipitación es fundamental para la agricultura, afectando la producción de alimentos y por tanto la seguridad alimentaria regional.

### 4.3 Índices de Vegetación

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `ndvi_promedio` | Índice de Vegetación de Diferencia Normalizada promedio | Adimensional (0-1) | Variable por año | MODIS |

**Definición:** Índice que mide la densidad y salud de la vegetación basado en la reflectancia de luz roja e infrarroja cercana.

**Fórmula:** NDVI = (NIR - RED) / (NIR + RED)

**Interpretación:**
- 0.0-0.2: Suelo desnudo, agua, áreas urbanas
- 0.2-0.4: Vegetación escasa, pastizales secos
- 0.4-0.6: Vegetación moderada, cultivos
- 0.6-0.8: Vegetación densa, bosques
- 0.8-1.0: Vegetación muy densa, selvas

**Relevancia:** Indicador proxy de productividad agrícola y disponibilidad de recursos naturales para seguridad alimentaria.

---

## 5. VARIABLES IPM - DIMENSIONES ESPECÍFICAS

### 5.1 Dimensión Educación

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Analfabetismo` | Hogares con al menos una persona de 15 años o más analfabeta | Porcentaje (%) | Variable | DANE - ECV |
| `Bajo_logro_educativo` | Hogares donde ninguna persona de 15 años o más alcanzó nivel educativo mínimo | Porcentaje (%) | Variable | DANE - ECV |
| `Inasistencia_escolar` | Hogares con al menos un niño/adolescente (6-16 años) que no asiste a institución educativa | Porcentaje (%) | Variable | DANE - ECV |
| `Rezago_escolar` | Hogares con al menos un niño/adolescente (7-17 años) en rezago escolar | Porcentaje (%) | Variable | DANE - ECV |

### 5.2 Dimensión Salud

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Barreras_acceso_salud` | Hogares con barreras de acceso a servicios de salud | Porcentaje (%) | Variable | DANE - ECV |
| `Sin_aseguramiento_salud` | Hogares con al menos una persona sin afiliación a sistema de salud | Porcentaje (%) | Variable | DANE - ECV |

### 5.3 Dimensión Trabajo

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Desempleo_larga_duracion` | Hogares con al menos una persona en desempleo de larga duración | Porcentaje (%) | Variable | DANE - ECV |
| `Trabajo_informal` | Hogares donde todas las personas ocupadas están en informalidad laboral | Porcentaje (%) | Variable | DANE - ECV |

---

## 6. VARIABLES ECV - CALIDAD DE VIDA

### 6.1 Percepción de Calidad de Vida

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Vida_general` | Percepción general de calidad de vida | Escala 1-10 | Variable | DANE - ECV |
| `Salud` | Percepción de estado de salud | Escala 1-10 | Variable | DANE - ECV |
| `Seguridad` | Percepción de seguridad en el entorno | Escala 1-10 | Variable | DANE - ECV |
| `Trabajo_actividad` | Satisfacción con trabajo/actividad principal | Escala 1-10 | Variable | DANE - ECV |
| `Tiempo_libre` | Satisfacción con tiempo libre disponible | Escala 1-10 | Variable | DANE - ECV |
| `Ingreso` | Satisfacción con nivel de ingresos | Escala 1-10 | Variable | DANE - ECV |

**Interpretación Escalas de Percepción:**
- 1-3: Muy insatisfecho/Muy malo
- 4-5: Insatisfecho/Malo
- 6-7: Neutral/Regular
- 8-9: Satisfecho/Bueno
- 10: Muy satisfecho/Excelente

### 6.2 Tenencia de Vivienda

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Propia_totalmente_pagada` | Hogares con vivienda propia totalmente pagada | Porcentaje (%) | Variable | DANE - ECV |
| `Propia_la_estan_pagando` | Hogares con vivienda propia en proceso de pago | Porcentaje (%) | Variable | DANE - ECV |
| `En_arriendo_o_subarriendo` | Hogares que viven en arriendo o subarriendo | Porcentaje (%) | Variable | DANE - ECV |
| `Con_permiso_sin_pago` | Hogares que viven con permiso del propietario sin pago | Porcentaje (%) | Variable | DANE - ECV |
| `Posesion_sin_titulo` | Hogares en posesión de vivienda sin título legal | Porcentaje (%) | Variable | DANE - ECV |
| `Propiedad_colectiva` | Hogares en vivienda de propiedad colectiva | Porcentaje (%) | Variable | DANE - ECV |

### 6.3 Déficit Habitacional

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Deficit_cuantitativo` | Déficit cuantitativo de vivienda (falta de viviendas) | Porcentaje (%) | Variable | DANE - ECV |
| `Deficit_cualitativo` | Déficit cualitativo de vivienda (mejoramiento requerido) | Porcentaje (%) | Variable | DANE - ECV |
| `Deficit_habitacional` | Déficit habitacional total (cuantitativo + cualitativo) | Porcentaje (%) | Variable | DANE - ECV |

### 6.4 Capacidad de Gasto

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `No_alcanzan_gastos_minimos` | Hogares que no alcanzan gastos mínimos de subsistencia | Porcentaje (%) | Variable | DANE - ECV |
| `Alcanzan_gastos_minimos` | Hogares que alcanzan exactamente gastos mínimos | Porcentaje (%) | Variable | DANE - ECV |
| `Cubren_mas_gastos_minimos` | Hogares que superan gastos mínimos de subsistencia | Porcentaje (%) | Variable | DANE - ECV |

### 6.5 Pobreza Monetaria

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Pobreza_monetaria` | Hogares en situación de pobreza monetaria | Porcentaje (%) | Variable | DANE - ECV |
| `No_pobres` | Hogares que no están en situación de pobreza monetaria | Porcentaje (%) | Variable | DANE - ECV |

### 6.6 Servicios Públicos

| Variable | Descripción | Unidad | Cobertura | Fuente |
|----------|-------------|--------|-----------|--------|
| `Energia` | Hogares con acceso a energía eléctrica | Porcentaje (%) | Variable | DANE - ECV |
| `Gas_natural` | Hogares con acceso a gas natural | Porcentaje (%) | Variable | DANE - ECV |
| `Acueducto` | Hogares con acceso a acueducto | Porcentaje (%) | Variable | DANE - ECV |
| `Alcantarillado` | Hogares con acceso a alcantarillado | Porcentaje (%) | Variable | DANE - ECV |
| `Recoleccion_basura` | Hogares con servicio de recolección de basura | Porcentaje (%) | Variable | DANE - ECV |
| `Telefono_fijo` | Hogares con servicio de telefonía fija | Porcentaje (%) | Variable | DANE - ECV |
| `Ningun_servicio` | Hogares sin acceso a ningún servicio público | Porcentaje (%) | Variable | DANE - ECV |

**Relevancia para Seguridad Alimentaria:**
- **Energía:** Necesaria para conservación y preparación de alimentos
- **Gas:** Combustible para cocción de alimentos
- **Acueducto:** Agua potable para higiene alimentaria
- **Alcantarillado:** Saneamiento que previene enfermedades
- **Recolección basura:** Manejo de residuos alimentarios

---

## 7. MARCO TEÓRICO Y CONCEPTUAL

### 7.1 Seguridad Alimentaria - Definición FAO

**Definición Oficial:** "La seguridad alimentaria existe cuando todas las personas tienen, en todo momento, acceso físico, social y económico a alimentos suficientes, inocuos y nutritivos que satisfacen sus necesidades energéticas diarias y preferencias alimentarias para llevar una vida activa y sana" (FAO, 1996).

**Cuatro Pilares de la Seguridad Alimentaria:**

1. **Disponibilidad:** Existencia de alimentos en cantidad suficiente
2. **Acceso:** Capacidad económica y física para obtener alimentos
3. **Utilización:** Uso adecuado de alimentos para lograr bienestar nutricional
4. **Estabilidad:** Acceso continuo a alimentos a lo largo del tiempo

### 7.2 Relación entre Variables del Estudio

**IPM → Seguridad Alimentaria:**
- Educación afecta conocimientos nutricionales y oportunidades laborales
- Salud influye en utilización de nutrientes y capacidad productiva
- Trabajo determina ingresos para acceso a alimentos
- Servicios públicos facilitan conservación y preparación de alimentos

**IPC → Seguridad Alimentaria:**
- Mide directamente la capacidad económica para acceder a alimentos básicos
- Indicador específico del pilar "acceso" de seguridad alimentaria

**Variables Climáticas → Seguridad Alimentaria:**
- Temperatura y precipitación afectan producción agrícola local
- NDVI indica productividad agrícola y disponibilidad de alimentos
- Variabilidad climática genera inestabilidad en disponibilidad

### 7.3 Escalas de Medición FIES

**Escala Rasch:** Las variables FIES siguen un modelo de Rasch que permite ordenar las experiencias por severidad:

1. **Leve:** Preocupación, ansiedad sobre acceso futuro
2. **Leve-Moderado:** Compromiso en calidad y variedad
3. **Moderado:** Reducción en cantidad, saltar comidas
4. **Grave:** Hambre física, ayuno involuntario

**Validación Psicométrica:** Cada pregunta FIES tiene un nivel de severidad específico validado internacionalmente, permitiendo comparaciones entre países y regiones.

---

## 8. METODOLOGÍA DE CONSTRUCCIÓN DE LA BASE

### 8.1 Proceso de Integración

**Fase 1: Preparación de Datos**
- Normalización de nombres de departamentos (eliminación de tildes)
- Estandarización de códigos DANE
- Conversión de formatos temporales (números a nombres de meses)

**Fase 2: Integración Temporal**
- IPM: Extrapolación de datos anuales a frecuencia mensual
- IPC: Integración mensual con mapeo de ciudades a departamentos
- FIES: Integración específica por año (2022-2024 prevalencia, 2023-2024 detalladas)
- Variables climáticas: Agregación espacial por departamento

**Fase 3: Validación y Verificación**
- Verificación de cobertura temporal por variable
- Validación de consistencia entre variables relacionadas
- Confirmación de integridad espacial (32 departamentos)

### 8.2 Tratamiento de Datos Específicos

**Duplicación Bogotá-Cundinamarca (IPC):**
- Datos IPC de Bogotá aplicados también a Cundinamarca
- Justificación: Proximidad geográfica y integración económica

**Exclusión San Andrés:**
- Mantenimiento de 32 departamentos continentales
- Justificación: Diferencias insulares en dinámicas socioeconómicas

**Extrapolación Mensual:**
- Variables anuales distribuidas uniformemente por meses
- Aplicado a: IPM, FIES prevalencia, variables ECV

### 8.3 Fuentes y Metodologías Oficiales

**DANE - Metodología IPM:**
- Basada en enfoque Alkire-Foster
- 5 dimensiones, 15 indicadores
- Punto de corte: 33.3% de privaciones ponderadas

**DANE - Metodología IPC:**
- Líneas de pobreza basadas en Canasta Básica de Alimentos (CBA)
- Componente alimentario específico del índice general
- Actualización periódica según inflación alimentaria

**FAO - Metodología FIES:**
- Escala validada internacionalmente
- Modelo de Rasch para ordenamiento por severidad
- 8 preguntas estándar aplicadas globalmente

---

## 9. LIMITACIONES Y CONSIDERACIONES

### 9.1 Limitaciones Temporales

**Datos Faltantes por Año:**
- **2022:** No disponible FIES detalladas (solo prevalencia)
- **2025:** Datos parciales para la mayoría de variables (año de predicción)
- **Variables ECV:** Cobertura variable según disponibilidad de encuestas

### 9.2 Limitaciones Espaciales

**Agregación Departamental:**
- Pérdida de variabilidad municipal
- Promediación que puede ocultar heterogeneidad interna
- No captura diferencias urbano-rurales dentro de departamentos

### 9.3 Limitaciones Metodológicas

**Extrapolación Mensual:**
- Variables socioeconómicas originalmente anuales
- Asunción de estabilidad mensual puede no reflejar variaciones reales
- Pérdida de estacionalidad natural en variables socioeconómicas

**Integración de Fuentes:**
- Diferentes metodologías de recolección entre encuestas
- Posibles diferencias en marcos muestrales
- Variabilidad en calidad de datos entre años

### 9.4 Consideraciones para Análisis

**Autocorrelación Temporal:**
- Variables extrapoladas mensualmente presentan alta correlación interna
- Necesario considerar en modelos econométricos

**Heterogeneidad Espacial:**
- Grandes diferencias entre departamentos requieren controles espaciales
- Considerar efectos de vecindad y spillovers regionales

**Estacionalidad Climática:**
- Variables climáticas con patrones estacionales marcados
- Importante para modelos predictivos y análisis de causalidad

---

## 10. NOTAS METODOLÓGICAS

### 6.1 Cobertura Temporal

- **2022:** IPM ✅ | IPC ✅ | FIES Prevalencia ✅ | FIES Detalladas ❌
- **2023:** IPM ✅ | IPC ✅ | FIES Prevalencia ✅ | FIES Detalladas ✅
- **2024:** IPM ✅ | IPC ✅ | FIES Prevalencia ✅ | FIES Detalladas ✅
- **2025:** IPM ❌ | IPC 50% | FIES ❌ (para predicción)

### 6.2 Tratamiento de Datos Faltantes

- **Variables socioeconomicas:** Valores faltantes para años sin datos disponibles
- **Variables climáticas:** Cobertura variable según disponibilidad de datos satelitales
- **2025:** Datos parciales para predicción y modelado

### 6.3 Normalización Espacial

- **Departamentos:** Nombres normalizados sin tildes para consistencia
- **Códigos DANE:** Códigos oficiales mantenidos
- **Exclusiones:** San Andrés excluido para mantener 32 departamentos continentales

### 6.4 Frecuencia Temporal

- **Datos anuales extrapolados:** Variables socioeconomicas distribuidas mensualmente
- **Datos mensuales nativos:** Variables climáticas con frecuencia original
- **Interpolación:** Aplicada donde fue necesario para completar series

---

## 7. REFERENCIAS Y FUENTES

### Fuentes de Datos

- **DANE:** Departamento Administrativo Nacional de Estadística
  - ECV: Encuesta de Calidad de Vida
  - GEIH: Gran Encuesta Integrada de Hogares
- **ERA5:** European Centre for Medium-Range Weather Forecasts
- **CHIRPS:** Climate Hazards Group InfraRed Precipitation with Station data
- **MODIS:** Moderate Resolution Imaging Spectroradiometer

### Metodologías

- **IPM:** Metodología oficial DANE basada en Alkire-Foster
- **FIES:** Food Insecurity Experience Scale (FAO)
- **IPC:** Líneas de pobreza oficiales DANE

---

## 11. USOS RECOMENDADOS

### 11.1 Análisis Descriptivo
- Caracterización de perfiles departamentales de pobreza y seguridad alimentaria
- Análisis de tendencias temporales 2022-2025
- Identificación de patrones estacionales en variables climáticas
- Comparaciones interdepartamentales de indicadores socioeconómicos

### 11.2 Modelado Predictivo
- Predicción de FIES para 2022 y 2025 usando datos 2023-2024
- Forecasting de variables climáticas para completar 2025
- Modelos de clasificación de inseguridad alimentaria
- Análisis de series temporales con componentes estacionales

### 11.3 Análisis Econométrico
- Modelos de efectos fijos por departamento
- Análisis de causalidad entre variables climáticas y seguridad alimentaria
- Estudios de impacto de políticas públicas
- Análisis de spillovers espaciales entre departamentos

---

## 12. GUÍA DE ANÁLISIS ESTADÍSTICO

### 12.1 Análisis Exploratorio Recomendado

**Estadísticas Descriptivas:**
```python
# Análisis por variable y año
df.groupby('año')[['IPM_Total', 'IPC_Total', 'FIES_inseguridad_alimentaria']].describe()

# Correlaciones entre variables principales
correlation_matrix = df[['IPM_Total', 'IPC_Total', 'precipitacion_promedio', 'temperatura_promedio']].corr()
```

**Visualizaciones Clave:**
- Mapas coropléticos por departamento
- Series temporales por variable
- Diagramas de dispersión entre variables socioeconómicas y climáticas
- Distribuciones de frecuencia de variables FIES

### 12.2 Modelos Sugeridos

**Especificaciones Econométricas:**
- **Efectos fijos:** Por departamento para controlar heterogeneidad no observada
- **Tendencias temporales:** Incluir variables de tiempo para capturar tendencias
- **Controles estacionales:** Variables dummy por mes o trimestre
- **Variables rezagadas:** Especialmente para variables climáticas (1-3 meses)

**Modelos Recomendados:**
- **Panel data:** Aprovechando dimensión temporal y espacial
- **Modelos de efectos mixtos:** Para jerarquía departamento-tiempo
- **Análisis de series de tiempo:** Para departamentos específicos
- **Modelos espaciales:** Considerando autocorrelación espacial

### 12.3 Validación de Modelos

**Validación Temporal:**
- Training: 2022-2023
- Validation: 2024
- Test: 2025 (predicción)

**Validación Cruzada Espacial:**
- Leave-one-department-out cross-validation
- Validación por regiones geográficas

---

## 13. CASOS DE USO ESPECÍFICOS

### 13.1 Predicción de Inseguridad Alimentaria

**Objetivo:** Predecir variables FIES usando indicadores socioeconómicos y climáticos

**Variables Predictoras:**
- IPM_Total, IPC_Total (indicadores de pobreza)
- precipitacion_promedio, temperatura_promedio (factores climáticos)
- Variables ECV relevantes (educación, salud, servicios)

**Variables Objetivo:**
- FIES_inseguridad_alimentaria (clasificación general)
- FIES_grave (casos más severos)
- Variables FIES detalladas (experiencias específicas)

### 13.2 Análisis de Impacto Climático

**Objetivo:** Evaluar efectos del cambio climático en seguridad alimentaria

**Metodología:**
1. Análisis de correlación clima-FIES por departamento
2. Modelos de regresión con rezagos temporales
3. Identificación de umbrales críticos climáticos

### 13.3 Consideraciones Especiales

**Datos Faltantes:**
- **2022:** Solo IPM, IPC y FIES prevalencia disponibles
- **2023-2024:** Cobertura completa de variables FIES
- **2025:** Datos parciales, ideal para predicción

**Limitaciones:**
- Variables ECV con cobertura variable por año
- Datos climáticos sujetos a disponibilidad satelital
- Extrapolación mensual de datos anuales en variables socioeconómicas

---

## 14. ANEXOS

### 14.1 Códigos DANE de Departamentos

| Código | Departamento | Región |
|--------|--------------|---------|
| 05 | Antioquia | Andina |
| 08 | Atlantico | Caribe |
| 11 | Bogotá | Andina |
| 13 | Bolivar | Caribe |
| 15 | Boyaca | Andina |
| 17 | Caldas | Andina |
| 18 | Caqueta | Amazónica |
| 19 | Cauca | Pacífica |
| 20 | Cesar | Caribe |
| 23 | Cordoba | Caribe |
| 25 | Cundinamarca | Andina |
| 27 | Choco | Pacífica |
| 41 | Huila | Andina |
| 44 | Guajira | Caribe |
| 47 | Magdalena | Caribe |
| 50 | Meta | Orinoquía |
| 52 | Nariño | Pacífica |
| 54 | Norte De Santander | Andina |
| 63 | Quindío | Andina |
| 66 | Risaralda | Andina |
| 68 | Santander | Andina |
| 70 | Sucre | Caribe |
| 73 | Tolima | Andina |
| 76 | Valle Del Cauca | Pacífica |
| 81 | Arauca | Orinoquía |
| 85 | Casanare | Orinoquía |
| 86 | Putumayo | Amazónica |
| 91 | Amazonas | Amazónica |
| 94 | Guainia | Amazónica |
| 95 | Guaviare | Amazónica |
| 97 | Vaupes | Amazónica |
| 99 | Vichada | Orinoquía |

### 14.2 Estructura de Archivos del Proyecto

```
data/
├── original/                    # Datos fuente originales
│   ├── anex-FIES-2024_limpio.csv
│   ├── ipm_departamental_2022_2024.csv
│   └── datos_climaticos_*.csv
├── procesado/                   # Datos procesados intermedios
│   ├── fies_2023_mensual.csv
│   ├── ecv_*_mensual_EXTRAPOLADO.csv
│   └── climaticos_*_procesados.csv
└── base de datos central/       # Base master final
    ├── BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.xlsx
    ├── BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.csv
    └── DICCIONARIO_VARIABLES_BASE_MASTER.md
```

---

## 15. REFERENCIAS

### 15.1 Fuentes Oficiales

- **DANE** (2022). Metodología Índice de Pobreza Multidimensional (IPM). Departamento Administrativo Nacional de Estadística.
- **DANE** (2023). Gran Encuesta Integrada de Hogares (GEIH) - Metodología. Departamento Administrativo Nacional de Estadística.
- **DANE** (2024). Encuesta de Calidad de Vida (ECV) - Metodología y resultados.
- **FAO** (2016). Methods for estimating comparable rates of food insecurity experienced by adults throughout the world. Food and Agriculture Organization.
- **FAO** (1996). Rome Declaration on World Food Security and World Food Summit Plan of Action.
- **IDEAM** (2023). Metodología de datos climáticos departamentales. Instituto de Hidrología, Meteorología y Estudios Ambientales.

### 15.2 Fuentes Técnicas

- **NASA** (2023). MODIS Vegetation Index Products (MOD13Q1). Land Processes Distributed Active Archive Center.
- **Alkire, S. & Foster, J.** (2011). Counting and multidimensional poverty measurement. Journal of Public Economics, 95(7-8), 476-487.
- **Ballard, T., Kepple, A. & Cafiero, C.** (2013). The food insecurity experience scale: development of a global standard for monitoring hunger worldwide. FAO Technical Paper.

### 15.3 Documentación Técnica

- **Rasch, G.** (1960). Probabilistic models for some intelligence and attainment tests. Danish Institute for Educational Research.
- **Smith, M.D., Rabbitt, M.P. & Coleman‐Jensen, A.** (2017). Who are the world's food insecure? New evidence from the Food and Agriculture Organization's food insecurity experience scale. World Development, 93, 402-412.

---

## 16. CONTACTO Y SOPORTE

### 16.1 Información del Proyecto

**Proyecto:** Análisis Predictivo de Seguridad Alimentaria en Colombia  
**Institución:** [Institución Académica]  
**Nivel:** Tesis de Maestría  
**Período de Estudio:** 2022-2025  
**Cobertura Geográfica:** 32 Departamentos Continentales de Colombia  

### 16.2 Archivos Principales

**Base de Datos Principal:**
- `BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.xlsx`
- `BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.csv`

**Documentación:**
- `DICCIONARIO_VARIABLES_BASE_MASTER.md` (este documento)

**Ubicación:** `D:\Tesis maestria\Tesis codigo\data\base de datos central\`

---

**Documento creado:** Diciembre 2024  
**Versión:** 1.0 - Versión Completa y Definitiva  
**Autor:** Análisis de Tesis - Seguridad Alimentaria Colombia  
**Última actualización:** Diciembre 2024  
**Estado:** Documento Finalizado - Listo para Uso en Tesis


# Configuración XGBoost Justificada para Datos PCA
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

def configurar_xgboost_pca():
    """Configuración XGBoost optimizada para datos PCA"""
    
    # Modelo base con parámetros justificados
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',  # Para regresión continua (FIES 0-100)
        random_state=42,               # Reproducibilidad
        n_jobs=-1,                     # Usar todos los cores
        verbosity=1                    # Información de progreso
    )
    
    # Grid search con parámetros justificados
    param_grid = {
        'estimator__n_estimators': [50, 100, 150],      # Conservador para datos limitados
        'estimator__max_depth': [3, 4, 5, 6],           # Incluye rango Martini + conservador
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Incluye rango Martini + gradual
        'estimator__subsample': [0.8, 0.9, 1.0],        # Mayoría de datos + baseline
        'estimator__colsample_bytree': [0.8, 1.0],       # Adaptado a 7 features PCA
        'estimator__reg_alpha': [0, 0.01, 0.1],          # L1: incluye baseline Martini
        'estimator__reg_lambda': [0, 0.01, 0.1]          # L2: incluye baseline Martini
    }
    
    # MultiOutput para 2 targets simultáneos
    modelo = MultiOutputRegressor(xgb_base, n_jobs=-1)
    
    # Validación temporal para datos de panel
    cv = TimeSeriesSplit(n_splits=5)
    
    # Grid search con métricas de Martini
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring=['r2', 'neg_mean_absolute_error'],
        refit='r2',                    # Optimizar por R² como Martini
        n_jobs=-1,
        verbose=2
    )
    
    return grid_search

# Justificación de cada parámetro:
# 1. objective='reg:squarederror': FIES son valores continuos 0-100, no probabilidades
# 2. n_estimators=[50,100,150]: Datos limitados requieren menos árboles
# 3. max_depth=[3,4,5,6]: PCA redujo complejidad, incluye rango Martini
# 4. learning_rate=[0.01-0.2]: Incluye rango Martini + aprendizaje gradual
# 5. subsample=[0.8,0.9,1.0]: Usar mayoría de datos limitados
# 6. colsample_bytree=[0.8,1.0]: Solo 7 features PCA disponibles
# 7. reg_alpha/lambda: Prevenir overfitting + comparar con Martini (0)
# 8. TimeSeriesSplit: Respeta estructura temporal de panel
# 9. MultiOutputRegressor: Predicción simultánea de 2 targets FIES

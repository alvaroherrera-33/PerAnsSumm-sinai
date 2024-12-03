from ydata_profiling import ProfileReport

# Carga de datos
import pandas as pd
df = pd.read_json("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/valid.json")

# Generación del reporte
profile = ProfileReport(df, title="Reporte de EDA", explorative=True)

# Guardar como archivo HTML
profile.to_file("eda_reporte.html")

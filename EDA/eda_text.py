import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos del dataset
ruta_dataset = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json'  # Asegúrate de colocar la ruta correcta
df = pd.read_json(ruta_dataset)

# 2. Crear carpeta de salida
output_folder = 'EDA'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, 'estadisticas_respuestas.txt')

# 3. Función para contar palabras en una respuesta
def contar_palabras(respuesta):
    return len(respuesta.split())

# 4. Inicializar lista para almacenar estadísticas
estadisticas_respuestas = []

# 5. Recorrer el dataset para procesar las respuestas
for index, row in df.iterrows():
    uri = row['uri']
    answers = row['answers']
    
    for i, answer in enumerate(answers):
        # Clasificar la respuesta según su tipo
        if "SUGGESTION" in row['labelled_answer_spans']:
            respuesta_tipo = "SUGGESTION"
        elif "CAUSE" in row['labelled_answer_spans']:
            respuesta_tipo = "CAUSE"
        elif "INFORMATION" in row['labelled_answer_spans']:
            respuesta_tipo = "INFORMATION"
        elif "QUESTION" in row['labelled_answer_spans']:
            respuesta_tipo = "QUESTION"
        elif "EXPERIENCE" in row['labelled_answer_spans']:
            respuesta_tipo = "EXPERIENCE"
        num_palabras = contar_palabras(answer)
        
        # Guardar estadísticas por cada respuesta
        estadisticas_respuestas.append({
            'uri': uri,
            'respuesta_tipo': respuesta_tipo,
            'respuesta': answer,
            'num_palabras': num_palabras
        })

# Convertir a DataFrame
df_estadisticas = pd.DataFrame(estadisticas_respuestas)

# 6. Calcular estadísticas generales sobre el número de palabras
estadisticas_generales = {
    'max_palabras': df_estadisticas['num_palabras'].max(),
    'min_palabras': df_estadisticas['num_palabras'].min(),
    'media_palabras': df_estadisticas['num_palabras'].mean(),
    'std_palabras': df_estadisticas['num_palabras'].std(),
}

# Guardar estadísticas generales
with open(output_file, 'w') as f:
    f.write("Estadísticas generales de las respuestas:\n")
    for key, value in estadisticas_generales.items():
        f.write(f"{key}: {value}\n")

# 7. Calcular estadísticas por tipo de respuesta
estadisticas_por_tipo = df_estadisticas.groupby('respuesta_tipo')['num_palabras'].describe()

# Guardar estadísticas por tipo
with open(output_file, 'a') as f:
    f.write("\nEstadísticas de número de palabras por tipo de respuesta:\n")
    f.write(estadisticas_por_tipo.to_string() + "\n")

# 8. Visualizar distribución de número de palabras por tipo de respuesta
plt.figure(figsize=(10, 6))
sns.boxplot(x='respuesta_tipo', y='num_palabras', data=df_estadisticas)
plt.title('Distribución del número de palabras por tipo de respuesta')
plt.savefig(os.path.join(output_folder, 'distribucion_palabras_por_tipo.jpg'))
plt.close()

# 9. Visualización: Comparar número de palabras por tipo de respuesta con un gráfico de dispersión
plt.figure(figsize=(10, 6))
sns.scatterplot(x='respuesta_tipo', y='num_palabras', data=df_estadisticas)
plt.title('Comparación de tipos de respuestas con número de palabras')
plt.savefig(os.path.join(output_folder, 'comparacion_etiquetas_respuestas.jpg'))
plt.close()

print(f"Análisis completo. Los resultados han sido guardados en la carpeta: {output_folder}")

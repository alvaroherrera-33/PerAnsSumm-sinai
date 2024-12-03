import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el archivo JSON
ruta_json = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/valid.json'  # Reemplaza con la ruta de tu archivo
df = pd.read_json(ruta_json)

# 1. Inspección inicial del DataFrame
# Ver las primeras filas del DataFrame
print(df.head())

# Información general sobre el DataFrame
print(df.info())

# Estadísticas descriptivas
print(df.describe())

# 2. Revisar valores nulos
# Verificar valores nulos
print(df.isnull().sum())

# 3. Distribución de respuestas
# Ver cuántas respuestas tiene cada pregunta
print(df['answers'].apply(len).describe())

# Mostrar la cantidad de respuestas por entrada
print(df['answers'].apply(len).value_counts())

# 4. Extraer todas las etiquetas completas de las respuestas
# Ver las etiquetas en las respuestas
answer_labels = df['labelled_answer_spans'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
answer_labels = answer_labels.apply(lambda x: x).explode().value_counts()
print("Nombres completos de las etiquetas en respuestas:")
print(answer_labels)

# 5. Extraer todas las etiquetas completas de los resúmenes
# Ver las etiquetas en los resúmenes
summary_labels = df['labelled_summaries'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
summary_labels = summary_labels.apply(lambda x: x).explode().value_counts()
print("Nombres completos de las etiquetas en resúmenes:")
print(summary_labels)

# 6. Relación entre preguntas y etiquetas
# Mostrar las preguntas junto con las etiquetas asociadas
print(df[['question', 'labelled_answer_spans']].head())

# 7. Visualización y guardar los gráficos como imágenes

# 7.1. Histograma de la cantidad de respuestas por pregunta
plt.figure(figsize=(8, 6))
df['answers'].apply(len).plot(kind='hist', bins=20, color='skyblue', edgecolor='black', title="Distribución del número de respuestas por pregunta")
plt.xlabel("Número de respuestas")
plt.ylabel("Frecuencia")
plt.savefig("histograma_respuestas.png")  # Guardar la imagen
plt.close()

# 7.2. Gráfico de barras para la distribución de etiquetas en las respuestas
plt.figure(figsize=(8, 6))
answer_labels.plot(kind='bar', color='lightgreen', title="Distribución de etiquetas en las respuestas")
plt.xlabel("Etiqueta")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.tight_layout()  # Ajuste para que las etiquetas no se superpongan
plt.savefig("distribucion_etiquetas_respuestas.png")  # Guardar la imagen
plt.close()

# 7.3. Gráfico de barras para la distribución de etiquetas en los resúmenes
plt.figure(figsize=(8, 6))
summary_labels.plot(kind='bar', color='lightcoral', title="Distribución de etiquetas en los resúmenes")
plt.xlabel("Etiqueta")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.tight_layout()  # Ajuste para que las etiquetas no se superpongan
plt.savefig("distribucion_etiquetas_resumenes.png")  # Guardar la imagen
plt.close()

print("Los gráficos han sido guardados como imágenes.")

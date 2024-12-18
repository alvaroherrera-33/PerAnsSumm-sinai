import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import Counter


# Configuración inicial
ruta_archivo = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json'  # Cambia esto por la ruta de tu archivo
output_folder = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/task_a'  # Cambia esto por tu carpeta de salida

# Leer el archivo CSV
df = pd.read_json(ruta_archivo)

# Seleccionar las columnas de las categorías (ajusta según tu dataset)
categorias = df.columns[3:]  # Excluir 'id', 'sentence_index', etc.

# Crear una columna con combinaciones de categorías activas
df['combinacion'] = df[categorias].apply(lambda x: ','.join([col for col in categorias if x[col] == 1]), axis=1)

# 1. Análisis de combinaciones de categorías
# Contar la frecuencia de cada combinación
frecuencia_combinaciones = df['combinacion'].value_counts()

# Guardar las combinaciones más frecuentes en un archivo
output_file_combinaciones = os.path.join(output_folder, 'frecuencia_combinaciones.txt')
with open(output_file_combinaciones, 'w') as f:
    f.write("Frecuencia de combinaciones de categorías:\n\n")
    for combinacion, frecuencia in frecuencia_combinaciones.items():
        f.write(f"Combinación: {combinacion}, Frecuencia: {frecuencia}\n")

# Visualizar las combinaciones más frecuentes
plt.figure(figsize=(10, 6))
sns.barplot(y=frecuencia_combinaciones.head(10).index, x=frecuencia_combinaciones.head(10).values)
plt.title('Top 10 combinaciones más frecuentes')
plt.xlabel('Frecuencia')
plt.ylabel('Combinaciones de categorías')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_10_combinaciones.jpg'))
plt.close()

# 2. Análisis de transiciones
# Ordenar por 'id' y 'sentence_index' para analizar las secuencias
df = df.sort_values(by=['id', 'sentence_index'])

# Agrupar por 'id' para analizar las secuencias de combinaciones
secuencias = df.groupby('id')['combinacion'].apply(lambda x: list(x)).reset_index()

# Crear tabla de transiciones
transiciones = []
for seq in secuencias['combinacion']:
    for i in range(len(seq) - 1):
        transiciones.append((seq[i], seq[i + 1]))

# Contar la frecuencia de las transiciones
frecuencia_transiciones = Counter(transiciones)

# Convertir a DataFrame
transiciones_df = pd.DataFrame(frecuencia_transiciones.items(), columns=['Transición', 'Frecuencia'])

# Separar origen y destino
transiciones_df[['Origen', 'Destino']] = pd.DataFrame(transiciones_df['Transición'].tolist(), index=transiciones_df.index)
transiciones_df = transiciones_df.drop(columns=['Transición'])

# Guardar las transiciones más frecuentes en un archivo
output_file_transiciones = os.path.join(output_folder, 'frecuencia_transiciones.txt')
with open(output_file_transiciones, 'w') as f:
    f.write("Frecuencia de transiciones entre combinaciones de categorías:\n\n")
    for _, row in transiciones_df.iterrows():
        f.write(f"Transición: {row['Origen']} -> {row['Destino']}, Frecuencia: {row['Frecuencia']}\n")

# Visualizar las transiciones más frecuentes
plt.figure(figsize=(10, 6))
sns.barplot(y=transiciones_df.head(10)['Origen'] + ' -> ' + transiciones_df.head(10)['Destino'], 
            x=transiciones_df.head(10)['Frecuencia'])
plt.title('Top 10 transiciones más frecuentes')
plt.xlabel('Frecuencia')
plt.ylabel('Transición')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_10_transiciones.jpg'))
plt.close()

print(f"Análisis completado. Los resultados están guardados en la carpeta: {output_folder}")

import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import Counter
from nltk.corpus import stopwords


# Asumiendo que ya has cargado el dataset en un DataFrame
# dataset = pd.read_json("ruta_al_archivo.json")  # Si usas un archivo JSON

# Obtener las stopwords de NLTK en español
stop_words = set(stopwords.words('english'))

# Función para limpiar y tokenizar el texto
def clean_and_tokenize(text):
    # Eliminar puntuación y poner todo en minúsculas
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizar
    words = text.split()
    # Filtrar las stopwords
    words = [word for word in words if word not in stop_words]
    return words

# Función para obtener las palabras más comunes
def get_most_common_words(texts, top_n=10):
    all_words = []
    for text in texts:
        all_words.extend(clean_and_tokenize(text))
    # Contar las palabras
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)
ruta_dataset = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/valid.json'  # Asegúrate de colocar la ruta correcta
dataset = pd.read_json(ruta_dataset)
# Crear un diccionario para almacenar las respuestas por etiqueta
label_texts = {}

# Iterar sobre el DataFrame para organizar las respuestas por etiqueta
for _, entry in dataset.iterrows():
    print(f"Analizando respuestas para la pregunta: {entry['question']}")
    
    # Acceder a la columna 'labelled_answer_spans' que es un diccionario
    if isinstance(entry["labelled_answer_spans"], dict):
        for label, answers in entry["labelled_answer_spans"].items():
            if isinstance(answers, list):  # Asegurarse de que 'answers' es una lista
                # Agrupar todos los textos de la etiqueta
                if label not in label_texts:
                    label_texts[label] = []
                texts = [answer["txt"] for answer in answers if isinstance(answer, dict) and "txt" in answer]
                label_texts[label].extend(texts)

# Ahora procesamos las palabras más comunes por etiqueta
for label, texts in label_texts.items():
    print(f"\nTop 10 palabras más comunes en la etiqueta '{label}':")
    most_common_words = get_most_common_words(texts, top_n=20)
    for word, freq in most_common_words:
        print(f"{word}: {freq}")

# Función para graficar las palabras más comunes
def plot_most_common_words(label, most_common_words, output_path):
    # Separar palabras y frecuencias
    words, freqs = zip(*most_common_words)
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(freqs), y=list(words), palette="viridis")
    plt.title(f"Top palabras más comunes para la etiqueta '{label}'")
    plt.xlabel("Frecuencia")
    plt.ylabel("Palabras")
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig(f"{output_path}/{label}_top_words.png")
    plt.close()  # Cerrar la figura para evitar sobrecarga

# Crear carpeta de salida si no existe
output_dir = "./output_graphs"
os.makedirs(output_dir, exist_ok=True)

# Procesar y graficar
for label, texts in label_texts.items():
    print(f"\nTop 10 palabras más comunes en la etiqueta '{label}':")
    most_common_words = get_most_common_words(texts, top_n=20)
    for word, freq in most_common_words:
        print(f"{word}: {freq}")
    
    # Graficar y guardar
    plot_most_common_words(label, most_common_words, output_dir)

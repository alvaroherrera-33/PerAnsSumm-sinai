import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, ClassLabel
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar el tokenizador y el modelo preentrenado de Bio_ClinicalBERT
tokenizer = AutoTokenizer.from_pretrained("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/bert-large-cased")
model = AutoModelForTokenClassification.from_pretrained("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/bert-large-cased", num_labels=12, ignore_mismatched_sizes=True)

# Función para cargar datos desde un archivo JSON
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Cargar datos desde archivos JSON separados
train_data = load_json_data("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/bio_train.json")
valid_data = load_json_data("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/bio_valid.json")

# Formatear los datos en el formato esperado
def format_data(data):
    formatted_data = {
        'answers': [],
        'tags': []
    }
    for item in data:
        answers = item['answers'].split()  # Asumiendo que los tokens están separados por espacios
        tags = item['tags']
        formatted_data['answers'].append(answers)
        formatted_data['tags'].append(tags)
    return formatted_data

train_formatted = format_data(train_data)
valid_formatted = format_data(valid_data)

# Convertir a Hugging Face Datasets
train_dataset = Dataset.from_dict(train_formatted)
valid_dataset = Dataset.from_dict(valid_formatted)

print(train_dataset[0])  # Muestra el primer elemento para ver cómo se estructura

# Asegúrate de que todas las etiquetas de tu dataset están en esta lista
label_list = ['O', 'B-INFORMATION', 'I-INFORMATION', 'B-CAUSE', 'I-CAUSE', 
              'B-EXPERIENCE', 'I-EXPERIENCE', 'B-SUGGESTION', 'I-SUGGESTION', 
              'B-QUESTION', 'I-QUESTION']
label_map = {label: i for i, label in enumerate(label_list)}

# Convertir las etiquetas a índices
def encode_tags(batch):
    return {
        'labels': [[label_map[tag] for tag in tags] for tags in batch['tags']]
    }

# Tokenización y alineación de etiquetas
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['answers'],
        truncation=True,
        padding=True, max_length=512,
        #padding='max_length',
        is_split_into_words=True
    )
    
    all_labels = examples['labels']
    new_labels = []
    
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Mapea tokens a palabras originales
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignorar tokens de padding
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])  # Etiqueta del primer token
            else:
                label_ids.append(labels[previous_word_idx])  # Etiquetas para subtokens
            previous_word_idx = word_idx
        new_labels.append(label_ids)
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# Mapear etiquetas a índices
train_dataset = train_dataset.map(encode_tags, batched=True)
valid_dataset = valid_dataset.map(encode_tags, batched=True)

# Tokenizar y alinear etiquetas
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
valid_dataset = valid_dataset.map(tokenize_and_align_labels, batched=True)


# Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=500,
    save_total_limit=2,
    fp16=True  # Precisión mixta
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

def analyze_probabilities(predictions, true_labels):
    confidences = np.max(predictions, axis=2)
    print("Confianza media por clase:")
    for i, label in enumerate(label_list):
        class_confidences = confidences[true_labels == i]
        print(f"  {label}: {np.mean(class_confidences):.2f}")


def analyze_errors(true_labels, pred_labels, output_dir="images"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_true = []
    all_pred = []
    for true, pred in zip(true_labels, pred_labels):
        all_true.extend(true)
        all_pred.extend(pred)
    
    # Crear matriz de confusión
    cm = confusion_matrix(all_true, all_pred, labels=label_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    
    # Guardar la imagen de la matriz de confusión
    confusion_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_image_path, bbox_inches='tight')
    plt.close()  # Cerrar la figura para liberar memoria
    print(f"Matriz de confusión guardada en: {confusion_image_path}")
    
    # Imprimir ejemplos mal clasificados
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            print(f"Ejemplo {i}:")
            print(f"  Verdadero: {true}")
            print(f"  Predicho: {pred}")

# Función para calcular métricas
# Función para calcular métricas
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # Obtener etiquetas predichas

    # Convertir índices a etiquetas
    true_labels = [[label_list[label] for label in label_row if label != -100] for label_row in labels]
    pred_labels = [[label_list[pred] for pred, lab in zip(pred_row, label_row) if lab != -100]
                   for pred_row, label_row in zip(predictions, labels)]

    # Calcular métricas generales
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Inicializar y entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('./final_model')

# Evaluar el modelo
metrics = trainer.evaluate()

# Obtener predicciones completas del conjunto de validación
outputs = trainer.predict(valid_dataset)
predictions, labels = outputs.predictions, outputs.label_ids
predictions = np.argmax(predictions, axis=2)  # Obtener etiquetas predichas

# Convertir índices a etiquetas
true_labels = [[label_list[label] for label in label_row if label != -100] for label_row in labels]
pred_labels = [[label_list[pred] for pred, lab in zip(pred_row, label_row) if lab != -100]
               for pred_row, label_row in zip(predictions, labels)]

# Análisis detallado
print("\n--- Análisis Detallado Final ---\n")

# Matriz de confusión y ejemplos mal clasificados
analyze_errors(true_labels, pred_labels)

# Reporte detallado por clase
print(classification_report(true_labels, pred_labels))

# Análisis de probabilidades
analyze_probabilities(outputs.predictions, labels)

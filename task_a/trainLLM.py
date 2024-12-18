# Instala las bibliotecas necesarias
# pip install transformers torch datasets

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

# Cargar el tokenizador y el modelo BioclinicalBert
# Cargar el tokenizador y el modelo BioclinicalBert con la opción para ignorar desajustes de tamaño
tokenizer = AutoTokenizer.from_pretrained("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/Bio_ClinicalBERT_medical", use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(
    "/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/Bio_ClinicalBERT_medical",
    num_labels=4,  # Ajusta el número de etiquetas
    ignore_mismatched_sizes=True  # Ignorar desajustes de tamaño en los parámetros
)

# Cargar los datos desde los archivos train.json y valid.json
train_dataset = load_dataset("json", data_files="/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json", split="train")
valid_dataset = load_dataset("json", data_files="/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/valid.json", split="train")

label_to_id = {'CAUSE': 1, 'INFORMATION': 2, 'SUGGESTION': 3, 'EXPERIENCE': 4, 'QUESTION': 5}


# Función para tokenizar y alinear las etiquetas
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['raw_text'], padding="max_length", truncation=True, max_length=128,  return_offsets_mapping=True)
    labels = [0] * len(tokenized_inputs['input_ids'])  # Inicia todas las etiquetas como 0 (ninguna perspectiva)

    for span in examples['labelled_answer_spans']:
        # Filtrar claves con contenido y extraer los "label_spans"
        filtered_label_spans = {
            key: [item['label_spans'] for item in value if 'label_spans' in item]  # Extraer solo los label_spans
            for key, value in span.items()
            if value is not None  # Solo claves con listas no vacías
        }

        # Etiqueta los tokens dentro del rango del span
        for label, spans in filtered_label_spans.items():
            label_id = label_to_id[label]  # Obtén el índice asociado a la etiqueta
            for start, end in spans:  # Desempaqueta las posiciones del span
                 for idx, offsets in enumerate(tokenized_inputs['offset_mapping']):
                    # Asegúrate de que offsets es una tupla o lista de longitud 2
                    if len(offsets) == 2:
                        token_start, token_end = offsets
                        if token_start is not None and token_end is not None:  # Ignorar valores especiales
                            if token_start >= start and token_end <= end:
                                labels[idx] = label_id  # Asigna la etiqueta correspondiente
    labels = labels[: len(tokenized_inputs['input_ids'])] 
    tokenized_inputs.pop('offset_mapping', None)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs
    
# Aplicar la tokenización y alineación de etiquetas al conjunto de datos
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
valid_dataset = valid_dataset.map(tokenize_and_align_labels, batched=True)

# Definir los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Configurar el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Entrenar el modelo
trainer.train()

# Función para evaluar el rendimiento del modelo con métricas de precisión, recall y F1
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"precision": precision, "recall": recall, "f1": f1}

# Evaluar el modelo
trainer.evaluate(compute_metrics=compute_metrics)

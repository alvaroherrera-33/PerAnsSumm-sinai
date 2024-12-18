import os
import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta del modelo entrenado
MODEL_PATH = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/Bio-Medical-Llama-3-8B'  # Modelo base
#FINETUNED_MODEL_PATH = '/mnt/beegfs/aarjonil/cardiologia/tasks/extract_acronyms/llama/results/final_checkpoints/llama3_1b_extract-final'   # Modelo ajustado (modifica según tu ruta)

# Inicializar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).cuda()
#model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)

model.config.pad_token_id = tokenizer.eos_token_id

# Definir el prompt
prompt_template = "Task Description: You are given a question \( Q \), a set of user answers \( A \), and a set of perspective categories: Information: Objective facts or explanations about the condition or procedure. Cause: Insights into what causes or could worsen the issue. Suggestion: Advice or recommendations to address the problem. Experience: Personal stories or shared experiences related to the problem. Question: Queries that seek additional clarification or recommendations. Your task is: 1. Span Identification: Locate specific spans within the answers \( A \) that correspond to one of these perspectives. 2. Classification: Assign the identified span to the correct perspective category. Input Example - Question (Q): 'I was just diagnosed with gallstones in my gall bladder. I really don’t want to have surgery and have been told that there are other ways to get rid of the stones. Suggestions?' - Answers (A): 1. 'Most gallstones are made of pure cholesterol. You might try a diet with low fat and very low saturated fats. I've had the surgery, and it really isn't a big deal. If you leave the gallstones there, they can get large enough to damage.' 2. 'Have you seen a gastroenterologist? They can do a minimally invasive procedure called an ERCP. I had the surgery myself about 10 years ago, and it really helped.' 3. 'The best remedy is surgery. I had surgery to have kidney stones removed. The surgery isn’t as bad as you think it may be.' Output Example - Information: 'Most gallstones are made of pure cholesterol.' Cause: 'If you leave the gallstones there, they can get large enough to damage.' Suggestion: 'You might try a diet with low fat and very low saturated fats.' Experience: 'I had the surgery myself about 10 years ago, and it really helped.' Question: 'Have you seen a gastroenterologist?' Key Considerations: Contextual understanding is crucial: Ensure that spans are precise and directly tied to the category. Medical knowledge will often be necessary to interpret and classify perspectives accurately. Output Format: For each span, return: 1. The text span. 2. The assigned perspective category."

'''
results = []
prompt = "Hola, en qué estás especializado como modelo?"
    
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    
    # Decodificar la salida
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)
results.append({'texto': prompt, 'predicción': prediction})

'''
# Leer archivo JSON de entrada
input_json = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/valid.json'  # Cambia la ruta al archivo JSON de entrada
output_json = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/task_a/salida.json'  # Ruta para guardar el archivo JSON de salida

#input_txt = '/mnt/beegfs/aarjonil/cardiologia/tasks/extract_acronyms/llama/informe10.txt'

# Leer el archivo de texto
#with open(input_txt, 'r', encoding='utf-8') as f:
 #   input_data = f.readlines()  # Lee todas las líneas del archivo TXT

with open(input_json, 'r', encoding='utf-8') as f:
    input_data = json.load(f)  # Espera una lista de objetos con clave "texto"

input_data = input_data[:10]
results = []
for entry in input_data:
    input_text = entry['answers']
    if not isinstance(input_text, str):
        input_text = str(input_text)
    prompt = prompt_template.format(texto=input_text)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text}
    ]
    for message in messages:
        if not isinstance(message["content"], str):
            raise ValueError(f"El contenido de un mensaje no es una cadena: {message}")

    inputs = tokenizer.apply_chat_template(messages, return_tensors= "pt", max_length=512, truncation=True, padding=True).to("cuda")
    #inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=400, do_sample=True)
    
    # Decodificar la salida
    prediction = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)

    #prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({'texto': input_text, 'predicción': prediction})

# Guardar resultados en un archivo JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Inferencia completada. Resultados guardados en {output_json}.")

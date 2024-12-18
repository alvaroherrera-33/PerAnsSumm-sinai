import json

def transform_to_bio(data):
    bio_data = []

    for item in data:
        for answer in item["answers"]:
            tokens = answer  # Caracteres individuales como tokens
            tags = ["O"] * len(tokens)  # Inicializamos con "O"

            for label, spans in item.get("labelled_answer_spans", {}).items():
                for span in spans:
                    if span["txt"] in answer:
                        start_idx = answer.find(span["txt"])
                        end_idx = start_idx + len(span["txt"])

                        # Asignar etiquetas BIO
                        tags[start_idx] = f"B-{label}"
                        for i in range(start_idx + 1, end_idx):
                            tags[i] = f"I-{label}"

            bio_data.append({
                "answers": tokens,
                "tags": tags
            })

    return bio_data


# Cargar datos del archivo JSON
input_file_path = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json'
output_file_path = 'bio_train.json'

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Transformar a formato BIO
    bio_tagged_data = transform_to_bio(dataset)

    # Guardar el resultado en un archivo JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(bio_tagged_data, output_file, indent=4, ensure_ascii=False)

    print(f"Transformación a formato BIO completada. Archivo guardado como '{output_file_path}'.")

except Exception as e:
    print(f"Error: {e}")

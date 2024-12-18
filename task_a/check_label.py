import json
import chardet

# Función para cargar el JSON con diferentes codificaciones
def load_json_with_encoding(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


# Usar esta función para cargar los datos desde un archivo JSON
file_path = '/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json'


def verify_spans(data):
    for entry in data:
        raw_text = entry.get("raw_text", "")
        labelled_spans = entry.get("labelled_answer_spans", {})

        #print(f"\nVerifying spans for URI: {entry['uri']}")

        for label, spans in labelled_spans.items():
            for span in spans:
                span_start, span_end = span["label_spans"]
                extracted_text = raw_text[span_start:span_end]

                #print(span_start, span_end)
                #if extracted_text != span["txt"]:
                #    print(f"  Mismatch in label '{label}':\n    Expected: '{span['txt']}'\n    Found:    '{extracted_text}'")
                print(extracted_text)
                


encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'windows-1252', 'ascii']


for encoding in encodings:
    try:
        print(f"  Using encoding: {encoding}")
        loaded_data = load_json_with_encoding(file_path, encoding)
        verify_spans(loaded_data)
    except Exception as e:
        print(f"  Error loading {file_path} with encoding {encoding}: {e}")

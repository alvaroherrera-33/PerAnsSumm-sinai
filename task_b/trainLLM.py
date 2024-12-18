import os
import torch
import pandas as pd
import json
from datetime import date
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer, is_xpu_available
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

os.environ["WANDB_DISABLED"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Solo usa la GPU 0

MODEL_PATH = '/mnt/beegfs/sinai-data/Llama-3.2-1B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

current_dir = os.path.realpath(os.path.dirname(__file__))

prompt = "Task Description: You are given a question \( Q \), a set of user answers \( A \), and a set of perspective categories: Information: Objective facts or explanations about the condition or procedure. Cause: Insights into what causes or could worsen the issue. Suggestion: Advice or recommendations to address the problem. Experience: Personal stories or shared experiences related to the problem. Question: Queries that seek additional clarification or recommendations. Your task is: 1. Span Identification: Locate specific spans within the answers \( A \) that correspond to one of these perspectives. 2. Classification: Assign the identified span to the correct perspective category. Input Example - Question (Q): 'I was just diagnosed with gallstones in my gall bladder. I really don’t want to have surgery and have been told that there are other ways to get rid of the stones. Suggestions?' - Answers (A): 1. 'Most gallstones are made of pure cholesterol. You might try a diet with low fat and very low saturated fats. I've had the surgery, and it really isn't a big deal. If you leave the gallstones there, they can get large enough to damage.' 2. 'Have you seen a gastroenterologist? They can do a minimally invasive procedure called an ERCP. I had the surgery myself about 10 years ago, and it really helped.' 3. 'The best remedy is surgery. I had surgery to have kidney stones removed. The surgery isn’t as bad as you think it may be.' Output Example - Information: 'Most gallstones are made of pure cholesterol.' Cause: 'If you leave the gallstones there, they can get large enough to damage.' Suggestion: 'You might try a diet with low fat and very low saturated fats.' Experience: 'I had the surgery myself about 10 years ago, and it really helped.' Question: 'Have you seen a gastroenterologist?' Key Considerations: Contextual understanding is crucial: Ensure that spans are precise and directly tied to the category. Medical knowledge will often be necessary to interpret and classify perspectives accurately. Output Format: For each span, return: 1. The text span. 2. The assigned perspective category."

print(prompt)
# Data generator for training with prompt
train_data = pd.read_json('/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json', orient = 'records')

# Divide el dataset en train y eval (80% - 20%)
#train_data, eval_data = train_test_split(train_data, test_size=0.2, random_state=42)
def prompt_generator_dataset():
    for index, example in train_data.iterrows():
        #abreviacion = example["abreviaturas"]
        labels = json.dumps(example["labelled_answer_spans"], ensure_ascii=False) 
        answers = example['answers']
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": answers},
            {"role": "assistant", "content": labels},
        ]

        messages = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=False)
        yield {
                "prompt": messages
            }
        
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
        
#####################
train_dataset = Dataset.from_generator(prompt_generator_dataset)
#eval_dataset = Dataset.from_generator(prompt_generator_dataset)

max_seq_length = 512
output_model_name = 'llama3_extract'
today = date.today()
print("TORCH SOPORTA BF16:", torch.cuda.is_bf16_supported())
training_args = TrainingArguments(
    output_dir=os.path.join(current_dir, f"results/checkpoints/{output_model_name}/"),
    logging_dir=os.path.join(current_dir, f"results/logs/{output_model_name}/"),
    logging_steps = 1,
    gradient_checkpointing=True, #Suele empeorar la velocidad de entrenamiento un 20%, pero es más eficiente en memoria
    optim = "adamw_bnb_8bit", #Optimizador que va más rápido para NVIDIA (8bytes al igual que adamw_torch o adamw_hf. Adafactor usa 4 bytes y adamw_bnb_8bit usa 2 bytes)
    #evaluation_strategy="no",
    save_strategy = "epoch",
    num_train_epochs= 100, #EarlyStopping callback will stop the training process when necessary
    #evaluation_strategy = "steps", #"epoch", #La evaluación la hacemos a posteriori con inferencia
    do_train=True,
    do_eval = False,
    #do_eval=True, #La evaluación la hacemos a posteriori con inferencia
    #gradient_accumulation_steps = 2, #Si el batch size es muy pequeño (<=4), evitamos que entrene de forma estocástica
    per_device_train_batch_size=8,
    learning_rate=5e-05,
    #num_train_epochs=num_train_epochs,
    #metric_for_best_model= 'train_loss',
    ddp_find_unused_parameters= False,
    # load_best_model_at_end = True,
    run_name = today.strftime("%b-%d-%Y") + output_model_name,
    seed=42,
)


qlora_config = LoraConfig(
    r=64, # Probar 4 si no entra
    lora_alpha=128, #32,
    lora_dropout=0.1,
    #use_gradient_checkpointing = True,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

torch_dtype = torch.float16
quant_storage_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    nb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=quant_storage_dtype,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
    torch_dtype=quant_storage_dtype,
    use_cache=False, 
)

base_model.config.pretraining_tp = 1

trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset, 
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="prompt",
    packing= False,
    max_seq_length=max_seq_length,
    compute_metrics=compute_metrics,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)

trainer.accelerator.print(f"{trainer.model}")
if trainer.accelerator.is_main_process:
    trainer.model.print_trainable_parameters()

print("Entrenamos")
trainer.train(resume_from_checkpoint = False)

#metrics = trainer.evaluate()
#with open(os.path.join(current_dir, "results", "metrics.json"), "w") as f:
 #   json.dump(metrics, f, indent=4)


if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.model.save_pretrained(os.path.join(current_dir, f"results/final_checkpoints/{output_model_name}-final/"))
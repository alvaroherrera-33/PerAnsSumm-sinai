
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load BERT base cased model and tokenizer
model = "/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/bert-large-cased"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model)

# Define new domain-specific tokens (example tokens)
new_tokens = ['O', 'B-INFORMATION', 'I-INFORMATION', 'B-CAUSE', 'I-CAUSE', 
              'B-EXPERIENCE', 'I-EXPERIENCE', 'B-SUGGESTION', 'I-SUGGESTION', 
              'B-QUESTION', 'I-QUESTION']
num_added_toks = tokenizer.add_tokens(new_tokens)
print('We have added', num_added_toks, 'tokens')

# Resize the token embeddings matrix to accommodate the new vocabulary size
model.resize_token_embeddings(len(tokenizer))

# Tokenize words with the enriched tokenizer
print(tokenizer.tokenize('COVID'))  # Output should be ['COVID']
print(tokenizer.tokenize('hospitalization'))  # Output should be ['hospitalization']

from transformers import AutoTokenizer, AutoModelForMaskedLM

# Example for adding new tokens to a Portuguese model
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Add new tokens to the tokenizer
print("[ BEFORE ] tokenizer vocab size:", len(tokenizer)) 
added_tokens = tokenizer.add_tokens(new_tokens)
print("[ AFTER ] tokenizer vocab size:", len(tokenizer))

# Resize the embeddings matrix of the model to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

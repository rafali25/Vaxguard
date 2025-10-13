import os
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pickle

def format_prompt(text):
    return "You are an expert misinformation detection assistant.\n\n"\
           "Your task is read the following text and determine which one of these categories best describes its author:\n\n"\
           "Religious Conspiracy Theorist\n"\
           "Misinformation Spreader\n"\
           "Anti Vaxxer\n"\
           "Fear Mongerer\n\n"\
           "Text:\n" + text + "\n"\
           "Respond with exactly one of these labels—and nothing else: "

responses = []
login("")
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
model = AutoModelForCausalLM.from_pretrained(
                'mistralai/Mistral-7B-v0.3',
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True
              )

@torch.no_grad()
def classify(text):
    prompt = format_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
             )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded.strip()
    responses.append(reply)

    if "Religious Conspiracy Theorist" in reply:
        return 0
    elif "Misinformation Spreader" in reply:
        return 1
    elif "Anti Vaxxer" in reply:
        return 2
    elif "Fear Mongerer" in reply:
        return 3
    else:
        return -1
base_folder = '/kaggle/input/vaxguard/'
csv_files =[['GPT-3.5/HPV/clean_HPV_texts_GPT-3.5_Anti-Vacciner.csv',
'GPT-3.5/HPV/clean_HPV_texts_GPT-3.5_fear mongerer.csv',
'GPT-3.5/HPV/clean_HPV_texts_GPT-3.5_Misinformation spreader.csv',
'GPT-3.5/HPV/clean_HPV_texts_GPT-3.5_religious Conspiracy theorist.csv'],
['GPT-4o/HPV/clean_HPV_texts_GPT-4o_Anti-Vacciner.csv',
'GPT-4o/HPV/clean_HPV_texts_GPT-4o_fear mongerer.csv',
'GPT-4o/HPV/clean_HPV_texts_GPT-4o_Misinformation spreader.csv',
'GPT-4o/HPV/clean_HPV_texts_GPT-4o_religious Conspiracy theorist.csv'],
['Llama 3/HPV/clean_HPV_texts_llama3_Anti-Vacciner.csv',
'Llama 3/HPV/clean_HPV_texts_llama3_Misinformation spreader.csv',
'Llama 3/HPV/clean_HPV_texts_llama_fear_monger.csv',
'Llama 3/HPV/clean_HPV_texts_llama_religious.csv'],
['PHI3/HPV/clean_HPV_texts_phi3_Anti-Vacciner.csv',
'PHI3/HPV/clean_HPV_texts_phi3_Misinformation spreader.csv',
'PHI3/HPV/clean_HPV_texts_Phi_fear.csv',
'PHI3/HPV/clean_HPV_texts_phi_religious_conspiracy.csv']]

csv_files = [[base_folder + file for file in csv] for csv in csv_files]

models = ['GPT-3.5', 'GPT-4o', 'Llama 3', 'Phi 3']
for i, csv in enumerate(csv_files):
    texts = []
    labels = []
    model = models[i]
    
    for j, file in enumerate(csv):
        df = pd.read_csv(file, encoding='latin1')['Misinformation']
        df.dropna(inplace=True)
        texts.extend(df.tolist())
        labels.extend([j] * len(df))

    preds = []
    for text in tqdm(texts, desc=f'model: {model}'):
        preds.append(classify(text))

    with open(f'response_HPV_{model}.pickle', 'wb') as file:
        pickle.dump(responses, file)
        responses = []
        
    with open(f'HPV_{model}.pickle', 'wb') as file:
        pickle.dump(texts, file)
        pickle.dump(labels, file)
        pickle.dump(preds, file)
        
    overall_acc = accuracy_score(labels, preds)
    overall_prec = precision_score(labels, preds, average="macro", zero_division=0)
    overall_rec = recall_score(labels, preds, average="macro", zero_division=0)
    overall_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f'{model}:')
    print(f'Accuracy: {overall_acc}')
    print(f'Precision: {overall_prec}')
    print(f'Recall: {overall_rec}')
    print(f'F1: {overall_f1}')


import os
import pandas as pd
import openai
from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
from tqdm import tqdm
import pickle  # for further processing on the outputs, if required

responses = []
openai.api_key = ""
client = openai.OpenAI(api_key=openai.api_key)

def format_prompt(text):
    return "You are an expert misinformation detection assistant.\n\n"\
           "Your task is read the following text and determine which one "\
           "of these categories best describes its author:\n\n"\
           "Religious Conspiracy Theorist\n"\
           "Misinformation Spreader\n"\
           "Anti Vaxxer\n"\
           "Fear Mongerer\n\n"\
           "Text:\n" + text + "\n"\
           "Respond with exactly one of these labels—and nothing else: "

def classify(text):
    prompt = format_prompt(text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        
        reply = response.choices[0].message.content.strip()
        responses.append(reply)
        reply = reply.lower()

        if "conspiracy" in reply:
            return 0
        elif "misinformation" in reply:
            return 1
        elif "vax" in reply:
            return 2
        elif "fear" in reply:
            return 3
        else:
            return -1
            
    except Exception as e:
        print(f"Error: {e}")
        return -1

def main():
    base_folder = ''
    files =[['GPT-4o/COVID-19/clean_COVID-19_texts_GPT-4o_religious Conspiracy theorist.csv',
             'GPT-4o/COVID-19/clean_COVID-19_texts_GPT-4o_Misinformation spreader.csv',
             'GPT-4o/COVID-19/clean_COVID-19_texts_GPT-4o_Anti-Vacciner.csv',
             'GPT-4o/COVID-19/clean_COVID-19_texts_GPT-4o_fear mongerer.csv'],
            ['Llama 3/COVID-19/clean_COVID-19_texts_llama_religious.csv',
             'Llama 3/COVID-19/clean_COVID-19_texts_llama3_Misinformation spreader.csv',
             'Llama 3/COVID-19/clean_COVID-19_texts_llama3_Anti-Vacciner.csv',
             'Llama 3/COVID-19/clean_COVID-19_texts_llama_fear_monger.csv'],
            ['Mistral/COVID-19/clean_COVID-19_texts_mistral_religious.csv',
             'Mistral/COVID-19/clean_COVID-19_texts_mistral_Misinformation spreader.csv',
             'Mistral/COVID-19/clean_COVID-19_texts_mistral_Anti-Vacciner.csv',
             'Mistral/COVID-19/clean_COVID-19_texts_mistral_fear.csv'],
            ['PHI3/COVID-19/clean_COVID-19_texts_phi_religious_conspiracy (2).csv',
             'PHI3/COVID-19/clean_COVID-19_texts_phi3_Misinformation spreader.csv',
             'PHI3/COVID-19/clean_COVID-19_texts_phi3_Anti-Vacciner.csv',
             'PHI3/COVID-19/clean_COVID-19_texts__phi3_fear.csv']]

    files = [[base_folder + csv_file for csv_file in model] for model in files]

    models = ['GPT-4o', 'Llama 3', 'Mistral', 'Phi 3']
    for i, model_files in enumerate(files):
        texts = []
        labels = []
        model = models[i]
    
        for j, file in enumerate(model_files):
            df = pd.read_csv(file, encoding='latin1')['Misinformation']
            df.dropna(inplace=True)
            texts.extend(df.tolist())
            labels.extend([j] * len(df))

        preds = []
        for text in tqdm(texts, desc=f'model: {model}'):
            preds.append(classify(text))

        with open(f'responses_COVID_{model}.pickle', 'wb') as file:
            pickle.dump(responses, file)
            responses = []
        
        with open(f'COVID_{model}.pickle', 'wb') as file:
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

if __name__ == '__main__':
    main()


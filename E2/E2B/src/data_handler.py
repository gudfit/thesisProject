# src/data_handler.py
from datasets import load_dataset
import re

def get_sentences_from_dataset(config: dict) -> list[str]:
    print(f"Loading dataset: {config['dataset_name']}, subset: {config['dataset_subset']}...")
    try:
        dataset = load_dataset(config['dataset_name'], config['dataset_subset'])
        
        split_name = config.get('test_split', 'validation') 
        if split_name not in dataset:
            raise KeyError(f"Split '{split_name}' not found in dataset. Available splits are: {list(dataset.keys())}")
            
        text_data = dataset[split_name]['text']
        
        all_sentences = []
        for paragraph in text_data:
            if not isinstance(paragraph, str):
                continue
            
            sentences = paragraph.split('\n')
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if clean_sentence and not re.match(r'^=.*=$', clean_sentence):
                    all_sentences.append(clean_sentence)
                    
        print(f"Found {len(all_sentences)} sentences in the '{split_name}' split.")
        return all_sentences

    except Exception as e:
        print(f"An error occurred in get_sentences_from_dataset: {e}")
        return [] 

import os

import pandas as pd
from tqdm.auto import tqdm
import torchaudio
from path import Path
from transformers import Wav2Vec2FeatureExtractor
from transformers import RobertaTokenizer

from .libri_speech import LibriSpeechDataset
from .cmu_mosi import CmuMosiDataset

def make_dataframe(dataPath):
    df = pd.DataFrame(columns=['file_path', 'text', 'speaker_id', 'chapter_id', 'utterance_id', 'duration'])

    i = 0
    for folder in tqdm(os.listdir(dataPath)):
        if os.path.isdir(dataPath / folder):
            folders = []
            for f in os.listdir(dataPath / folder):
                folders.append(f)
            if len(folders)>0:
                for fold in folders:
                    with open(dataPath / folder / fold / f'{folder}-{fold}.trans.txt', 'r') as fd:
                        data = fd.readlines()
                        for utterance_id, line in enumerate(data):
                            file_id = line.replace('\n', '').split(' ')[0] 
                            sentence = ' '.join(line.replace('\n', '').split(' ')[1:])
                            waveform, _ = torchaudio.load(dataPath / f'{folder}/{fold}/{file_id}.flac')
                            df.at[i] = [dataPath / f'{folder}/{fold}/{file_id}.flac', sentence,  folder, fold, utterance_id, waveform.shape[1] / 16000]
                            i+=1
    
    return df

def create_dataframes_librispeech(librispeech_dataset_path):
    """
    
    """
    train_dataset_path, dev_dataset_path, test_dataset_path = None, None, None 
    ## Check if path exists and contains train-clean-100, dev-clean and test-clean
    if os.path.exists(librispeech_dataset_path):
        librispeech_dataset_path = Path(librispeech_dataset_path)
        
        if os.path.exists(librispeech_dataset_path / 'train-clean-100'):
            train_dataset_path = librispeech_dataset_path / 'train-clean-100'
        
        if os.path.exists(librispeech_dataset_path / 'dev-clean'):
            dev_dataset_path = librispeech_dataset_path / 'dev-clean'
        
        if os.path.exists(librispeech_dataset_path / 'test-clean'):
            test_dataset_path = librispeech_dataset_path / 'test-clean'
    else:
        raise Exception(f"Dataset path {librispeech_dataset_path} doesn't exist")
    
    if (train_dataset_path is None) or (dev_dataset_path is None) or (test_dataset_path is None):
        raise Exception(f'Some of the dataset path train-clean-100, dev-clean, test-clean does\'nt exist') 
        
    ## Create dataframes
    train_df = create_dataframes_librispeech(librispeech_dataset_path / 'train-clean-100')
    dev_df = create_dataframes_librispeech(librispeech_dataset_path / 'dev-clean')
    test_df = create_dataframes_librispeech(librispeech_dataset_path / 'test-clean')
    
    ## Store the dataframes
    if not os.path.exists('data/'):
        os.mkdir('data/')
    
    train_df.to_csv('data/librispeech_train_df.csv')
    dev_df.to_csv('data/librispeech_dev_df.csv')
    test_df.to_csv('data/librispeech_test_df.csv')

def fetch_wav2vec2_feature_extractor():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

    return feature_extractor

def get_roberta_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    return tokenizer

def store_weights_wav2vec2_feature_extractor():
    pass

def store_weights_wav2vec2_feature_projection():
    pass

def fetch_datasets(config):
    train_df = pd.read_csv('data/librispeech_train_df.csv')
    val_df = pd.read_csv('data/librispeech_dev_df.csv')
    test_df = pd.read_csv('data/librispeech_test_df.csv')

    feature_extractor = fetch_wav2vec2_feature_extractor()
    tokenizer = get_roberta_tokenizer()

    train_dataset = LibriSpeechDataset(config, train_df, feature_extractor, tokenizer)
    val_dataset = LibriSpeechDataset(config, val_df, feature_extractor, tokenizer)
    test_dataset = LibriSpeechDataset(config, test_df, feature_extractor, tokenizer)

    return train_dataset, val_dataset, test_dataset

def fetch_mosi_datasets(config):
    train_df = pd.read_csv('data/mosi_train_df.csv')
    val_df = pd.read_csv('data/mosi_val_df.csv')
    test_df = pd.read_csv('data/mosi_test_df.csv')

    feature_extractor = fetch_wav2vec2_feature_extractor()
    tokenizer = get_roberta_tokenizer()

    train_dataset = CmuMosiDataset(config, train_df, feature_extractor, tokenizer)
    val_dataset = CmuMosiDataset(config, val_df, feature_extractor, tokenizer)
    test_dataset = CmuMosiDataset(config, test_df, feature_extractor, tokenizer)

    return train_dataset, val_dataset, test_dataset    
   
if __name__=="__main__":
    pass

import torchaudio
import torch

class CmuMosiDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, feature_extractor, tokenizer):
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw_speech_tensor, sampling_rate = torchaudio.load(self.df.iloc[idx]['audio_file_path'])
        
        raw_speech_tensor = raw_speech_tensor.reshape(-1,)
        
        audio_features_out = self.feature_extractor(raw_speech_tensor, sampling_rate=self.config['SAMPLING_RATE'], padding='max_length', max_length=self.config['MAX_AUDIO_LENGTH'], return_attention_mask=True, return_tensors="pt")
        
        tokenizer_out = self.tokenizer(self.df.iloc[idx]['text'],  return_attention_mask=True, return_tensors="pt")
        input_ids = tokenizer_out['input_ids']
        
        if (self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1])>0:
            words_attention_mask = torch.concat((tokenizer_out['attention_mask'][0], torch.zeros((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1]), dtype=torch.int64)))
            
            input_ids = torch.concat((input_ids[0],torch.full((self.config['MAX_TEXT_LENGTH'] - input_ids.shape[1],), self.tokenizer.pad_token_id, dtype=torch.int64)))

        else:
            input_ids = input_ids[0, :self.config['MAX_TEXT_LENGTH']]
            words_attention_mask = tokenizer_out['attention_mask'][0, :self.config['MAX_TEXT_LENGTH']]
        
        score = torch.tensor(self.df.iloc[idx]['score'], dtype=torch.float32)
        
        return audio_features_out['input_values'].reshape(-1,)[:self.config['MAX_AUDIO_LENGTH']], input_ids, words_attention_mask, score
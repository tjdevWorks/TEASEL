import torchaudio
import torch

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, config, df, feature_extractor, tokenizer):
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw_speech_tensor, sampling_rate = torchaudio.load(self.df.iloc[idx]['file_path'])
        
        raw_speech_tensor = raw_speech_tensor.reshape(-1,)
        
        audio_features_out = self.feature_extractor(raw_speech_tensor, sampling_rate=self.config['SAMPLING_RATE'], padding='max_length', max_length=self.config['MAX_AUDIO_LENGTH'], return_attention_mask=True, return_tensors="pt")
        
        tokenizer_out = self.tokenizer(self.df.iloc[idx]['text'],  return_attention_mask=True, return_tensors="pt")
        
        gt_input_ids = tokenizer_out['input_ids'].clone()
        
        input_ids = tokenizer_out['input_ids']
        
        masked_indices = ((torch.rand(gt_input_ids.shape[1]) < 0.15) * (gt_input_ids!=0) * (gt_input_ids!=2))[0].nonzero().flatten()
        
        for idx in masked_indices:
            input_ids[0, idx] = 50264
        
        if (self.config['MAX_TEXT_LENGTH'] - gt_input_ids.shape[1])>0:
            input_ids = torch.concat((input_ids[0],torch.full((self.config['MAX_TEXT_LENGTH'] - gt_input_ids.shape[1],), self.tokenizer.pad_token_id, dtype=torch.int64)))

            words_attention_mask = torch.concat((tokenizer_out['attention_mask'][0], torch.zeros((self.config['MAX_TEXT_LENGTH'] - gt_input_ids.shape[1]), dtype=torch.int64)))

            gt_input_ids = torch.concat((gt_input_ids[0], torch.ones((self.config['MAX_TEXT_LENGTH'] - gt_input_ids.shape[1]), dtype=torch.int64)))
        else:
            input_ids = input_ids[0, :self.config['MAX_TEXT_LENGTH']]
            words_attention_mask = tokenizer_out['attention_mask'][0, :self.config['MAX_TEXT_LENGTH']]
            gt_input_ids = gt_input_ids[0,:self.config['MAX_TEXT_LENGTH']]
        
        return audio_features_out['input_values'].reshape(-1,)[:self.config['MAX_AUDIO_LENGTH']], input_ids, words_attention_mask, gt_input_ids

if __name__=="__main__":
    pass
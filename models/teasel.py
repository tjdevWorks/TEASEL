import torch
import transformers
from transformers.modeling_outputs import MaskedLMOutput

from .laa import LightAttentiveAggregation
from .roberta import RobertaForMaskedLM_Teasel, RobertaForSequenceClassification

class TeaselPretrain(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        
        self.device = device
        
        self.wav_ft_model = transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureExtractor(transformers.Wav2Vec2Config())
        
        self.wav_ft_model.load_state_dict(torch.load('artificats/wav2vec2-base-960h-featureExtractor.pth')['feature_extractor_weights'])
        
        self.wav_fp_model = transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection(transformers.Wav2Vec2Config())
        
        self.wav_fp_model.load_state_dict(torch.load('artificats/wav2vec2-base-960h-featureProjection.pth')['feature_projection_weights'])
        
        self.laa = LightAttentiveAggregation(audio_output_size=768)
        
        self.roberta_config = transformers.RobertaConfig()

        self.roberta_config.max_position_embeddings = 514
        
        self.roberta_config.vocab_size = 50265
        self.roberta_config.type_vocab_size = 1
        
        self.roberta_maskedlm = RobertaForMaskedLM_Teasel.from_pretrained('roberta-base', config=self.roberta_config)
        
        #Freeze Layers
        for parameter in self.wav_ft_model.parameters():
            parameter.requires_grad = False
        
        for parameter in self.wav_fp_model.parameters():
            parameter.requires_grad = False
        
        for parameter in self.roberta_maskedlm.parameters():
            parameter.requires_grad = False
        
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_audio, masked_ids, attention_mask, gt_unmasked_ids=None):
        ## Wav2Vec2FeatureExtractor
        z = self.wav_ft_model(input_audio)
        
        z = z.transpose(1,2)
        
        ## Wav2Vec2FeatureProjection
        proj_output, _ = self.wav_fp_model(z)
        
        ## LAA - (ca output)
        ca = self.laa(proj_output)

        ## <Create the input to maskedlm model> even adjust the attention_mask
        attention_mask = torch.concat((torch.unsqueeze(attention_mask[:, 0],1), torch.ones((ca.shape[0],ca.shape[1]), device=self.device), attention_mask[:, 1:]), 1).type(torch.long)
        
        token_type_ids = torch.zeros(masked_ids.shape, dtype=torch.long, device=self.device)
        
        maskedlm_output = self.roberta_maskedlm(ca, masked_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        loss = None
        if gt_unmasked_ids is not None:
            loss = self.loss_fct(torch.concat((torch.unsqueeze(maskedlm_output.logits[:,0,:],1), maskedlm_output.logits[:,3:,:]), 1).view(-1, self.roberta_config.vocab_size), gt_unmasked_ids.view(-1))
        
        return MaskedLMOutput(loss = loss,
                              logits=maskedlm_output.logits,
                              hidden_states=maskedlm_output.hidden_states,
                              attentions=maskedlm_output.attentions)

class TeaselFineTuneMOSI(torch.nn.Module):
    def __init__(self, weights_file, device='cpu'):
        super().__init__()
        
        self.device = device
        
        self.wav_ft_model = transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureExtractor(transformers.Wav2Vec2Config())
        
        self.wav_ft_model.load_state_dict(torch.load('artificats/wav2vec2-base-960h-featureExtractor.pth')['feature_extractor_weights'])
        
        self.wav_fp_model = transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection(transformers.Wav2Vec2Config())
        
        self.wav_fp_model.load_state_dict(torch.load('artificats/wav2vec2-base-960h-featureProjection.pth')['feature_projection_weights'])
        
        self.laa = LightAttentiveAggregation(audio_output_size=768)
        
        self.roberta_config = transformers.RobertaConfig()

        self.roberta_config.max_position_embeddings = 514
        
        self.roberta_config.vocab_size = 50265
        self.roberta_config.type_vocab_size = 1
        self.roberta_config.num_labels = 1
        
        self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=self.roberta_config)
        
        #Freeze Layers
        for parameter in self.wav_ft_model.parameters():
            parameter.requires_grad = False
        
        for parameter in self.wav_fp_model.parameters():
            parameter.requires_grad = False
        
        for parameter in self.laa.bigru.parameters():
            parameter.requires_grad = False
        
        self.reload_weights(weights_file)
        
    def forward(self, input_audio, input_ids, attention_mask, score_tensor=None):
        ## Wav2Vec2FeatureExtractor
        z = self.wav_ft_model(input_audio)
        
        z = z.transpose(1,2)
        
        ## Wav2Vec2FeatureProjection
        proj_output, _ = self.wav_fp_model(z)
        
        ## LAA - (ca output)
        ca = self.laa(proj_output)

        ## <Create the input to maskedlm model> even adjust the attention_mask
        attention_mask = torch.concat((torch.unsqueeze(attention_mask[:, 0],1), torch.ones((ca.shape[0],ca.shape[1]), device=self.device), attention_mask[:, 1:]), 1).type(torch.long)
        
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=self.device)
        
        if score_tensor is not None:
            final_output = self.roberta_model(ca, input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=score_tensor)
        else:
            final_output = self.roberta_model(ca, input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        return final_output
    
    def reload_weights(self, weights_file):
        model_dict = torch.load(weights_file)
        print("Reloading Model Weights")
        self.load_state_dict(model_dict['model_state_dict'], strict=False)
        print("Model Weights Reloaded! :D")
        #assert torch.all(self.laa.bigru.weight_hh_l0 == model_dict['model_state_dict']['laa.bigru.weight_hh_l0'])

    
if __name__=="__main__":
    model = TeaselPretrain()

    audio_input = torch.rand((2, 480000), dtype=torch.float32)
    text_masked_input_ids =  torch.randint(0, 50000, (2, 75), dtype=torch.long)
    attention_mask =  torch.randint(0, 2, (2, 75), dtype=torch.long)

    print(f"Input Shapes:\nAudio_Input: {audio_input.shape}\nText_Mask_Input_Ids: {text_masked_input_ids.shape}\nAttention_Mask: {attention_mask.shape}")
    
    maskedlm_output = model(audio_input, text_masked_input_ids, attention_mask)
    
    print(maskedlm_output)
    print("Logits Shape: ", maskedlm_output.logits.shape)
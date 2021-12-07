import torch

class LightAttentiveAggregation(torch.nn.Module):
    def __init__(self, audio_output_size=768):
        super().__init__()
        ##BiGRU
        self.audio_output_size = audio_output_size 
        self.bigru = torch.nn.GRU(input_size=768, hidden_size=audio_output_size, num_layers=2, bias=True, batch_first=True, bidirectional=True)
        
        ##Aggregation / Attention Module check paper citation 40
        self.agg_1 = torch.nn.Linear(in_features=audio_output_size, out_features=audio_output_size, bias=True)
        
        self.agg_2 = torch.nn.Linear(in_features=audio_output_size, out_features=1, bias=True)
        
        self.layer_norm = torch.nn.LayerNorm((2, audio_output_size), )
    
    def forward(self, fp_out):
        gru_output_sequence, _ = self.bigru(fp_out)
        gru_output_sequence = gru_output_sequence.view(-1, gru_output_sequence.shape[1], 2, self.audio_output_size)
        p1_gru_output_sequence, p2_gru_output_sequence = gru_output_sequence[:,:,0,:], gru_output_sequence[:,:,1,:]
        u_1 = torch.sigmoid(self.agg_1(p1_gru_output_sequence))
        u_2 = torch.sigmoid(self.agg_1(p2_gru_output_sequence))
        alph_1 = torch.softmax(self.agg_2(u_1),1)
        alph_2 = torch.softmax(self.agg_2(u_2),1)
        ca_1 = torch.bmm(alph_1.transpose(1,2), p1_gru_output_sequence)
        ca_2 = torch.bmm(alph_2.transpose(1,2), p2_gru_output_sequence)
        ca = torch.concat((ca_1, ca_2),1)
        ca = self.layer_norm(ca)
        return ca


if __name__=="__main__":
    laa_model = LightAttentiveAggregation()

    proj_outs = torch.rand((1, 1499, 768), dtype=torch.float32)

    print("Input Shape to model: ", proj_outs.shape)
    
    ca = laa_model(proj_outs)
    
    print("Output shape of model: ", ca.shape)
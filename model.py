from transformers.models.roberta.modeling_roberta import *
from transformers import AutoTokenizer, RobertaModel
from loss import Rational_Tagging
# Code model baseline 
class ModelQA(nn.Module):
    def __init__(self, config):
        super(ModelQA, self).__init__()
        
#         self.number_label = config.number_labels
        
        self.model = RobertaModel.from_pretrained(config.model_name)
        # Đóng băng các thông số của text_encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.config = self.model.config
        # Use FC layer to get the start logits l_start and end logits_end
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, 2)
        
        self.tagging = Rational_Tagging(self.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        '''
        output: model will return hidden state, pooler,..
        
        qa_output: return (batch, row, colum) example (1, 8, 768)
        
        logits contain probability of an word is start position and end position
        
        example:
                    tensor([[[-0.1880, -0.0796],
                            [-0.2347, -0.1440],
                            [-0.2825, -0.1179],
                            [-0.3406, -0.1836],
                            [-0.3912,  0.0133],
                            [-0.1169, -0.3032],
                            [-0.3016, -0.1336],
                            [-0.1779, -0.0750]]], grad_fn=<ViewBackward0>) 
        
        '''
        output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        
        qa_ouputs = output[0]
        
        logits = self.qa_outputs(qa_ouputs)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        pt =  self.tagging(qa_ouputs)
        
        return pt, start_logits, end_logits
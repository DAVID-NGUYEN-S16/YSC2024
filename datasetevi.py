from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
class Data(Dataset):
    def __init__(self, df, tokenizer, max_len=400):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
#     def get_top_sentence(self, context, claim):
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        claim, context, label, ids = self.get_input_data(row)
#         claim = segmentation_token(claim)
#         context = select_sentance_text(claim=claim, context=context)
#         text = claim + '. ' + evidence
        
        encoding = self.tokenizer.encode_plus(
            claim,
            context,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        return {
#             'text': text,
            'id': ids,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }


    def labelencoder(self, text):
        if text=='NEI':
            return 0

        else:
            return 1

    def get_input_data(self, row):
        # Preprocessing: {remove icon, special character, lower}
        claim = row['claim']
        context = row['sentence']
        ids = row['id']
        label = row['label']
        
#         text = ' '.join(simple_preprocess(text))
#         label = self.labelencoder(row['verdict'])

        return str(claim), str(context), label, ids
def load_data(config, tokenizer):
    train_data = pd.read_csv(config.data.train).reset_index()
    test_data = pd.read_csv(config.data.test).reset_index()
    train_df = train_data
    test_df = test_data
    
    train_dataset = Data(train_df, tokenizer, max_len=256)
    test_dataset = Data(test_df, tokenizer, max_len=256)

    return train_dataset, test_dataset
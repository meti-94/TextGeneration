"""
Training loop 
"""
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from data import PersonaChatDataset_v3, PersonaChatPytorchDataset_v3

from model import SimpleEncoderDecoder

import numpy as np

SPECIAL_TOKENS = ["<bos>", "<eos>", "<persona>", "<speaker1>", "<speaker2>", "<pad>"]

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
						 'additional_special_tokens': ['<speaker1>', '<speaker2>', '<persona>']}


tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
# train = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=False)

test = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=True)

train_data = test.model_food[:10]
train_dataloader = DataLoader(PersonaChatPytorchDataset_v3(train_data), batch_size=1)
model = SimpleEncoderDecoder(tokenizer)
for batch in train_dataloader:

	print(type(batch), len(batch), batch.keys(), batch['input_ids'])
	# batch = {key:np.array(value) for key, value in batch.items()}
	# print(model(**batch))


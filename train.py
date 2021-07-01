"""
Training loop 
"""
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from data import PersonaChatDataset_v3, PersonaChatPytorchDataset_v3

from model import SimpleEncoderDecoder, greedy_generate, save, load

import numpy as np

import sys

import torch

from tqdm import tqdm

from transformers import AdamW

from sklearn.model_selection import train_test_split

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from transformers import logging as tlogging
# tlogging.set_verbosity_error()

####################################################
############## Preparing Tokenizer and Model #######
####################################################

SPECIAL_TOKENS = ["<bos>", "<eos>", "<persona>", "<speaker1>", "<speaker2>", "<pad>"]

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
						 'additional_special_tokens': ['<speaker1>', '<speaker2>', '<persona>']}

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
model = SimpleEncoderDecoder(tokenizer)

####################################################
######### Preparing Train, Test & Validation #######
####################################################

train = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=True)
test = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=True)
# train_dataset = PersonaChatPytorchDataset_v3(train)
train, valid = train_test_split(train.model_food, test_size=0.1, random_state=44)
# train, valid = torch.utils.data.random_split(train, [len(train)-10000, 10000])
trainset = train
validset = valid
testset = test.model_food[:10]

train_dataloader = DataLoader(PersonaChatPytorchDataset_v3(trainset), batch_size=4, shuffle=True)
valid_dataloader = DataLoader(PersonaChatPytorchDataset_v3(validset), batch_size=2, shuffle=False)
test_dataloader = DataLoader(PersonaChatPytorchDataset_v3(testset), batch_size=1, shuffle=False)

####################################################
######### Training Loop  ###########################
####################################################

epochs = 10
freezeemb = False
 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = []
for paramname, param in model.named_parameters():
	if paramname.startswith("bert.embeddings.word_embeddings"):
		if not freezeemb:
			params.append(param)
	else:
		params.append(param)
optimizer = AdamW(params, lr=1e-3)

model.to(device)


step = 0
for epoch in range(epochs):
	
	train_losses = []
	train_custom_losses = []
	for _, batch in enumerate(tqdm(train_dataloader, desc=f"Train Epoch Number {epoch+1}")):
		model.train()
		model.zero_grad()
		device_batch = {key:value.to(device) for key, value in batch.items()}
		model_output = model(**device_batch, return_dict=True)
		loss = model_output['loss']
		closs = model_output['past_key_values']
		train_losses.append(loss)
		train_custom_losses.append(closs)
		loss.backward()
		optimizer.step()
		step+=1
		if step%10==0:
			model.eval()
			losses = []
			custom_losses = []
			for _, batch in enumerate(tqdm(valid_dataloader, desc=f"Eval Epoch Number {epoch+1}")):
				with torch.no_grad():
					device_batch = {key:value.to(device) for key, value in batch.items()}
					model_output = model(**device_batch, return_dict=True)
					loss = model_output['loss']
					closs = model_output['past_key_values']
					losses.append(loss)
					custom_losses.append(closs)
			logging.info(f'Epoch number: {epoch+1} Eval Loss is equal: {sum(losses)/len(losses)}')
			logging.info(f'Epoch number: {epoch+1} Eval cLoss is equal: {sum(custom_losses)/len(custom_losses)}')
			logging.info(model.automatic_metrics(validset))

		if step%1000==0:
			save(model)
	logging.info(f'Epoch number: {epoch+1} Train Loss is equal: {sum(train_losses)/len(train_losses)}') 
	logging.info(f'Epoch number: {epoch+1} Train Loss is equal: {sum(train_custom_losses)/len(train_custom_losses)}') 
	
	
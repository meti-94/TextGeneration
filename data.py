

"""
Wrapper class to load and prepare data
"""

import json

from tqdm import tqdm 

from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader

import sys 

import torch

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<persona>", "<speaker1>", "<speaker2>", "<pad>"]

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
						 'additional_special_tokens': ['<speaker1>', '<speaker2>', '<persona>']}

encoder_max_length=512
decoder_max_length=128


######################################

class PersonaChatDataset_v3():
	def __init__(self, json_file_path, tokenizer, test=False):
		self.is_test = test
		try:
			data_type = 'Test' if self.is_test else 'Train'

			with open('./data/PersonaChatDataset_{}_v3.json'.format(data_type), 'r') as fin:
				logging.info('There is a pre-processed food file (Data Type = {}) in Data Directory ... '.format(data_type))
				self.model_food = json.load(fin)
				return None
		except:
			logging.info('There is no pre-processed model food file to load\nSo let\'s CREATE one')
		with open(json_file_path) as json_file:
			data_dict = json.load(json_file)
		self.tokenizer = tokenizer
		self.is_test = test
		self.data_dict = data_dict
		self.samples = None
		self.bertified = None
		self.model_food = None
		self.run()
		return None

	def data_to_samples(self):
		samples=[]
		for dialogue in (self.data_dict['train'] if self.is_test==False else self.data_dict['valid']):
			original_persona = dialogue['personality']
	
			for item in dialogue['utterances']:
				original_persona = [original_persona[-1]] + original_persona[:-1]
				history = item['history']
				response = item['candidates'][-1]
				samples.append({
					  'persona':original_persona,
					  'history':history,
					  'response':response
					})
		self.samples = samples
	
	def bertifier(self):
		bertified_data = []
		for item in self.samples:
			persona = ' <persona> '.join(item['persona'])
			persona = '<bos> ' + persona
			history = ''
			speakers = [" <speaker1> ", " <speaker2> "]
			speaker = 0
			for hst in item['history'][::-1]:
				history = speakers[speaker] + hst + history
				speaker = 1 - speaker
			response = '<bos> ' + item['response'] + ' <eos>'
			bertified_data.append({
			  'persona':persona.replace('  ', ' '),
			  'history':history.replace('  ', ' '),
			  'input': persona.replace('  ', ' ')+' '+history.replace('  ', ' '), 
			  'response':response.replace('  ', ' ')
				})
		self.bertified = bertified_data

	def to_model_food(self):
		model_food = []
		pbar = tqdm(self.bertified)
		desc_type = 'Testing' if self.is_test else 'Training'
		pbar.set_description('Working on {} Samples ...'.format(desc_type))
		for item in pbar:
			inputs = self.tokenizer(item['input'], add_special_tokens=True, max_length=encoder_max_length, truncation=True, padding="max_length")
			outputs = self.tokenizer(item['response'], add_special_tokens=True, max_length=decoder_max_length, truncation=True, padding="max_length")
			assert len(inputs.input_ids)==512
			assert len(inputs.attention_mask)==512
			assert len(outputs.input_ids)==128
			assert len(outputs.attention_mask)==128
			
			_item = {}
			_item["input_ids"] = inputs.input_ids
			_item["attention_mask"] = inputs.attention_mask
			_item["decoder_input_ids"] = outputs.input_ids
			_item["decoder_attention_mask"] = outputs.attention_mask
			_item["labels"] = outputs.input_ids.copy()
			model_food.append(_item)
		self.model_food = model_food 

	def save(self):
		data_type = 'Test' if self.is_test else 'Train'
		with open('./data/PersonaChatDataset_{}_v3.json'.format(data_type), 'w') as fout:
			json.dump(self.model_food, fout)

	def run(self):
		self.data_to_samples()
		self.bertifier()
		self.to_model_food()
		self.save()
	
	

class PersonaChatPytorchDataset_v3(Dataset):
	'''
	  Convert Data to proper Tensor dataset
	'''
	def __init__(self, samples):
		self.samples = samples
		self.n_samples = len(self.samples)

	def __getitem__(self, index):
		# returns specific item
		sample = self.samples[index]
		return {key:torch.tensor(value) for key, value in sample.items()}

	def __len__(self):
		return self.n_samples


if __name__=='__main__':
	tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
	tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
	train = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=False)
	train = PersonaChatDataset_v3('./data/personachat_self_original.json', tokenizer, test=True)
	
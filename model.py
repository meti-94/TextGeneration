"""
Training loop 
"""
from transformers import BertModel, BertConfig, AutoModelWithLMHead

import torch

class SimpleEncoderDecoder(torch.nn.Module):
	def __init__(self, tokenizer):
		super().__init__()
		self.encoder = BertModel.from_pretrained('prajjwal1/bert-tiny')
		self.decoder = AutoModelWithLMHead.from_pretrained('prajjwal1/bert-tiny')
		self.tokenizer = tokenizer

		self.encoder.resize_token_embeddings(len(self.tokenizer))

	def forward(
			self,
			input_ids=None,
			attention_mask=None,
			decoder_input_ids=None,
			decoder_attention_mask=None,
			labels=None
		):
		# Encode
		encoder_outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

		encoder_hidden_states = encoder_outputs[0]
		# Decode
		decoder_outputs = self.decoder(
				input_ids=decoder_input_ids,
				attention_mask=decoder_attention_mask,
				encoder_hidden_states=encoder_hidden_states,
				labels=labels
			)
		return decoder_outputs.loss, decoder_outputs.logits
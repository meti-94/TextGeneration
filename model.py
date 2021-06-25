"""
Training loop 
"""
from transformers import BertModel, BertConfig, AutoModelForCausalLM

from transformers.modeling_outputs import Seq2SeqLMOutput

import torch

from collections import namedtuple

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SimpleEncoderDecoder(torch.nn.Module):
	def __init__(self, tokenizer, **kw):
		super().__init__(**kw)
		self.encoder = BertModel.from_pretrained('prajjwal1/bert-tiny')
		self.decoder = AutoModelForCausalLM.from_pretrained('prajjwal1/bert-tiny', )
		self.decoder.config.is_decoder=True
		self.tokenizer = tokenizer

		self.encoder.resize_token_embeddings(len(self.tokenizer))
		self.decoder.resize_token_embeddings(len(self.tokenizer))

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		decoder_input_ids=None,
		decoder_attention_mask=None,
		encoder_outputs=None,
		past_key_values=None,
		inputs_embeds=None,
		decoder_inputs_embeds=None,
		labels=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		**kwargs,
		):
		

		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
				inputs_embeds=inputs_embeds,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

		encoder_hidden_states = encoder_outputs[0]

		# Decode
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=attention_mask,
			inputs_embeds=decoder_inputs_embeds,
			labels=labels,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			# use_cache=use_cache,
			# past_key_values=past_key_values,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return Seq2SeqLMOutput(
			loss=decoder_outputs.loss,
			logits=decoder_outputs.logits,
			# past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			# cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)



	def forward_old(
			self,
			input_ids=None,
			attention_mask=None,
			decoder_input_ids=None,
			decoder_attention_mask=None,
			encoder_outputs=None,
			labels=None,
			return_dict=True
		):
		# Encode
		if encoder_outputs is None:
			encoder_outputs = self.encoder(
					input_ids=input_ids,
					attention_mask=attention_mask,
					return_dict=return_dict
				)

		encoder_hidden_states = encoder_outputs[0]
		# Decode
		decoder_outputs = self.decoder(
				input_ids=decoder_input_ids,
				attention_mask=decoder_attention_mask,
				encoder_hidden_states=encoder_hidden_states,
				labels=labels,
				return_dict=return_dict
			)
		if not return_dict:
			return decoder_outputs.loss, decoder_outputs.logits
		
		return Seq2SeqLMOutput(
			loss=decoder_outputs.loss,
			logits=decoder_outputs.logits,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			encoder_last_hidden_state=encoder_outputs,
		)
	
def greedy_generate(
		input_ids=None,
		device=None,
		model=None,
		tokenizer=None,
		):

	encoderoutput = namedtuple('encoderoutput' , 'last_hidden_state hidden_states attentions')
	decoder_input_ids = tokenizer('<bos>', add_special_tokens=False, return_tensors="pt").input_ids
	decoder_input_ids = decoder_input_ids.to(device)
	outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
	encoded_sequence = encoderoutput(outputs.encoder_last_hidden_state, outputs.encoder_hidden_states, outputs.encoder_attentions)
	lm_logits = outputs.logits
	next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
	decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
	while decoder_input_ids.size()[-1] <= 126:
		lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
		next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
		decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

	logging.info(f"Generated Text: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}")

def save(model, save_path='./models/bert_encoder_decoder.pt'):
		torch.save(model, save_path)
		logging.info(f'Model is saved in {save_path}')

def load(save_path='./models/bert_encoder_decoder.pt'):
	# logging.info(f'Model is saved in {save_path}')
	model = torch.load(save_path)
	logging.info(f'Model is loaded from {save_path}')
	return model 
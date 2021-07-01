"""
Training loop 
"""
from transformers import BertModel, Seq2SeqTrainingArguments, AutoModel, Seq2SeqTrainer, BertConfig, EncoderDecoderModel, AutoModelForCausalLM, EncoderDecoderConfig

from transformers.modeling_outputs import Seq2SeqLMOutput

import torch

from collections import namedtuple

import datasets

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def model_init(model):
    return model


rouge = datasets.load_metric("rouge")
def compute_metrics(pred, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


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

	

	def convert_to_huggingface(self):
		self.encoder.save_pretrained('./tmp_encoder')
		self.decoder.save_pretrained('./tmp_decoder')
		encoder_decoder_config = EncoderDecoderConfig.from_pretrained('./models/checkpoint-1500')

		encoder = AutoModel.from_pretrained('./tmp_encoder')

		decoder = AutoModelForCausalLM.from_pretrained('./tmp_decoder', add_cross_attention=True)

		huggingface_model = EncoderDecoderModel(config=encoder_decoder_config, encoder=encoder, decoder=decoder)

		return huggingface_model 

	
	# def model_init():
	#     return model

	def automatic_metrics(self, dataset):
		huggingface_model = self.convert_to_huggingface()
		huggingface_model.config.decoder_start_token_id = self.tokenizer.cls_token_id
		huggingface_model.config.eos_token_id = self.tokenizer.sep_token_id
		huggingface_model.config.pad_token_id = self.tokenizer.pad_token_id
		huggingface_model.config.vocab_size = huggingface_model.config.encoder.vocab_size
		huggingface_model.config.add_cross_attention=True
		huggingface_model.config.no_repeat_ngram_size = 3
		huggingface_model.config.early_stopping = True
		huggingface_model.config.length_penalty = 2.0
		huggingface_model.config.num_beams = 4
		util_args = Seq2SeqTrainingArguments(
			predict_with_generate=True,
			output_dir='./tmp'
			)
		util = Seq2SeqTrainer(
			args=util_args,
			model=huggingface_model,
    		compute_metrics=lambda pred:compute_metrics(pred, self.tokenizer),
    		eval_dataset=dataset,
			tokenizer=self.tokenizer,
			)
		return util.evaluate()



	
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
	model = torch.load(save_path, map_location=torch.device('cpu'))
	logging.info(f'Model is loaded from {save_path}')
	return model 
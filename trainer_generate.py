from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import BertModel, BertTokenizer
from collections import namedtuple

import torch
from model import SimpleEncoderDecoder, load

SPECIAL_TOKENS = ["<bos>", "<eos>", "<persona>", "<speaker1>", "<speaker2>", "<pad>"]

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
						 'additional_special_tokens': ['<speaker1>', '<speaker2>', '<persona>']}


tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)


encoder_decoder_config = EncoderDecoderConfig.from_pretrained('./models/checkpoint-1200')
model = EncoderDecoderModel.from_pretrained('./models/checkpoint-1200', config=encoder_decoder_config)
model.get_encoder().resize_token_embeddings(len(tokenizer))
model.get_decoder().resize_token_embeddings(len(tokenizer))
print(type(model.get_encoder()), type(model.get_decoder()))
# model = SimpleEncoderDecoder(tokenizer)
# model = load()
# model.to('cpu')

# create ids of encoded input vectors
input_ids = tokenizer("I want to buy a car", return_tensors="pt").input_ids

# create BOS token
decoder_input_ids = tokenizer("<bos>", add_special_tokens=False, return_tensors="pt").input_ids

# print(model.config.decoder_start_token_id)

# assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"

# STEP 1

# pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
# encoder = model.get_encoder()

encoderoutput = namedtuple('encoderoutput' , 'last_hidden_state hidden_states attentions')
# print(type(encoder))
outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

# get encoded sequence
encoded_sequence = encoderoutput(outputs.encoder_last_hidden_state, outputs.encoder_hidden_states, outputs.encoder_attentions)
print(type(encoded_sequence[0]))
# get logits
lm_logits = outputs.logits

# sample last token with highest prob
next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)

# concat
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

# STEP 2

# reuse encoded_inputs and pass BOS + "Ich" to decoder to second logit
# lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits

lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits


# sample last token with highest prob again
next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)

# concat again
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

# STEP 3
lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

# let's see what we have generated so far!
print(f"Generated so far: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}")

# This can be written in a loop as well.
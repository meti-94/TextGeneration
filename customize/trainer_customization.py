from torch import torch
from torch import nn
from transformers.trainer_utils import PredictionOutput
from transformers import Seq2SeqTrainer
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
import random 
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = 128,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        (loss, logits, labels) = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        random_subset = random.sample(range(0, len(inputs['labels'])), 10)
        for item in random_subset:
            persona_history_input = self.tokenizer.batch_decode([inputs['input_ids'][item]])
            real_output = self.tokenizer.batch_decode([inputs['labels'][item]])
            model_output = self.tokenizer.batch_decode([logits[item]])
            print("History & Persona & Input: ", persona_history_input[0].replace("<pad>", ""))
            print("Input: ", real_output[0].replace("<pad>", ""))
            print("Output: ", model_output[0].replace("<pad>", ""))
        return loss, logits, labels
        
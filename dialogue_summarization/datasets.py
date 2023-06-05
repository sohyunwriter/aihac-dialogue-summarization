import json
from typing import Any, List

from torch.utils.data import Dataset
from tqdm import tqdm
import re
import random
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import BatchEncoding, PreTrainedTokenizerBase
import math
doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{2,}')
number_pattern = re.compile('[0-9]')
punctuation_pattern = re.compile('[,\.\?\!]')
symbol_pattern = re.compile('[()\[\]\{\}`]')
hangle_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]')
alphabet_pattern = re.compile('[a-zA-Z]')

hangle_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
hangle_number_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9]')
text_filter = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,\.\?\!\"\'-()\[\]\{\}]')

def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    sent = doublespace_pattern.sub(' ', sent)
    return sent.strip()


def sen2encode(tokenizer, strategy, sentence, max_length):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids[0]
    if strategy == "cut":
        if input_ids.size(-1) > max_length:
            input_ids = input_ids[:max_length]
        return input_ids

    elif strategy == "drop":
        if input_ids.size(-1) > max_length:
            return []

# def collate_fn(self, batch):
#     input_ids = [batch[i]["input_ids"] for i in range(len(batch))]
#     labels = [batch[i]["labels"] for i in range(len(batch))]

#     encoder = self.tokenizer.pad(
#         {"input_ids": input_ids},
#         padding="max_length",
#         max_length=self.config.max_length,  # 모델 뺌
#     )

#     decoder = self.tokenizer.pad(
#         {"input_ids": labels},
#         padding="max_length",
#         max_length=self.config.max_length, # 모델 뺌
#     )

#     return {
#         "input_ids": encoder["input_ids"],
#         "attention_mask": encoder["attention_mask"],
#         "labels": decoder["input_ids"],
#     }


 

def text_infil(tokenizer, 
                    all_inputs: torch.Tensor,
                    special_tokens_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def one_text_infil(tokenizer, inputs, special_tokens_mask):
        ## input : A <sep> B <sep>
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * mlm_probability))

        if num_to_mask == 0:
            return inputs, labels
        #print("num_to_mask", num_to_mask)
        #print("label shape", labels.shape)
        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask.bool()] = tokenizer.mask_token_id
        labels[~mask.bool()] = -100

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs, fill_value=tokenizer.pad_token_id)
        #new_inputs = torch.full_like(inputs, fill_value=False)
        #new_inputs = inputs
        for i, example in enumerate(torch.split(inputs, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        return new_inputs, labels


    #print(inputs)
    ans = []
    ans_new_inputs = []
    ans_labels = []
    for i in all_inputs:
        new_inputs, labels = one_text_infil(tokenizer, torch.unsqueeze(i, 0), special_tokens_mask)
        #print("new_inputs", new_inputs, labels)
        #print(new_inputs[0].shape, new_inputs)
        if new_inputs[0].shape[0] >= 300:
            new_inputs = new_inputs[:300]
        #ans.append({"input_ids": new_inputs[0], "labels": labels[0]})
        ans_new_inputs.append(new_inputs[0])
        ans_labels.append(labels[0])
    #print("ans_new_inputs", ans_new_inputs, ans_labels)

    return ans_new_inputs, ans_labels


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorForTextInfilling:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        ## input : A <sep> B <sep>
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch

    def mask_tokens(self,
                    inputs: torch.Tensor,
                    special_tokens_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ## input : A <sep> B <sep>
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * self.mlm_probability))

        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask.bool()] = self.tokenizer.mask_token_id
        labels[~mask.bool()] = -100

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(torch.split(inputs, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        return new_inputs, labels



class SummarizationDataset(Dataset):
    def __init__(
        self,
        datafile: str, 
        separator: str,
        meta_sep: str,
        tokenizer: Any,
        max_length: int,
        strategy: str,
        is_test = 0,
        is_val = 0,
    ) -> None:
        self.datasets = []
        self.is_test = is_test

        assert strategy in [
            "cut",
            "drop",
        ], "param `strategy` must be on of ['cut', 'drop']"


        if not self.is_test:
            for sample in tqdm(datafile, leave=True):
                header = sample["header"]
                dialogueID = header["dialogueInfo"]["dialogueID"]

                body = sample["body"]
                summary = body["summary"]
                
                dialogue = [u["utterance"] for u in body["dialogue"]]
                dialogue = separator.join(dialogue)

                # final = ""
                # for i, u in enumerate(sample["body"]["dialogue"]):
                #     if i == 0:
                #         final += meta_sep
                #         final += sample["header"]["dialogueInfo"]["topic"]
                #         final += meta_sep
                #         final += u["utterance"]
                #         continue

                #     elif (sample["body"]["dialogue"][i-1]["turnID"] == u["turnID"]):
                #         final += " "
                #         final += u["utterance"]
                #     else:
                #         final += separator
                #         final += u["utterance"]
                
                ## turnid 같은 문장 모으기 -> final
                final = []
                temp = " "
                for i, u in enumerate(sample["body"]["dialogue"]):
                    if i == 0:
                        temp += u["utterance"]
                        continue 
                    elif (sample["body"]["dialogue"][i-1]["turnID"] == u["turnID"]):
                        temp += " "
                        temp += u["utterance"]
                    else:
                        final.append(temp)
                        temp = " "
                        temp += u["utterance"]
                final.append(temp)

                for i, v in enumerate(final):
                    final[i] = repeat_normalize(v, num_repeats=2)  # 전처리

                # random(final)

                # ## text infill
                # data_collator = DataCollatorForTextInfilling(tokenizer)
                # if len(sen2encode(tokenizer, strategy, final, max_length)) > 0:
                #     input_ids = sen2encode(tokenizer, strategy, final, max_length)
                # else:
                #     continue

                ## input, label
                final = separator.join(final)

                if len(sen2encode(tokenizer, strategy, final, max_length)) > 0:
                    input_ids = sen2encode(tokenizer, strategy, final, max_length)
                    labels = sen2encode(tokenizer, strategy, final, max_length)
                else:
                    continue
                
                self.datasets.append({"input_ids": input_ids, "labels": labels, "dialogueID": dialogueID})

                # if not is_val:
                #     if len(sen2encode(tokenizer, strategy, dialogue, max_length)) > 0:
                #         input_ids = sen2encode(tokenizer, strategy, dialogue, max_length)
                #     else:
                #         continue

                #     self.datasets.append({"input_ids": input_ids, "labels": labels, "dialogueID": dialogueID})

            self.datasets = sorted(
                self.datasets,
                key=lambda k: k["input_ids"].size(-1),
                reverse=True,
            )

        else:
            for sample in tqdm(datafile, leave=True):
                header = sample["header"]
                dialogueID = header["dialogueInfo"]["dialogueID"]

                body = sample["body"]
            
            
                dialogue = [u["utterance"] for u in body["dialogue"]]
                dialogue = separator.join(dialogue)

                final = ""
                for i, u in enumerate(sample["body"]["dialogue"]):
                    if i == 0:
                        final += meta_sep
                        final += sample["header"]["dialogueInfo"]["topic"]
                        final += meta_sep
                        final += u["utterance"]
                        continue

                    elif (sample["body"]["dialogue"][i-1]["turnID"] == u["turnID"]):
                        final += " "
                        final += u["utterance"]
                    else:
                        final += separator
                        final += u["utterance"]
                final = repeat_normalize(final, num_repeats=2)  # 전처리

                
                #input_ids = tokenizer(final, return_tensors="pt",max_length=max_length,padding="max_length").input_ids[0]
                
                if len(sen2encode(tokenizer, strategy, final, max_length)) > 0:
                    input_ids = sen2encode(tokenizer, strategy, final, max_length)
                else:
                    continue

                # if strategy == "cut":
                #     if input_ids.size(-1) > max_length:
                #         input_ids = input_ids[:max_length]

                # elif strategy == "drop":
                #     if input_ids.size(-1) > max_length:
                #         continue
                    
                self.datasets.append({"input_ids": input_ids, "dialogueID": dialogueID})
            
            self.datasets = sorted(
                self.datasets,
                key=lambda k: k["input_ids"].size(-1),
                reverse=True,
            )

    def __getitem__(self, index):

        if not self.is_test:
            return {
                "input_ids": self.datasets[index]["input_ids"],
                "labels": self.datasets[index]["labels"],
            }

        else:
            return{
                "input_ids": self.datasets[index]["input_ids"],
                "dialogueID": self.datasets[index]["dialogueID"]
            }


    def __len__(self) -> int:
        return len(self.datasets)

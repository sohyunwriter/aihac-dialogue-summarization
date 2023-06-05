#encoding=utf-8

from transformers import (
    BartForConditionalGeneration, BartTokenizer, BartForCausalLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
  )

import torch
from torch.utils.data import random_split


# ## Initiating model and trainer for training
from transformers import BartModel, BartConfig
from transformers import BartTokenizerFast


configuration = BartConfig(
    vocab_size=52000,
    max_position_embeddings=258,
    d_model=256,
    encoder_layers=3,
    decoder_layers=3,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    decoder_ffn_dim=1024,
    encoder_ffn_dim=1024,
)
model = BartForCausalLM(configuration)
tokenizer = BartTokenizerFast.from_pretrained("./dic", max_len=256, additional_special_tokens=['[CH]', '[OTHER]', '[VAR]', '[NUM]'])




# splitting dataset into train, validation
split = 0.2
train_dataset, eval_dataset = random_split(data, lengths=[int((1-split)*len(data))+1, int(split*len(data))])


# defining collator functioon for preparing batches on the fly ..
def data_collator(features:list):
   inputs = [f["seq2seq"]["input"] for f in features]
   batch = tokenizer.prepare_seq2seq_batch(src_texts=inputs, max_length=256, padding='max_length')
   batch["labels"] = batch["input_ids"].copy()
   for k in batch:
        batch[k] = torch.tensor(batch[k])
   return batch


batch_out = data_collator(eval_dataset)
print(batch_out)
print(batch_out['input_ids'].shape,batch_out['labels'].shape,batch_out['attention_mask'].shape)


# defining training related arguments
args = Seq2SeqTrainingArguments(output_dir="clm-checkpoints",
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        learning_rate=5e-5,
                        num_train_epochs=1,
                        logging_dir="./logs")


# defining trainer using 🤗
trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset)


# ## Training time
trainer.train()
# It will take hours to train this model on this dataset


# lets save model
trainer.evaluate(eval_dataset=eval_dataset)
trainer.save_model("clm-checkpoints")
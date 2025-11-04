import argparse
import os
import math

import torch
from datasets import load_from_disk
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import evaluate


def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # replace -100 in the labels as we can't decode them
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu expects list[str]
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="models/data/chichewa_en")
    parser.add_argument("--model-name", type=str, default="Helsinki-NLP/opus-mt-ny-en")
    parser.add_argument("--output-dir", type=str, default="models/chi_en_finetuned")
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_from_disk(args.data_dir)
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    # enable gradient checkpointing to save memory
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    tokenized_train = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_val = dataset["validation"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # choose fp16 if running on CUDA and supported
    fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size * 2,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        predict_with_generate=True,
        fp16=fp16,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=True,
        push_to_hub=False,
    )
    # compute metrics wrapper
    def compute_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_wrapper,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

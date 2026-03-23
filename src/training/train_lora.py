#!/usr/bin/env python3
"""LoRA SFT fine-tuning script for Qwen2.5-14B-Instruct on credit agent data."""

import argparse
import json
import os
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChatSFTDataset(Dataset):
    """Multi-turn chat dataset that masks non-assistant tokens."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        paths = data_path.split(",")
        for p in paths:
            p = p.strip()
            with open(p) as f:
                for line in f:
                    obj = json.loads(line)
                    if "messages" in obj:
                        self.samples.append(obj["messages"])
        logger.info(f"Loaded {len(self.samples)} samples from {paths}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        # Use Qwen's chat template to build the full text
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Tokenize full conversation
        full = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)

        # Build labels: mask everything except assistant turns
        labels = input_ids.clone()
        # Tokenize prefix up to each assistant response to find boundaries
        labels[:] = -100  # mask all first

        # Re-build incrementally to find assistant spans
        prefix = ""
        for i, msg in enumerate(messages):
            # Build text up to and including this message
            partial = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            if msg["role"] == "assistant":
                # Previous prefix length in tokens
                prev_tokens = self.tokenizer(
                    prefix, truncation=True, max_length=self.max_length
                )["input_ids"]
                cur_tokens = self.tokenizer(
                    partial, truncation=True, max_length=self.max_length
                )["input_ids"]
                start = len(prev_tokens)
                end = len(cur_tokens)
                if end <= len(labels):
                    labels[start:end] = input_ids[start:end]
            prefix = partial

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_path: str, lora_config: LoraConfig):
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="data/base_models/Qwen2.5-14B-Instruct")
    p.add_argument("--data_path", type=str, default="outputs/sft_data_selected/train.jsonl",
                    help="Comma-separated JSONL paths")
    p.add_argument("--val_data_path", type=str, default="outputs/sft_data_selected/val.jsonl",
                    help="Validation JSONL path (optional)")
    p.add_argument("--output_dir", type=str, default="outputs/qwen2.5_14b_lora")
    p.add_argument("--run_name", type=str, default=None,
                    help="Experiment name. Output goes to output_dir/run_name/. Auto-generates timestamp if not set.")
    p.add_argument("--max_length", type=int, default=2048, help="Max token length per sample (data range: 485-2048)")
    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=-1, help="Set >0 for sanity check")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)
    # Merge
    p.add_argument("--merge_and_save", action="store_true",
                    help="Merge LoRA weights into base model and save")
    p.add_argument("--lora_checkpoint", type=str, default=None,
                    help="Path to LoRA checkpoint for merging")
    return p.parse_args()


def merge_lora(args):
    """Merge LoRA adapter into base model and save."""
    ckpt = args.lora_checkpoint or args.output_dir
    logger.info(f"Merging LoRA from {ckpt} into {args.model_path}")
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ckpt)
    merged = model.merge_and_unload()
    out = os.path.join(args.output_dir, "merged")
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info(f"Merged model saved to {out}")


def main():
    args = parse_args()

    if args.merge_and_save:
        merge_lora(args)
        return

    # Auto-generate run name with timestamp if not specified
    if args.run_name is None:
        from datetime import datetime
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Experiment: {args.run_name}, output: {args.output_dir}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    model, tokenizer = load_model_and_tokenizer(args.model_path, lora_config)
    train_dataset = ChatSFTDataset(args.data_path, tokenizer, args.max_length)
    val_dataset = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        val_dataset = ChatSFTDataset(args.val_data_path, tokenizer, args.max_length)

    eval_strategy = "steps" if val_dataset else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=args.gradient_checkpointing,
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt"
        ),
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

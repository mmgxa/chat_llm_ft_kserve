from pathlib import Path

import typer
from rich import print
from typing_extensions import Annotated


import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset


app = typer.Typer(pretty_exceptions_enable=False)


def format_instruction(sample):
    output_text = f"### Instruction:\n{sample['instruction']}\n\n"
    output_text += (
        f"### Context:\n{sample['context']}\n\n" if len(sample["context"]) > 0 else ""
    )
    output_text += f"### Response:\n{sample['response']}\n"
    return output_text


@app.command(help="Fine Tune LLM with LoRA and SFTTrainer")
def supertune(
    base_model: Annotated[
        str,
        typer.Option(
            help="Base Model to tune on",
        ),
    ] = "Deci/DeciLM-7B",
    sft_dataset: Annotated[
        str,
        typer.Option(
            help="Dataset on which to be fine-tuned",
        ),
    ] = "databricks/databricks-dolly-15k",
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            help="Number of update steps to accumulate before performing a backward pass",
        ),
    ] = 2,
    learning_rate: Annotated[
        float,
        typer.Option(help="Initial learning rate for training, after warmup period"),
    ] = 2.5e-5,
    lora_rank: Annotated[
        int, typer.Option(help="The dimension of the LoRA update matrices")
    ] = 64,
    output_dir: Annotated[
        str,
        typer.Option(
            help="The output directory to store the logs, checkpoints and final lora model weights",
        ),
    ] = "deci-7b-dolly-qlora",
):
    dolly_dataset = load_dataset(sft_dataset, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        use_cache=False,
        attn_implementation="flash_attention_2",  # not on T4
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    output_dir = Path(output_dir)
    logging_dir = Path(output_dir, "logs")
    logging_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # True makes slower
        num_train_epochs=1,
        learning_rate=learning_rate,
        logging_steps=50,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",
        logging_dir=logging_dir,
        save_strategy="steps",
        save_steps=50,  # Save checkpoints every 50 steps
        report_to="tensorboard",
    )

    max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        train_dataset=dolly_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=train_args,
        neftune_noise_alpha=5,
    )

    trainer.train()

    print(f"training complete!")

    trainer.save_model()


if __name__ == "__main__":
    app()

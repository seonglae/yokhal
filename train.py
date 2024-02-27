from huggingface_hub import login
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import SFTTrainer


login(
  token="", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

# Set up TPU device.
model_id = "google/gemma-2b"

# Load the pretrained model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Load the dataset and format it for training.
data = load_dataset("kor_hate", split="train")
max_seq_length = 1024

# Finally, set up the trainer and train the model.
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=3,  # This is actually the global batch size for SPMD.
        num_train_epochs=15,
        max_steps=-1,
        output_dir="./hannam",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last = False,  # Required for SPMD.
    ),
    peft_config=lora_config,
    dataset_text_field="comments",
    max_seq_length=max_seq_length,
    packing=True,
)

trainer.train()
model.save_pretrained('hannam-2b')
model.base_model.save_pretrained('hannam-2b')
tokenizer.save_pretrained('hannam-2b')
tokenizer.save_vocabulary('hannam-2b')
model.push_to_hub('seonglae/hannam-2b')
tokenizer.push_to_hub('seonglae/hannam-2b')
model.base_model.push_to_hub('seonglae/hannam-2b')

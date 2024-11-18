from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
import bitsandbytes as bnb
from src.config.config import *

processor = ViTImageProcessor.from_pretrained(MODEL)

# Show running device
print("Device:", DEVICE)
base_model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)

lora_config = LoraConfig(init_lora_weights="gaussian")
Q_lora_config = LoraConfig(init_lora_weights="gaussian", target_modules="all-linear")


def get_lora_config(type_: str) -> LoraConfig:
    if type_ == "lora":
        return get_peft_model(base_model, lora_config), None

    elif type_ == "Q_lora":
        return get_peft_model(base_model, Q_lora_config), None

    elif type_ == "lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        return get_peft_model(base_model, lora_config), optimizer

    elif type_ == "Q_lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, Q_lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        return get_peft_model(base_model, Q_lora_config), optimizer

    else:
        raise ValueError(f"Invalid type: {type_}")

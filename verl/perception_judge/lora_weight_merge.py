import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

lora_adapter_ckpt = 'sungnyun/Flex-VL-32B-LoRA'

# Load base model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_adapter_ckpt)

merged_model = model.merge_and_unload()

save_directory = "Flex-VL-32B-Instruct"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

merged_model.save_pretrained(save_directory, safe_serialization=True)
processor.save_pretrained(save_directory)
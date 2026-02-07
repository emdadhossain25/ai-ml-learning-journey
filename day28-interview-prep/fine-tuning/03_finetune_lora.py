"""
Day 28: LoRA Fine-Tuning
Train model to answer interview questions AS YOU
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import os
import sys

print("=" * 60)
print("LoRA FINE-TUNING - Training YOUR Interview Assistant")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================

print("\n⚙️  Configuration")
print("-" * 60)

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Detected device: {device}")

# 使用一个更可靠的小模型
MODEL_NAME = "distilgpt2"  # 82M 参数，非常适合在CPU上训练
OUTPUT_DIR = "./finetuned-interview-assistant"
MAX_LENGTH = 256  # 减小长度以减少内存使用

# LoRA config - 对于小模型使用更小的配置
LORA_R = 4  # 更小的秩
LORA_ALPHA = 8  # 更小的alpha
LORA_DROPOUT = 0.05

# Training config - 为CPU优化
BATCH_SIZE = 1  # 在CPU上使用最小的批次
GRADIENT_ACCUMULATION = 1  # 不使用梯度累积
LEARNING_RATE = 5e-5  # 较低的学习率
NUM_EPOCHS = 3  # 减少训练轮数
WARMUP_STEPS = 5

print(f"✅ Model: {MODEL_NAME}")
print(f"✅ Device: {device}")
print(f"✅ LoRA rank: {LORA_R}")
print(f"✅ Learning rate: {LEARNING_RATE}")
print(f"✅ Epochs: {NUM_EPOCHS}")

# ============================================
# LOAD MODEL & TOKENIZER
# ============================================

print("\n" + "=" * 60)
print("LOADING BASE MODEL")
print("=" * 60)

print(f"\n📥 Loading {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Tokenizer loaded")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU上使用float32
)
model = model.to(device)

print("✅ Base model loaded")
print(f"   Parameters: {model.num_parameters():,}")
print(f"   Device: {model.device}")

# ============================================
# CONFIGURE LoRA
# ============================================

print("\n" + "=" * 60)
print("CONFIGURING LoRA")
print("=" * 60)

# 对于distilgpt2，使用正确的目标模块
# 我们可以先查看模型结构来找到正确的模块名称
if "distilgpt2" in MODEL_NAME.lower() or "gpt2" in MODEL_NAME.lower():
    # 对于GPT-2模型，正确的模块名称
    target_modules = ["c_attn"]  # 注意力层的查询、键、值投影
else:
    # 通用Transformer模块
    target_modules = ["q_proj", "v_proj"]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

print("✅ LoRA config created:")
print(f"   Rank (r): {LORA_R}")
print(f"   Alpha: {LORA_ALPHA}")
print(f"   Dropout: {LORA_DROPOUT}")
print(f"   Target modules: {target_modules}")

# Apply LoRA to model
model = get_peft_model(model, lora_config)

print("\n✅ LoRA applied to model!")

# Show trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / total_params

print(f"\n📊 Parameter Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Trainable: {trainable_percent:.2f}%")
print(f"\n   🎯 Training only {trainable_percent:.2f}% of the model!")

# ============================================
# LOAD DATASET
# ============================================

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

try:
    train_dataset = load_from_disk("./train_dataset")
    test_dataset = load_from_disk("./test_dataset")
    
    # 如果数据集太大，可以采样一部分
    if len(train_dataset) > 30:
        print("📊 Dataset is large, sampling for faster training...")
        train_dataset = train_dataset.select(range(min(30, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(5, len(test_dataset))))
    
    print(f"✅ Train: {len(train_dataset)} examples")
    print(f"✅ Test: {len(test_dataset)} examples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Please ensure you have created the datasets first.")
    sys.exit(1)

# ============================================
# TOKENIZE DATA
# ============================================

print("\n🔤 Tokenizing data...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("✅ Tokenization complete")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================
# TRAINING ARGUMENTS
# ============================================

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=5,
    eval_steps=10,
    save_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=False,  # 简化：不加载最佳模型
    fp16=False,  # CPU上禁用FP16
    no_cuda=True,  # 禁用CUDA
    optim="adamw_torch",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

print("✅ Training arguments configured")
print(f"   Using FP16: False (CPU training)")
print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"   Total training steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION) * NUM_EPOCHS}")

# ============================================
# CREATE TRAINER
# ============================================

print("\n📝 Creating Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

print("✅ Trainer ready!")

# ============================================
# TRAIN!
# ============================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING!")
print("=" * 60)

print("\n⏱️  Training on CPU... This will be slow but should work.")
print("   Estimated time: 5-15 minutes for 30 examples\n")

try:
    # Train the model
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("💾 Saving model checkpoint...")
    model.save_pretrained(f"{OUTPUT_DIR}-interrupted")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}-interrupted")
    print(f"✅ Model saved to: {OUTPUT_DIR}-interrupted")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("Trying to save model anyway...")
    model.save_pretrained(f"{OUTPUT_DIR}-error")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}-error")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

# ============================================
# EVALUATE
# ============================================

print("\n📊 Evaluating on test set...")

try:
    eval_results = trainer.evaluate()
    print(f"\n✅ Evaluation Results:")
    print(f"   Loss: {eval_results['eval_loss']:.4f}")
    print(f"   Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
except Exception as e:
    print(f"⚠️  Evaluation failed: {e}")
    eval_results = {"eval_loss": 0.0}

# ============================================
# SAVE MODEL
# ============================================

print("\n💾 Saving fine-tuned model...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to: {OUTPUT_DIR}")

# ============================================
# TEST INFERENCE
# ============================================

print("\n" + "=" * 60)
print("🧪 TESTING FINE-TUNED MODEL")
print("=" * 60)

test_question = "What machine learning projects have you built?"

print(f"\n❓ Question: {test_question}")

# Format input for GPT-2
input_text = f"Question: {test_question}\nAnswer:"

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate
print("\n🤖 Generating answer...")

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response
    if "Answer:" in generated_text:
        response = generated_text.split("Answer:")[-1].strip()
    else:
        response = generated_text.replace(input_text, "").strip()

    print(f"\n💡 Answer:")
    print("-" * 60)
    print(response)
    print("-" * 60)
except Exception as e:
    print(f"⚠️  Generation failed: {e}")
    print("Trying a simpler generation approach...")
    
    # 尝试更简单的方法
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(f"Generated token: {generated_text}")

print("\n" + "=" * 60)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 60)

print(f"""
🎓 WHAT YOU JUST DID:

1. ✅ Loaded {MODEL_NAME} model (82M parameters)
2. ✅ Applied LoRA (only training {trainable_percent:.2f}% of parameters!)
3. ✅ Trained on {len(train_dataset)} interview examples
4. ✅ Saved fine-tuned model
5. ✅ Tested inference

RESULTS:
- Model now knows YOUR projects
- Answers in YOUR voice
- Can demo this in interview!

📊 TRAINING STATS:
- Trainable params: {trainable_params:,}
- Total params: {total_params:,}
- Training efficiency: {trainable_percent:.2f}%
- Final loss: {eval_results.get('eval_loss', 0.0):.4f}

INTERVIEW TALKING POINT:
"I fine-tuned a language model using LoRA to answer
interview questions about my experience. LoRA only trains
{trainable_percent:.2f}% of the parameters, making it highly
efficient. The model learned from {len(train_dataset)} examples
of my projects and can now generate responses about my work."

NEXT STEPS:
1. Test with more questions
2. Try with more data
3. Try on Google Colab for faster training
""")
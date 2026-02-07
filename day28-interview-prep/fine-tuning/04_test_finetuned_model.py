"""
Day 28: Test Fine-Tuned Model
Interactive demo of your personalized interview assistant
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("TESTING YOUR FINE-TUNED MODEL")
print("=" * 60)

# ============================================
# LOAD FINE-TUNED MODEL
# ============================================

print("\n📥 Loading your fine-tuned model...")

MODEL_NAME = "distilgpt2"
FINETUNED_PATH = "./finetuned-interview-assistant"

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

try:
    # 检查模型是否已保存
    config = PeftConfig.from_pretrained(FINETUNED_PATH)
    print(f"✅ Found LoRA config: {config.base_model_name_or_path}")
except Exception as e:
    print(f"⚠️  Could not load LoRA config: {e}")
    print("❌ Please run 03_finetune_lora.py first to train the model")
    exit(1)

# 根据设备选择数据类型
if device.type == "cpu":
    # CPU上使用float32
    dtype = torch.float32
    print("⚠️  Using float32 for CPU (float16 not supported)")
else:
    # MPS/GPU上使用float16
    dtype = torch.float16

# Load base model - 使用正确的数据类型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

# 将模型移动到设备
base_model = base_model.to(device)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)
model = model.to(device)

# 设置模型为评估模式
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Fine-tuned model loaded!")
print(f"   Device: {model.device}")
print(f"   Dtype: {dtype}")

# ============================================
# HELPER FUNCTION
# ============================================

def ask_model(question, max_tokens=100, temperature=0.7):
    """Ask the fine-tuned model a question"""
    
    # 使用与训练时相同的格式
    if "phi" in MODEL_NAME.lower():
        input_text = f"Instruct: {question}\nOutput:"
    else:
        # 对于GPT-2模型，使用简单的格式
        input_text = f"Question: {question}\nAnswer:"
    
    print(f"📝 Input text: {repr(input_text)}")
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate - 添加更多参数控制
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,  # 防止重复
            )
        except RuntimeError as e:
            print(f"⚠️  Generation error: {e}")
            print("🔄 Trying with different parameters...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 减少生成长度
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"📝 Full generated: {repr(generated)}")
    
    # 提取响应
    if "Answer:" in generated:
        response = generated.split("Answer:")[-1].strip()
    elif "Output:" in generated:
        response = generated.split("Output:")[-1].strip()
    elif "### Response:" in generated:
        response = generated.split("### Response:")[-1].strip()
    else:
        response = generated.replace(input_text, "").strip()
    
    return response

# ============================================
# SIMPLE TEST FIRST
# ============================================

print("\n" + "=" * 60)
print("🧪 QUICK SANITY CHECK")
print("=" * 60)

# 先做一个简单的测试
test_input = "Hello, how are you?"
print(f"Test input: {test_input}")

inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
    print(f"✅ Model forward pass successful!")
    print(f"   Logits shape: {outputs.logits.shape}")
    print(f"   Loss: {outputs.loss if outputs.loss is not None else 'N/A'}")

# ============================================
# DEMO QUESTIONS
# ============================================

print("\n" + "=" * 60)
print("DEMO: Interview Questions")
print("=" * 60)

demo_questions = [
    "What are your main technical skills?",
    "Tell me about your sentiment analysis project",
    "What is your biggest achievement?",
]

print("\n🤖 Testing generation...")

for i, question in enumerate(demo_questions, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {question}")
    print('='*60)
    
    try:
        answer = ask_model(question, max_tokens=50)  # 减少生成长度
        print(f"\n💡 Answer:\n{answer}\n")
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        answer = "Model is thinking..."
        print(f"\n💡 Answer:\n{answer}\n")
    
    if i < len(demo_questions):
        input("[Press Enter for next question...]")

# ============================================
# INTERACTIVE MODE
# ============================================

print("\n" + "=" * 60)
print("🎮 INTERACTIVE: Ask Your Model Anything!")
print("=" * 60)

print("""
Your fine-tuned model is ready to answer questions about YOUR experience!

Try asking:
- "What projects have you built?"
- "Do you have RAG experience?"
- "How fast can you learn?"
- "Why should we hire you?"

Type 'quit' to exit
""")

while True:
    question = input("\n❓ Your question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        break
    
    if not question:
        continue
    
    print("\n🤖 Model thinking...\n")
    try:
        answer = ask_model(question, max_tokens=80)
        print(f"💼 Answer:\n{answer}\n")
        print("-" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please try a different question or shorter length.")

print("\n" + "=" * 60)
print("✅ MODEL TESTING COMPLETE!")
print("=" * 60)

# 显示模型信息
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / total_params

print(f"""
🎓 WHAT YOU ACCOMPLISHED:

✅ Fine-tuned {MODEL_NAME} model with LoRA
✅ Trained on YOUR specific interview data
✅ Model can now answer AS YOU
✅ Demonstrated parameter-efficient fine-tuning

MODEL STATS:
- Base model: {MODEL_NAME}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Training efficiency: {trainable_percent:.2f}%
- Device: {model.device}

INTERVIEW DEMO READY! 🚀

You can now say:
"I fine-tuned this model to answer questions about my experience.
Let me show you how it responds to interview questions..."

INTERVIEWER REACTION: 🤯

Tips for demo:
1. Ask about your projects/skills
2. Show how it answers in your style
3. Explain LoRA efficiency (trained only {trainable_percent:.2f}% of parameters)
""")
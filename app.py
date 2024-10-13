import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import time
from datetime import datetime, timedelta

# Use a smaller model for better performance
model_name = "Qwen/Qwen-VL-Chat"

@st.cache_resource
def load_model():
    if os.path.exists("quantized_model.pth"):
        model = torch.load("quantized_model.pth")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor

def quantize_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model, "quantized_model.pth")

# Load model (it will be quantized if not already done)
try:
    model, processor = load_model()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

# Rest of your code remains the same
def extract_product_info(image):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": "Identify the following information from the product image: Brand Name, quantity, date of packaging/manufacturing, expiry date/use by date (if best before in months is given, calculate it by adding to manufacture date), and price. Present the information in a structured format."
                    }
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False
            )

        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        st.error(f"Error in extracting product info: {str(e)}")
        return None

# The rest of your code (parse_product_info, analyze_product, and Streamlit UI) remains the same

st.title("Product Information Extractor")
st.write("Upload an image of a product to extract information such as brand name, quantity, manufacturing date, expiry date, and price.")

uploaded_file = st.file_uploader("Choose a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Product Image", use_column_width=True)
    if st.button("Analyze Product"):
        analyze_product(image)

st.markdown("---")
st.write("Note: This model's accuracy may vary depending on the clarity and content of the uploaded image.")

import streamlit as st
import os
import sys

def check_dependencies():
    required_packages = [
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "einops",
        "transformers_stream_generator",
        "numpy",
        "tqdm"
    ]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        st.error(f"The following required packages are missing: {', '.join(missing_packages)}")
        st.error("Please make sure all required packages are installed. You may need to update your requirements.txt file.")
        st.stop()

check_dependencies()

import torch
import torchvision
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import einops
import transformers_stream_generator
from PIL import Image
import time
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

# Use a smaller model for better performance
model_name = "Qwen/Qwen-VL-Chat"

@st.cache_resource
def load_model():
    try:
        if os.path.exists("quantized_model.pth"):
            model = torch.load("quantized_model.pth")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.stop()

def quantize_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model, "quantized_model.pth")

# Load model (it will be quantized if not already done)
model, processor = load_model()

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

def parse_product_info(raw_output):
    # Your existing parse_product_info function here
    pass

def analyze_product(image):
    # Your existing analyze_product function here
    pass

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

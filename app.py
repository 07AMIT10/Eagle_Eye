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
        model = AutoModelForCausalLM.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def quantize_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
    try:
        info = {
            "Brand Name": "",
            "Quantity": "",
            "Manufacturing Date": "",
            "Expiry Date": "",
            "Price": ""
        }
        
        lines = raw_output.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in info:
                    info[key] = value
        
        if "months" in info["Expiry Date"].lower() and info["Manufacturing Date"]:
            try:
                mfg_date = datetime.strptime(info["Manufacturing Date"], "%d/%m/%Y")
                months = int(info["Expiry Date"].split()[0])
                exp_date = mfg_date + timedelta(days=30*months)
                info["Expiry Date"] = exp_date.strftime("%d/%m/%Y")
            except:
                st.warning("Unable to calculate expiry date from 'Best Before' information.")

        return info
    except Exception as e:
        st.error(f"Error in parsing product info: {str(e)}")
        return None

def analyze_product(image):
    start_time = time.time()
    
    raw_result = extract_product_info(image)
    if raw_result:
        parsed_result = parse_product_info(raw_result)
        
        end_time = time.time()
        inference_time = end_time - start_time

        if parsed_result:
            st.subheader("Extracted Product Information:")
            for key, value in parsed_result.items():
                st.write(f"{key}: {value}")
            st.write(f"\nInference Time: {inference_time:.2f} seconds")
            
            st.subheader("Raw Model Output:")
            st.text(raw_result)
        else:
            st.error("Failed to parse product information.")
    else:
        st.error("Failed to extract product information from the image.")

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

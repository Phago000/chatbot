import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Bank Transaction Image Analyzer",
    page_icon="🏦",
    layout="wide"
)

import google.generativeai as genai
import PIL.Image
import io
import json
import base64
import fitz  # PyMuPDF
import numpy as np

# Configuration
MULTIMODAL_MODEL = 'gemini-2.0-flash-exp'

# Show title and description
st.title("🏦 Bank Transaction Image Analyzer")
st.write(
    "This app analyzes bank transaction images using Google's Gemini Vision model. "
    "To use this app, you need to provide a Google API key, which you can get from the "
    "[Google AI Studio](https://makersuite.google.com/app/apikey)."
)

# Ask user for their Google API key
google_api_key = st.text_input("Google API Key", type="password")
if not google_api_key:
    st.info("Please add your Google API key to continue.", icon="🔑")
else:
    # Initialize Gemini with user's API key
    genai.configure(api_key=google_api_key)

    def convert_pdf_to_images(pdf_file):
        """Converts PDF file to list of PIL Images."""
        pdf_bytes = pdf_file.read()
        images = []
        
        # Open PDF from memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get the page's image at a higher DPI (300) for better quality
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Convert PyMuPDF pixmap to PIL Image
            img_bytes = pix.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_bytes))
            
            images.append(img)
        
        doc.close()
        return images

    def analyze_image(image, result_container):
        """Analyzes the image using Gemini."""
        model = genai.GenerativeModel(model_name=MULTIMODAL_MODEL)
        
        # Prepare image
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Enhanced prompt with account number instructions
        prompt = """
        Analyze this image and extract the following transaction details:
        
        - Transaction Date
        - Client Bank Account Name (Payer):
          * If an account number is visible, include it in the format: "Account Name - Account Number"
          * If no account number is visible, just return the account name
        - Transaction Currency
        - Transaction Amount
        - WMC Bank Account Name (Payee):
          * If an account number is visible, include it in the format: "Account Name - Account Number"
          * If no account number is visible, just return the account name
        
        Important:
        1. For account names, always check for and include account numbers if they are present
        2. Account numbers might appear near the account names or in separate fields
        3. The format should be "Account Name - XXXX" where XXXX is the account number
        
        Return the results in JSON format:
        {
          "transactionDate": "YYYY-MM-DD or NOT FOUND",
          "clientBankAccountName": "Name - Account Number (if available) or just Name or NOT FOUND",
          "transactionCurrency": "Currency code or NOT FOUND",
          "transactionAmount": "Amount or NOT FOUND",
          "wmcBankAccountName": "Name - Account Number (if available) or just Name or NOT FOUND"
        }
        """
        
        # Prepare content and get response
        contents = [prompt, image]
        response = model.generate_content(contents)
        
        # Update the result container with the response
        try:
            result_dict = json.loads(response.text)
            with result_container:
                st.subheader("Analysis Results:")
                for key, value in result_dict.items():
                    # Format the display of the results
                    display_key = {
                        "transactionDate": "Transaction Date",
                        "clientBankAccountName": "Client Bank Account Name (Payer)",
                        "transactionCurrency": "Transaction Currency",
                        "transactionAmount": "Transaction Amount",
                        "wmcBankAccountName": "WMC Bank Account Name (Payee)"
                    }.get(key, key)
                    st.write(f"**{display_key}:** {value}")
        except json.JSONDecodeError:
            with result_container:
                st.write(response.text)

    # Main app functionality
    uploaded_file = st.file_uploader(
        "Upload a bank transaction document",
        type=["png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                images = convert_pdf_to_images(uploaded_file)
                st.write(f"Converted PDF to {len(images)} images")
                
                # Analyze each page
                for i, image in enumerate(images):
                    st.write(f"### Page {i+1}")
                    st.image(image, caption=f"Page {i+1}")
                    
                    # Create a container for results
                    result_container = st.empty()
                    
                    if st.button(f"Analyze Page {i+1}"):
                        with st.spinner("Analyzing image..."):
                            try:
                                analyze_image(image, result_container)
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
            else:
                # Handle regular image files
                image = PIL.Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
                
                # Create a container for results
                result_container = st.empty()
                
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        try:
                            analyze_image(image, result_container)
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

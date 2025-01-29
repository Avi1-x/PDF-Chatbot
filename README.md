# PDF Chatbot

## Overview

This project is a chatbot powered by the Llama 3.2 model running locally. It allows you to upload a PDF and interact with the content by asking questions and getting relevant responses.

## Features

- **Local Llama Model**: The chatbot runs the Llama 3.2 model locally.
- **Interactive Chat**: Ask questions and receive context-aware responses based on the content of the PDF.
- **Message History**: Tracks conversation history for improved context in responses.
- **Web Interface**: Simple Streamlit interface for user interaction.

## Requirements

Before you can use the chatbot, ensure that you:

1. **Have the Llama model installed locally**.
   - You can download the Llama 3.2 model either from [Meta](https://www.llama.com/llama-downloads/) or [Hugging Face](https://huggingface.co/).
   - If you download the model from Meta, you'll need to convert the model to Hugging Face format. Follow the instructions in the [Transformers library](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).
   
2. **Convert the model from Hugging Face format to GGUF** using [Llama.cpp](https://github.com/ggerganov/llama.cpp) for quantization.
   - Full instructions on converting and quantizing the model can be found [here](https://github.com/ggerganov/llama.cpp/discussions/2948).

3. **Update the file paths** in the code to point to your local model files.

Once everything is set up, you’re ready to run the chatbot.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
```

### Step 2: Set up the environment

1. **Create a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

1. **Navigate to the directory where the Py file is stored

2. **Run the following command**

```bash
streamlit run pdf_chatbot.py
```

## Screenshots of the app.





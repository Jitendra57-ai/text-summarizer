# Text Summarizer

A web-based text summarization application built with a fine-tuned **T5-small** model. The application takes a block of text or dialogue and generates a concise summary through a simple browser interface.

## What This Project Does

The project fine-tunes a pretrained T5-small model for dialogue summarization using the **SAMSum dataset**. The trained model is then integrated into a lightweight web application where users can enter text and generate a summary.

The project covers the complete workflow from **model training and preprocessing to deployment through an API and web interface**.

## Features

- Fine-tuned T5-small model for text summarization
- Dialogue preprocessing and tokenization
- Beam search for summary generation
- FastAPI backend for model inference
- Simple HTML/CSS web interface
- Local model inference without requiring an external API

## Tech Stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **T5-small**
- **FastAPI**
- **Uvicorn**
- **HTML / CSS**

## Model Training

The model was fine-tuned on the **SAMSum dialogue summarization dataset**.

During preprocessing, the dialogue text is cleaned and tokenized before being passed to the T5 model. The trained model is then saved and loaded by the FastAPI application for inference.

## Project Structure

```text
text-summarizer/
│
├── app.py                  # FastAPI application and model inference
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── templates/
│   └── index.html          # Web interface
│
└── static/
    └── style.css           # Application styling
```

> The trained model files and training checkpoints are excluded from this repository because of their size.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/Jitendra57-ai/text-summarizer.git
cd text-summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
uvicorn app:app --reload
```

### 4. Open the application

Open the local address shown by Uvicorn in your browser.

## Example

**Input:**

> John: Did you finish the project report?  
> Sarah: Not yet. I completed the introduction and methodology sections.  
> John: When do you think you can finish it?  
> Sarah: I should be able to finish the remaining sections by Friday.

**Generated Summary:**

> Sarah completed the introduction and methodology sections and plans to finish the remaining report sections by Friday.

## Project Goal

This project was built as a practical application of **Natural Language Processing and Transformer-based models**, with the goal of taking a fine-tuned machine learning model beyond the training environment and integrating it into a usable application.

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


#initialize our fastapi app
app = FastAPI(title = "Text Summarizer App", description= " Text Summarization using a fine tuned T5 model", version="1.0")

# model and tokenizer
final_model = r"F:\Text_summarizer\final_model"

test_model = T5ForConditionalGeneration.from_pretrained(final_model)
test_tokenizer = T5Tokenizer.from_pretrained(final_model)

# device

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


test_model.to(device)

# Put model into evaluation mode
test_model.eval()


# templating

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory= "static"), name= "static")

# input schema for dialogue ==string

class DialogueInput(BaseModel):
    dialouge: str

def clean_data(text):
    
    text = re.sub(r"\r\n", " ", text) #lines
    text = re.sub(r"\s+", " ", text) #spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue):

  #clean the dialogue using the clean_data function
  dialogue = clean_data(dialogue)

  #Tokenize the dialogue

  inputs = test_tokenizer(
      dialogue, padding = "max_length",
      max_length = 512,
      trunctaion = True,
      return_tensors = "pt"
  )

  # Generate summary token ids
  with torch.no_grad():

    summary_ids = test_model.generate(
    input_ids = inputs["input_ids"].to(device),
    attention_mask = inputs["attention_mask"].to(device),
    max_length = 150,
    num_beams = 4,
    early_stopping=True
)
  # Decode summary token ids into text

  summary = test_tokenizer.decode(
      summary_ids[0],
      skip_special_tokens = True
  )


  # Return the summary

  return summary

# API Endpoints 
@app.post("/summarize/")
async def summarize(request: Request):

    # Read JSON sent by the frontend
    data = await request.json()

    # Get the dialogue from the JSON
    dialogue = data["dialogue"]

    # Generate summary
    summary = summarize_dialogue(dialogue)

    # Send summary back to frontend
    return {"summary": summary}


@app.get("/", response_class= HTMLResponse)
async def home(request: Request):
   return templates.TemplateResponse(
    request=request,
    name="index.html"
)







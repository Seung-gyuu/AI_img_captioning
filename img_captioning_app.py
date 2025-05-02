from calendar import c
import gradio as gr
import numpy as np
from PIL import Image
from model_loader import processor, model, pipe 

def caption_image(input_image: np.ndarray):
    if input_image is None:
        return "No image provided. Please upload an image."   
    
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    inputs = processor(images=raw_image, return_tensors="pt") 
    outputs = model.generate(**inputs, max_length=50)
    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    vit_results = pipe(raw_image, top_k=3)
    classification = {item['label'] : item['score'] for item in vit_results}
    
    
    return caption, classification
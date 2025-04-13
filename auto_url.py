import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def get_image_caption(image_url):
    try:
        response = requests.get(image_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_elements = soup.find_all('img')
        
        res = []
        for i in img_elements:
            img_url = i.get('src')
            if not img_url:
                continue
            if 'svg' in img_url or '1x1' in img_url:
                continue
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith('http://') and not img_url.startswith('https://'):
                continue
            
            try:
                # Download the image
                response = requests.get(img_url)
                # Convert the image data to a PIL Image
                raw_image = Image.open(BytesIO(response.content))
                if raw_image.size[0] * raw_image.size[1] < 400:  # Skip very small images
                    continue

                raw_image = raw_image.convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(model.device)
                

                out = model.generate(**inputs, max_new_tokens=50)

                caption = processor.decode(out[0], skip_special_tokens=True)
                res.append((raw_image, caption))
                if len(res) >= 5:  
                    break
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue

        return res if res else "No suitable images found."

    except Exception as e:
        return f"Error loading page: {e}"


with gr.Blocks(theme=gr.themes.Soft(),title="🖼️ Web Image Captioning") as app:
    gr.Markdown("<div style='text-align:center; font-size: 36px;'>🌐 Web Page Image Captioning with BLIP</div>")
    gr.Markdown("Enter a URL, and get AI-generated captions for images on that page.")

    url_input = gr.Textbox(label="Enter URL", placeholder="https://en.wikipedia.org/wiki/IBM")
    submit_btn = gr.Button("🔍 Generate Captions")

    gallery = gr.Gallery(label="Image Captions", show_label=False)
    submit_btn.click(fn=get_image_caption, inputs=url_input, outputs=gallery)

app.launch(debug=True)
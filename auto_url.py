import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from model_loader import processor, model, pipe

img_list = []

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
        
                vit_results = pipe(raw_image, top_k=3)
                classification = {item['label']: item['score'] for item in vit_results}

                
                res.append((raw_image, caption, classification))
                
                if len(res) >= 5:  
                    break
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue
        if not res:
            return "No suitable images found."

        global img_list
        img_list = res

        return [item[0] for item in res]  

    except Exception as e:
        return f"Error loading page: {e}"


def get_selected_image_info(index):
    if 0 <= index < len(img_list):
        return img_list[index]
    return None

from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline

# Autoprocessor: used for preprocessing data for the BLIP model. 
# can handle both image and text data, preparing it for input into the BLIP model.

# BLIP model: a transformer-based model designed for image captioning tasks.

# BlipForConditionalGeneration : can generate text based on an input image and an optional piece of text. 
# This makes it useful for tasks like image captioning or visual question answering, 
# where the model needs to generate text that describes an image or answer a question about an image.

#image classification
pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
#image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
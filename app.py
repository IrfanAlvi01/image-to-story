from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())

# img2text
def image2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

    # text = image_to_text(url)[0]["generated_text"]
    text = image_to_text(url)

    print(text)

    return text


image2text("students.jpg")
# lim

# text to speech
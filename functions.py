from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection, pipeline
from PIL import Image
import torch

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI

def get_image_caption(image_path):

    image = Image.open(image_path).convert("RGB")

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors= "pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_token = True)

    return caption

def detect_objects(image_path):
    image = Image.open(image_path).convert("RGB")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += "[{},{},{},{}]".format(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
        detections += " {}".format(model.config.id2label[int(label)])
        detections += " {}\n".format(float(score))

    return detections


def generate_story(scenario):
    template = """
    you are a story teller;
    you can generate short story based on a simple narrative, the story should be no more than 200 words

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    # llm=OpenAI(openai_api_key="key" ,model_name="gpt-3.5-turbo", temperature = 0)
    llm = ChatOpenAI(
    # openai_api_key="key",
    temperature=0,
    model_name="gpt-3.5-turbo"
    )

    story_llm = LLMChain(llm=llm, prompt = prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    # print(story)
    return story

def get_story(image_path):
    caption = get_image_caption(image_path)
    story = generate_story(caption)

    return story


if __name__ == '__main__':
    image_path = "temp.jpg"
    # caption = get_image_caption(image_path)
    # print(caption)
    # detections = detect_objects(image_path)
    # print(detections)
    # story = generate_story("three women sitting on a bench in a field of tulips")
    # print(story)

    story = get_story(image_path)
    print(story)
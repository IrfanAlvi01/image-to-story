from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

class StoryTeller(BaseTool):
    name = "Story teller"
    description = "Use this tool when ask to tell or descirbe a story related to some scenario. " \
                  "It will return a small story in numbers of words asked by user."

    def _run(self, img_path):
        image = Image.open(img_path).convert("RGB")

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors= "pt").to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        scenario = processor.decode(output[0], skip_special_token = True)
        # scenario or caption from image is generated

        template = """
        you are a story teller;
        you can generate short story based on a simple narrative, the story should be no more than 200 words

        CONTEXT: {scenario}
        STORY:
        """
        prompt = PromptTemplate(template=template, input_variables=["scenario"])

        # llm=OpenAI(openai_api_key="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" ,model_name="gpt-3.5-turbo", temperature = 0)
        llm = ChatOpenAI(
        openai_api_key="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",
        temperature=0,
        model_name="gpt-3.5-turbo"
        )

        story_llm = LLMChain(llm=llm, prompt = prompt, verbose=True)

        story = story_llm.predict(scenario=scenario)

        # print(story)
        return story

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
# from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
# import os

load_dotenv(find_dotenv())

# Set the OPENAI_API_KEY from environment variables
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# img2text
def image2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]
    # text = image_to_text(url)

    print(text)

    return text

 
# lim
def generate_story(scenario):
    template = """
    you are a story teller;
    you can generate short story based on a simple narrative, the story should be no more than 20 words

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    llm=OpenAI(openai_api_key="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" ,model_name="gpt-3.5-turbo", temperature = 1)

    story_llm = LLMChain(llm=llm, prompt = prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    # print(story)
    return story

scenario = image2text("students.jpg")
story = generate_story(scenario)

# text to speech
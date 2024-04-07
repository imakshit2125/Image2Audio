from dotenv import find_dotenv, load_dotenv
from transformers import pipeline 
from langchain import PromptTemplate, LLMChain,  OpenAI
import  requests
import streamlit as st

load_dotenv(find_dotenv())

# convert an image into a text
def img2Text(url):
    imageToText = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text= imageToText(url)

    print(text[0]["generated_text"])
    return text[0]["generated_text"]

# Create a story on based of this text

def createStory(scenario):
    template="""
    You are a story teller;
    Generate a story on based of simple barrative , the story should bew less than 20 words;

    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    storyLLM = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=0.7
    ),prompt=prompt,verbose=True);

    storyGenerated = storyLLM.predict(scenario=scenario);

    print(storyGenerated)
    return storyGenerated


#Story_to_Audio

def StoryToAudio(story):

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer your_token"}
    payload = {
        "inputs":story
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac','wb') as file:
        file.write(response.content);
    

# text1= img2Text("pics.jpeg")
# Story1 = createStory(text1)
# StoryToAudio(Story1)

def main():
    st.set_page_config(page_title="Image to Audible Story")

    st.header("Turn the image into audio Story")
    uploaded_file = st.file_uploader("Choose the Image",type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb")as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption="Uploaded Image",
                 use_column_width=True)
        text1= img2Text(uploaded_file.name)
        Story1 = createStory(text1)
        StoryToAudio(Story1)

        with   st.expander("text1"):
            st.write(text1)
        with st.expander("Story1"):
            st.write(Story1)
        
        st.audio("audio.flac")

if __name__=='__main__':
    main()
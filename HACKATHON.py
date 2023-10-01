#pip install opencv
#pip install pytesseract
#pip install gtts

import cv2
import pytesseract
from gtts import gTTS
import requests
import json
import openai
from serpapi import GoogleSearch
import urllib.request
from io import BytesIO

from PIL import Image

# Path to Tesseract executable (change this to the path on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image using OpenCnV
image = cv2.imread('aliceinwonderland.jpg')

# Convert the image to grayscale (optional)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply some preprocessing like thresholding or blurring if necessary
# Example: gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(gray)

# Print the extracted text
print(text)

def TTS():
    #the text to speech
        language = "en"
        speech = gTTS(text=text, lang=language, slow=False, tld="com.au")
        speech.save("texttospeech.mp3")
        print("The TTS is ready!")

#dictionary
def dictionary():
    givenword = str(input("What word do you need the definition of? "))
    word = givenword
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    data = response.json()
    definition = data[0]['meanings'][0]['definitions'][0]['definition']
    print(definition)

####################################################################################
print("####################################################################################")

# Set up your OpenAI API key
api_key = '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'
openai.api_key = api_key

def AI():
    # Define a prompt for text generation

    givenprompt = input("What would you like to ask the AI")
    
    prompt = givenprompt
    # Make an API call to generate text
    response = openai.Completion.create(
        engine="davinci",  # You can choose from different engines (e.g., "davinci", "davinci-codex")
        prompt=prompt,
        max_tokens=100,  # Maximum number of tokens in the generated text
        temperature=0.7,  # Controls the randomness of the output
        top_p=1.0,  # Limits the diversity of the output (higher values make output more focused)
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Get the generated text from the API response
    generated_text = response.choices[0].text.strip()

    # Print the generated text
    print(generated_text)

#################

def GoogleImage():

    search = input("Enter your Google Image Search:")
    owner_id = 310433679342043136
    params = {
    "q": search,
    "tbm": "isch",
    "ijn": "0",
    "api_key": "55330978bd4a00efc00e670cd65c57f910c5984cc3a4916f6cf0d61d6e250691"
    }


    search = GoogleSearch(params)
    results = search.get_dict()
    thumbnail = results["images_results"][0]["thumbnail"]
    titlegoogle = results["images_results"][0]["title"]
    link = results["images_results"][0]["link"]
    source = results["images_results"][0]["source"]
    print(thumbnail)



number = int(input("What Features would you like to use? 1:Text to Speech 2:Dictionary 3:AI 4:GoogleSearch \n"))

if number == 1:
    TTS()
elif number == 2:
    dictionary()
elif number == 3:
    AI()
elif number == 4:
    GoogleImage()
else:
    print("function not found")
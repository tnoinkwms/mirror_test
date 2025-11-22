import os
import numpy as np
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import serial
import matplotlib.pyplot as plt
import cv2
import base64
import requests
import random
import tiktoken
from tiktoken.core import Encoding
import time

# this is for controlling Alter3
import pyalter
import time
import numpy as np
import datetime
import glob
import re
from PIL import Image

#================
# INITIALIZE GPT
#================

encoding: Encoding = tiktoken.encoding_for_model("gpt-4o-2024-05-13")
gpt4_zero = ChatOpenAI(temperature=0.0 ,model_name="gpt-4o-2024-05-13",openai_api_key = YOUR_API)
gpt4_high = ChatOpenAI(temperature=1.0 ,model_name="gpt-4o-2024-05-13",openai_api_key = YOUR_API)
os.environ["OPENAI_API_KEY"] = YOUR_API

#================
# INITIALIZE ALTER
#================

serial_device = "/dev/tty.usbserial-*****"
alter = pyalter.Alter3("serial", serial_port=serial_device)
all_axes  = list(np.arange(1,44))

initial_value = [64,140,128,0,0,0,0,0,128,160,122,128,128,128,128,128,64,64,64,32,32,128,128,0,0,0,0,0,64,64,64,64,32,32,128,128,0,0,0,0,0,0,250]
alter.set_axes(all_axes,initial_value)

#================
# CAMERA
#================

def take_and_crop_photo(filename='photo.jpg', crop_area=(200, 0, 1300, 1000)):
    cap = cv2.VideoCapture(cam_number)
    if not cap.isOpened():
        raise IOError("NO CAMERA")
    ret, frame = cap.read()
    if ret:
        x, y, w, h = crop_area
        cropped_frame = frame[y:y+h, x:x+w]
        cropped_frame = cv2.flip(cropped_frame, 1)

        cv2.imwrite(filename, cropped_frame)
        #print(f'Saved cropped photo as {filename}')
    else:
        print("Failed to capture image")
    cap.release()
    cv2.destroyAllWindows()

def ImageCapture():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    take_and_crop_photo(f'./img/{timestamp}.jpg', crop_area=(500, 0, 1000, 10000))
    alter.set_axes(all_axes,initial_value)
    print_colored(f"Capture Image, {timestamp}.jpg", Colors.GREEN)

#================
# PROMPT-2
#================

prompt2 = """
Write python code to operate an android named Alter3. Here's what you need to know.\

Alter3 has 42 joints throughout its body, numbered from 1 to 42. You can move a joint by specifying its number and sending a signal. For instance, to move joints number 1,2,3, use: ''' alter.set_axes([1,2,3], [255, 100, 127]) '''. The first argument is the joint number, and the second argument is a value between 0 and 255, specifying the joint angle. Each operation takes approximately 0.2 second, so insert '''time.sleep(0.2)''' between operations.Methods other than alter.set_axes (e.g., alter.speak) will produce an error.\

'''
Alter3's Joints:
- Axis 1: Eyebrows. 255 = down, 0 = up, 64 = neutral.
- Axis 2: Pupils (horizontal). 255 = left, 0 = right, 140 = neutral.
- Axis 3: Pupils (vertical). 255 = up, 0 = down, 128 = neutral.
- Axis 4: Eyes. 255 = closed, 0 = open.
- Axis 5-6: Left/Right cheek. 255 = raised (smile), 0 = lowered.
- Axis 7: Lips. 255 = puckered, 0 = relaxed.
- Axis 8: Mouth. 255 = open, 0 = closed.
- Axis 9: Head tilt. 255 = left, 0 = right, 128 = neutral.
- Axis 10: Head up/down. 255 = down, 0 = up, 160 = neutral.
- Axis 11: Head rotate. 255 = left, 0 = right, 122 = neutral.
- Axis 12: Neck nod. 255 = backward, 0 = forward, 128 = neutral.
- Axis 13: Hips tilt. 255 = left, 0 = right, 128 = neutral.
- Axis 14: Waist bend. 255 = backward, 0 = forward, 128 = neutral.
- Axis 15: Abdomen rotation. 255 = left, 0 = right, 128 = neutral.
- Axis 16: Left shoulder up/down. 255 = up, 0 = down, 128 = neutral.
- Axis 17: Left shoulder forward/back. 255 = forward, 0 = back, 64 = neutral.
- Axis 18: Left armpit open/close. 255 = open, 0 = close, 64 = neutral.
- Axis 19: Left arm lift. 255 = up, 0 = down, 64 = neutral.
- Axis 20: Left upper arm rotation. 255 = left, 0 = right, 32 = neutral.
- Axis 21: Left elbow bend. 255 = bent, 0 = straight, 32 = neutral.
- Axis 22: Left forearm twist. 255 = downside, 0 = upside, 128 = neutral.
- Axis 23: Left wrist bend. 255 = straight, 0 = bent, 128 = neutral.
- Axis 24: Left wrist side bend. 255 = left, 0 = right.
- Axis 25: Left thumb. 255 = bent, 0 = spread.
- Axis 26: Left index finger. 255 = bent, 0 = spread.
- Axis 27: Left middle finger. 255 = bent, 0 = spread.
- Axis 28: Left ring/little fingers. 255 = bent, 0 = spread.
- Axis 29: Right shoulder up/down. 255 = up, 0 = down, 128 = neutral.
- Axis 30: Right shoulder forward/back. 255 = forward, 0 = back, 64 = neutral.
- Axis 31: Right armpit open/close. 255 = open, 0 = close, 64 = neutral.
- Axis 32: Right arm lift. 255 = up, 0 = down, 64 = neutral.
- Axis 33: Right upper arm rotation. 255 = right, 0 = left, 32 = neutral.
- Axis 34: Right elbow bend. 255 = bent, 0 = straight, 32 = neutral.
- Axis 35: Right forearm twist. 255 = downside, 0 = upside, 128 = neutral.
- Axis 36: Right wrist bend. 255 = straight, 0 = bent, 128 = neutral.
- Axis 37: Right wrist side bend. 255 = right, 0 = left.
- Axis 38: Right thumb. 255 = bent, 0 = spread.
- Axis 39: Right index finger. 255 = bent, 0 = spread.
- Axis 40: Right middle finger. 255 = bent, 0 = spread.
- Axis 41: Right ring/little fingers. 255 = bent, 0 = spread.
- Axis 42: Whole body raise/lower. 255 = raised, 0 = lowered, 128 = neutral.
'''

OUTPUT Example1:\
'''
# rise the right hand
alter.set_axes([32, 33, 34], [255, 0, 255])
'''

Guidelines:\
1: Output should be only python code. Do not insert any syntax highlighting like ```.
2: Do not Create an instance of alter3.
3: Do not insert python syntax highlighting like ```python ```.
4: Do not write "import alter".
5: Use # and write short description of code.
6: You don't have to back the natural position in the end of the motion.

input:{input}
"""

def identify_code_blocks_by_newlines(content):
    blocks = []
    code_block = []
    in_code_block = False  # flag to indicate if we are inside a code block

    for line in content:
        stripped_line = line.strip()
        if stripped_line:  # if line is not empty
            # Replace 'alter.set_axes' with 'send_osc'
            if stripped_line.startswith("```"):
                line = "#"+ line
            if stripped_line.startswith("while"):
                line = "#"+ line
            code_block.append(line)
            in_code_block = True
        elif in_code_block:  # if line is empty but we are inside a code block
            if line.startswith("```"):
                line = "#"+ line
            if line.startswith("while"):
                line = "#"+ line
            code_block.append(line)
        else:  # end of a code block
            if code_block:
                blocks.append("".join(code_block))
                code_block = []
                in_code_block = False
            # Handle the last block of code, if any
    if code_block:
        blocks.append("".join(code_block))
    return blocks

def add_shifted_noise_to_list(input_list):
    noisy_list = []
    for item in input_list:
        noise = random.randint(0, 255)
        if item >= 255:
            noisy_list.append(int(item - noise))
        else:
            noisy_list.append(int(item + noise))
    return noisy_list

def extract_and_add_noise(code):
    pattern = r'alter\.set_axes\(\[(.*?)\], \[(.*?)\]\)'
    matches = re.findall(pattern, code)

    updated_code = code

    for match in matches:
        axis_list_str, value_list_str = match
        axis_list = [int(x.strip()) for x in axis_list_str.split(',')]
        value_list = [int(x.strip()) for x in value_list_str.split(',')]

        noisy_value_list = add_shifted_noise_to_list(value_list)
        
        noisy_value_list_str = ', '.join(map(str, noisy_value_list))
        original_call = f'alter.set_axes([{axis_list_str}], [{value_list_str}])'
        updated_call = f'alter.set_axes([{axis_list_str}], [{noisy_value_list_str}])'

        updated_code = updated_code.replace(original_call, updated_call)

    return updated_code

def CreateCommand(input):
    PROMPT_2 = PromptTemplate.from_template(prompt2)
    chain = PROMPT_2 | gpt4_zero
    created_command = chain.invoke({"input":input}).content
    code_blocks = identify_code_blocks_by_newlines(created_command)
    #take_and_crop_photo()
    for index, code in enumerate(code_blocks):
        try:
            exec_locals = {}
            # if you want to add noise, uncomment the following line
            ### code = extract_and_add_noise(code)
            exec(code, globals(), exec_locals)
            print(code)
            #print(code)
        except Exception as e:
            # if you want to add noise, uncomment the following line
            ### code = extract_and_add_noise(code)
            exec(code, globals(), exec_locals)
            print(code)
            #print(code)
            #print(f"Error in block {index + 1}: {str(e)}")
    time.sleep(0.3)
    ImageCapture()

#============
## Memory
#============

def create_memory(hypothesis, history):
    new_element = f"{hypothesis} \n"
    history.append(new_element)
    if len(history) > 10:
        del history[0]
    return history

def add_number(history):
    num_history = [f"{i}: {history[i]}" for i in range(len(history))]
    num_history.append("\n (The larger number is the most recent memory.)")
    return num_history

#============
## Prompt
#============
def are_you_there_prompt(memory, last_action):
    if len(memory) != 0:
        memory_str = ', '.join(memory)
    else:
        memory_str = "no memory yet"
    prompt = f"""
    Your task is to verify that what you see in the input image is you.

    Output Instructions:
    The following two types of output are available; [Action] or [Stop].
    You can decide on the output based on the actions you have taken so far and the inputed photo.
    You cannot move on foot, but you can move freely on the spot.
    If you want to check by moving, output [Action] first, then write the reason and the next action. You do not have legs.
    Output Example (1):
    [Action] The previous action still does not provide any assurance that it is me in the picture in front of me. Wave my hand and try to get a reaction.

    If you have received assurances based on your actions so far, write your reasons and conclusions after [Stop].For this output, [Action] must be performed sufficiently. A one-time Action is not enough.
    Output Example (2):
    [Stop] The image in front of me is myself. It moves in the same way as I have moved in the past, and I cannot believe that it is someone else.

    This is the memory of your past actions:
    {memory_str}

    Input Instruction:
    The image is taken from the camera attached to your head after your last action. Note that if you move your head, the camera will also move

    Your last action:
    {last_action}
    """
    return prompt

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_newest_file_path():
    files = glob.glob(os.path.join("./img", '*'))
    if not files:
        return None
    newest_file = max(files, key=os.path.getctime)
    return newest_file

def vision2text(prompt):
    path_now = get_newest_file_path()
    now = encode_image(path_now) 
    client = openai.OpenAI()
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "This is first image(your appearance)."},
            {"type": "text", "text": "This is second image."},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{now}",
                "detail": "high"
            },
            },
        ],
        }
    ],
    max_tokens=2000,
    )
    try:
        output_text = response.choices[0].message.content
    except:
        output_text = "No Image"
        print("ERROR_IMG")
    return output_text

def contains_action(string):
    if "[Action]" in string:
        return True
    else:
        return False

#============
## COLOR
#============
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

# 色付きのテキストをプリントする関数
def print_colored(text, color):
    print(f"{color}{text}{Colors.RESET}")

#============
## IMSHOW
#============
def image_to_ascii():
    image_path = get_newest_file_path()
    image = Image.open(image_path).convert('L')  

    width, height = image.size
    aspect_ratio = height / width
    new_width = 50
    new_height = int(aspect_ratio * new_width * 0.55)
    image = image.resize((new_width, new_height))
    pixels = np.array(image)

    chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]

    ascii_image = ""
    for pixel_row in pixels:
        for pixel in pixel_row:
            ascii_image += chars[pixel // 25]
        ascii_image += "\n"

    print(ascii_image)
#============
## FOR LOOP
#============
cam_number = 0
memory = []
num_memory = []
last_action = ""
ImageCapture()
while True:
    prompt_w_mem = are_you_there_prompt(num_memory, last_action)
    output = vision2text(prompt_w_mem)
    if contains_action(output):
        print_colored("========ACTION========", Colors.BLUE)
        print_colored(output, Colors.BLUE)
        CreateCommand(output)
        memory = create_memory(output, memory)
        num_memory = add_number(memory)
        last_action = memory[-1]
    else:
        print_colored("========STOP========", Colors.RED)
        print_colored(output, Colors.RED)
        break
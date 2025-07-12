from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
import openai
from openai import OpenAI
import asyncio
import logging                                                                    #importing all the stuff needed
from transformers import pipeline, CLIPProcessor, CLIPModel
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import uuid
import matplotlib.pyplot as plt
from gtts import gTTS
import tempfile
import requests
import json
import datetime
import torchaudio

pipe = StableDiffusionPipeline.from_pretrained(        #loading stable diffusion model
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
).to("cpu")

audpipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")     #loading whisper for audio transcription


clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")           #loading clip model for image to prompt
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")



TELEGRAM_BOT_TOKEN = "your-telegram-token-here"
OPENAI_API_KEY = "your-openai-api-key-here"




bot = Bot(token=TELEGRAM_BOT_TOKEN) 
dp = Dispatcher()                                                                     #initialising the bot and client
client = OpenAI(api_key=OPENAI_API_KEY)                     

#creating a memory and cache
memory = {}
user_images = {}                                             #stores images sent by user


logging.basicConfig(level=logging.INFO)                      #logging




#=============================================================== helper functions =================================================================

def log_messages_to_json(user_name, user_id, role, message):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_name": user_name,                                          #logs all the conversations into a json file
        "user_id": user_id,
        "role": role,
        "message": message
    }
    with open(r"D:\Elricus Bot\chat_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def ai_reply(user_id: int, user_input: str) -> str:
    try:
        if user_id not in memory:
            memory[user_id] = []
 
        memory[user_id].append({'role': 'user', 'content': user_input})               #replies to the user

        system_prompt = {
            "role": "system",
            "content": "You're a sarcastic, funky, funny, taunting, fun loving, and bossy bot, you really love to use emojis and don't hold back on them"
        }

        messages = [system_prompt] + memory[user_id]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.3
        )

        reply = response.choices[0].message.content.strip()
        memory[user_id].append({'role': 'assistant', 'content': reply})

        return reply

    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "GPT had a memory meltdown brrrrrrrrr...."

def generate(pipe, prompt, params):
    try:
        output = pipe(prompt, **params)
        images = output.images
        img_paths = []

        for i, img in enumerate(images):
            path = f"generated_{uuid.uuid4().hex[:8]}_{i}.png"                   #generates images
            img.save(path)
            img_paths.append(path)

        return img_paths

    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        return []

def reverse_prompt(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    
    # Sample captions to match
    text_inputs = [
        "a dog playing fetch",
        "a cyberpunk city at night",
        "an astronaut riding a horse",
        "a scenic mountain landscape",                                                                
        "a delicious slice of pizza",
        "a portrait of a mysterious woman",                                      #tries to math the image to prompts by giving some samples
        "a robot cooking in a kitchen",
        "a surreal dreamscape with melting clocks",
        "a beach with palm trees and clear water",
        "a cat sitting on a laptop"
    ]

    inputs = clip_processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best = probs[0].argmax().item()

    return text_inputs[best]

#========================================================================== bot commands ==========================================================

@dp.message(Command("start"))
async def start(message: Message):                              #/start
    await message.answer("Hey there, howdy doody? 🤠")

@dp.message(Command("help"))
async def help(message: Message):
    await message.answer(
        "1. /start to start the convo\n\n"
        "2. /imagine to generate an image\n\n"
        "3. /speak to generate audio from text\n\n"
        "4. /wipe to erase the memory\n"                                                    #/help
        "5. Just send a voice note and I'll reply with text \n\n"
        "6. Send an image and say 'prompt this' to reverse it into a prompt "
    )

@dp.message(Command("wipe"))
async def wipe(message: Message):
    memory.clear()                                                                          #/wipe
    user_images.clear()
    await message.answer("Memory wiped! Now I'm the same as Ghajini 🧠")

@dp.message(Command("imagine"))
async def imagine(message: Message):
    prompt = message.text.replace("/imagine", "").strip()

    if not prompt:
        await message.answer("......prompt where?, do better kiddo 😮‍💨")                          #/imagine
        return

    await message.answer("Generating your image 👷🏻‍♂️⚒️")
    params = {"num_inference_steps": 30, "guidance_scale": 7.5}
    paths = generate(pipe, prompt, params)

    if not paths:
        await message.answer("I had a creative breakdown 😩, please try later")
        return

    for path in paths:
        await message.answer_photo(types.FSInputFile(path))
        os.remove(path)

@dp.message(Command("speak"))
async def speak(message: Message):
    text = message.text.replace("/speak", "").strip()

    try:
        if not text:
            await message.answer("Can't speak empty thoughts, bro 😵")
            return

        await message.answer("Beep Boop... Elricus warming up his vocal cords 🎤")
        tts = gTTS(text=text, lang='en')
        audiopath = f"tts_{message.message_id}.mp3"                                                #/speak
        tts.save(audiopath)

        await message.answer_voice(types.FSInputFile(audiopath))
        os.remove(audiopath)

    except Exception as e:
        logging.error(f"Audio generation failed: {e}")
        await message.answer("I got a sore throat, please try later 😷")

#====================================================================== handelrs ====================================================================

@dp.message(lambda m: m.photo)
async def handle_image(message: Message):
    user_id = message.from_user.id
    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
    response = requests.get(file_url)

    path = f"temp_{uuid.uuid4().hex[:8]}.jpg"
    with open(path, "wb") as f:
        f.write(response.content)

    user_images[user_id] = path
    await message.answer("Got the image! 📸 Now say 'prompt this' to reverse it into a prompt 🧠")

@dp.message()
async def handle_voice_or_text(message: Message):
    user_id = message.from_user.id
    user_name = message.from_user.full_name

    if message.voice:
        try:
            file = await bot.get_file(message.voice.file_id)
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
            response = requests.get(file_url)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
            temp_file.write(response.content)
            temp_file.close()

            waveform, sample_rate = torchaudio.load(temp_file.name)
            transcription = audpipe({
                "array": waveform.squeeze().numpy(),
                "sampling_rate": sample_rate
            })["text"]

            os.remove(temp_file.name)

            reply = ai_reply(user_id, transcription)
            log_messages_to_json(user_name, user_id, "user", transcription)
            log_messages_to_json("Elricus", "BOT", "bot", reply)

            await message.answer(reply)

        except Exception as e:
            logging.error(f"Voice process failed: {e}")
            await message.answer(f"Something glitched 😵\n\n`{e}`")
        return

    if message.text:
        user_text = message.text.strip().lower()
        print(f"\n👤 {user_name} ({user_id}): {user_text}")

        # If user previously sent an image and says "prompt this"
        if user_id in user_images:
            image_path = user_images[user_id]

            if "prompt" in user_text:
                await message.answer("Let me reverse-engineer this masterpiece... 🧠🧑‍🎨")
                try:
                    guessed_prompt = reverse_prompt(image_path)
                    await message.answer(f"🤖 I think this looks like: *'{guessed_prompt}'*")
                except Exception as e:
                    logging.error(f"Prompt reversion failed: {e}")
                    await message.answer("Ugh, can't figure this one out 😵‍💫")

                os.remove(image_path)
                del user_images[user_id]
                return

            await message.answer("You sent me an image, but didn’t say what to do 🤨. Try 'prompt this'.")
            return

       
        reply = ai_reply(user_id, user_text)
        log_messages_to_json(user_name, user_id, "user", user_text)
        log_messages_to_json("Elricus", "BOT", "bot", reply)                     

        print(f"🤖 Elricus: {reply}\n")
        await message.answer(reply)

# ================================================================ bot runner ==========================================================

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

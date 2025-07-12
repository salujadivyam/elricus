
# Elricus - Multi-Modal Telegram AI Assistant

Developer: Divyam Saluja

Elricus is a smart, multi-modal AI assistant built on Telegram that processes voice, image, and text seamlessly. It combines powerful GenAI models to create a conversational experience with memory and personality.

## ✨ Features

- Natural text conversation powered by GPT-4o-mini
- Voice message transcription using Whisper
- Image captioning with BLIP
- Image understanding and matching via CLIP
- Image generation using Stable Diffusion
- Text-to-speech replies using gTTS
- Contextual memory per user
- Built with async Python and aiogram

## 🧠 Tech Stack

- Python (async + aiogram)
- OpenAI GPT-4o-mini
- Whisper
- Hugging Face Transformers
  - Salesforce BLIP
  - OpenAI CLIP
- Stable Diffusion (diffusers)
- gTTS (Google Text-to-Speech)
- torch, PIL, matplotlib

## 🔐 Security

All API keys and tokens are securely managed using environment variables or a `.env` file.  
Do not hardcode secrets in your code. The `.env.example` file is provided for reference.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/elricus.git
cd elricus
```

### 2. Create a `.env` file

Create a `.env` file in the root directory:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
```

You can also copy the provided `.env.example` and fill in your details.

### 3. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the bot

```bash
python main.py
```

Make sure your bot is registered with BotFather and has the correct token.

## 🧪 Example Usage

- Voice: Send a voice message, get transcribed and replied to
- Image: Send an image, get a caption or analysis
- Text: Chat naturally; Elricus remembers your previous messages
- Generation: Ask it to generate images with text prompts

## 📁 Project Structure

```
elricus/
├── main.py                  # Entry point
├── handlers/                # Bot command & message logic
│   ├── text.py
│   ├── image.py
│   └── voice.py
├── utils/                   # Model and helper functions
│   ├── captioning.py
│   ├── clip_utils.py
│   └── speech.py
├── assets/                  # Optional media/static files
├── .env.example             # Template for secrets
├── requirements.txt
└── README.md
```

## ✅ Models Used

- openai/whisper
- Salesforce/blip-image-captioning-base
- openai/clip-vit-base-patch32
- stabilityai/stable-diffusion-2-1
- gTTS
- GPT-4o-mini

## 📚 Acknowledgments

This project was built as part of my summer school learning and personal experiments with GenAI.  
Thanks to OpenAI, Hugging Face, and the dev community for resources and model support.

## 📄 License

This project is licensed under the MIT License.  
See the `LICENSE` file for more details.

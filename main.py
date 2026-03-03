import os
import io
import telebot
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from flask import Flask
from threading import Thread

# Load secrets
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Clients
bot = telebot.TeleBot(BOT_TOKEN)
client = genai.Client(api_key=GEMINI_KEY)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "👋 **Welcome to Med-Clear!**\n\n"
        "I am your AI Medical Jargon Interpreter, powered by Gemini 3 Flash.\n\n"
        "📸 **How to use me:**\n"
        "1. Take a clear photo of a medical report, lab result, or prescription.\n"
        "2. Send it to this chat.\n"
        "3. I will simplify the complex terms and give you questions for your doctor.\n\n"
        "⚠️ *Disclaimer: I am an AI, not a doctor. My summaries are for informational purposes only and do not replace professional medical advice.*"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")
@bot.message_handler(content_types=['photo'])
def handle_medical_image(message):
    try:
        bot.reply_to(message, "🔍 Scanning your document with Gemini 3 Flash...")

        # 1. Download image from Telegram
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file))

        # 2. Configure the "Brain"
        # We use HIGH resolution for OCR and MEDIUM thinking for medical logic
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM
            ),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
        )

        prompt = """
        You are a Medical Jargon Interpreter. 
        Analyze the attached medical document/prescription:
        1. Extract names of labs or medications.
        2. Explain what they are in simple, non-scary language.
        3. Flag any values with '⚠️' if they seem outside a standard range.
        4. Provide 3 questions for the user to ask their doctor.
        
        STRICT: Do not diagnose. Use a supportive, clear tone.
        Include disclaimer: 'AI-generated summary. Not medical advice.'
        """

        # 3. Call Gemini 3 Flash
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, img],
            config=config
        )

        # 4. Send result back
        bot.reply_to(message, response.text, parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"❌ Oops! Something went wrong: {str(e)}")

print("🚀 Med-Clear Bot is live! Send a photo of a medical doc to your Telegram bot.")

# 2. THE RENDER KEEP-ALIVE LOGIC

server = Flask('')

@server.route('/')
def home():
    return "Med-Clear is running!"

def run():
    # Render's dynamic port
    port = int(os.environ.get("PORT", 10000))
    server.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True # This ensures the thread shuts down if the bot stops
    t.start()

# Start the web server thread
keep_alive()
bot.infinity_polling()
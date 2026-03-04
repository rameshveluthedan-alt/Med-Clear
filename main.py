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
        
        # Convert to bytes for Gemini (more stable than PIL object)
        img_bytes = downloaded_file 

        # 2. Configure the "Brain"
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
        Format your response using these HTML rules:

        1. **Disclaimer**: Wrap the disclaimer in <i><b>bold italics</b></i> at the very top.
        2. **Sections**: Use <u><b>UNDERLINED BOLD</b></u> for headers (e.g., <u><b>1. CLINICAL TERMS</b></u>).
        3. **Definitions**: Use <b>bold</b> for the term and <code>monospace</code> for the explanation.
            Example: <b>URTI</b>: <code>Upper Respiratory Tract Infection</code>.
        4. **Highlights**: use <code>code tags</code> for the numbers.
        
        STRICT: Do not diagnose. Use a supportive, clear tone.
        Include disclaimer: 'AI-generated summary. Not medical advice.'
        """

        # 3. Model Logic (NOW CORRECTLY INDENTED)
        try:
            # Try the high-quality OCR model first
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')],
                config=config
            )
        except Exception as e:
            # If Gemini 3 is busy (503), try the stable 2.0 version
            print(f"Gemini 3 busy/error: {e}. Falling back to 2.0...")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')],
                # Note: 2.0-flash may ignore thinking_level, but config is still valid
                config=config
            )

        # 4. Send result back
        bot.reply_to(message, response.text, parse_mode="HTML")

    except Exception as e:
        print(f"General Error: {e}")
        bot.reply_to(message, "❌ Oops! Something went wrong. Please try again in a moment.")
# 2. THE RENDER KEEP-ALIVE LOGIC

server = Flask('')

@server.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Med-Clear | AI Medical Jargon Interpreter</title>
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                margin: 0; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                min-height: 100vh; 
                background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
                color: #1e293b;
            }
            .container { 
                background: white; 
                padding: 40px 30px; 
                border-radius: 24px; 
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                max-width: 450px; 
                width: 90%; 
                text-align: center;
            }
            .icon-circle {
                width: 80px;
                height: 80px;
                background: #0088cc;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 20px;
                font-size: 40px;
            }
            h1 { font-size: 2rem; margin-bottom: 12px; color: #0f172a; }
            p { font-size: 1.1rem; color: #64748b; line-height: 1.5; margin-bottom: 30px; }
            .btn { 
                background: #0088cc; 
                color: white; 
                padding: 18px 40px; 
                text-decoration: none; 
                border-radius: 12px; 
                font-weight: 700; 
                font-size: 1.1rem; 
                display: block;
                transition: all 0.2s ease;
                box-shadow: 0 4px 6px rgba(0, 136, 204, 0.2);
            }
            .btn:hover { 
                background: #0077b5; 
                transform: translateY(-2px);
                box-shadow: 0 10px 15px rgba(0, 136, 204, 0.3);
            }
            .features {
                text-align: left;
                margin: 30px 0;
                font-size: 0.95rem;
                color: #475569;
            }
            .feature-item { margin-bottom: 8px; display: flex; align-items: center; }
            .feature-item::before { content: '✅'; margin-right: 10px; }
            .footer { 
                margin-top: 25px; 
                padding-top: 20px; 
                border-top: 1px solid #f1f5f9; 
                font-size: 0.75rem; 
                color: #94a3b8; 
                line-height: 1.4;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon-circle">🔍</div>
            <h1>Med-Clear</h1>
            <p>Understand your medical reports in plain English.</p>
            
            <div class="features">
                <div class="feature-item">Powered by Gemini 3.1 Flash-Lite</div>
                <div class="feature-item">Simplifies complex clinical terms</div>
                <div class="feature-item">Checks lab values for range</div>
                <div class="feature-item">Completely private & secure</div>
            </div>

            <a href="https://t.me/med_clear_bot" class="btn">Start Chat on Telegram</a>
            
            <div class="footer">
                <strong>Disclaimer:</strong> AI-generated summaries are for informational purposes only and do not replace professional medical advice. Always consult a physician.
            </div>
        </div>
    </body>
    </html>
    """

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
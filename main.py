"""
Med-Clear Telegram Bot — Refactored
====================================
Tech Stack : pyTelegramBotAPI + Flask + Google Gemini
Hosting    : Render Free Tier / Google Cloud Run ready
Author note: All 5 fine-tuning objectives addressed (see README below)
"""

import os
import re
import time
import logging
import threading

import telebot
from flask import Flask
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# 0.  BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("med-clear")

# Environment variables (works on Render AND Cloud Run)
BOT_TOKEN  = os.environ["TELEGRAM_TOKEN"]        # raises KeyError early — fail fast
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

bot    = telebot.TeleBot(BOT_TOKEN, threaded=True)
client = genai.Client(api_key=GEMINI_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SHARED PROMPT & GEMINI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are Med-Clear, a friendly Medical Jargon Interpreter.
Analyze the attached medical document (image or PDF) and do the following:
  1. Extract every lab test, medication, diagnosis code, or medical term.
  2. Explain each item in plain language a patient can understand.
  3. For numerical results, flag with ⚠️ if outside the standard adult reference range.
  4. List 3 specific questions the patient should ask their doctor.

════════════════════════════════════════
STRICT OUTPUT FORMAT — HTML ONLY
You MUST follow every rule below. Violating even one rule makes the response unusable.
════════════════════════════════════════
RULE 1 — DISCLAIMER (first line, always):
<i><b>DISCLAIMER: This is an AI-generated summary for informational purposes only. It is NOT medical advice. Always consult a qualified doctor.</b></i>

RULE 2 — SECTION HEADERS:
<u><b>1. SECTION TITLE</b></u>

RULE 3 — TERM DEFINITIONS:
<b>Term Name</b>: <code>Plain-language explanation</code>

RULE 4 — NUMERICAL VALUES:
Always wrap every number and its unit inside <code> tags.
Example: Your result was <code>6.8 mmol/L</code> ⚠️ (above the normal range of <code>3.9–5.5 mmol/L</code>).

RULE 5 — BANNED FORMATTING:
Do NOT use Markdown. No **, no __, no #, no *, no backtick outside <code>.

RULE 6 — NO DIAGNOSIS:
Never state a diagnosis. Use "may suggest", "can indicate", or "your doctor will interpret".

RULE 7 — MULTI-PAGE / TABLE DOCUMENTS:
Process every page and every table row. Do not skip values.

Example output skeleton (follow exactly):
<i><b>DISCLAIMER: ...</b></i>

<u><b>1. SUMMARY OF FINDINGS</b></u>
• <b>RBS (Random Blood Sugar)</b>: <code>Blood sugar measured at any time of day</code>. Your result: <code>165 mg/dL</code> ⚠️ (normal fasting: <code>70–99 mg/dL</code>).

<u><b>2. MEDICATIONS / PRESCRIPTIONS</b></u>
• <b>Metformin 500 mg</b>: <code>A medication commonly used to manage blood sugar levels in Type 2 diabetes.</code>

<u><b>3. QUESTIONS TO ASK YOUR DOCTOR</b></u>
1. ...
2. ...
3. ...
"""

_GEMINI_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
)

_MODEL_PRIMARY  = "gemini-2.5-flash-preview-05-20"   # best OCR + thinking
_MODEL_FALLBACK = "gemini-2.0-flash"                 # stable backup


def _gemini_generate(parts: list) -> str:
    """
    Two-tier Gemini call with retry logic.
    Returns the raw response text (HTML formatted by the model).
    """
    for attempt, model in enumerate((_MODEL_PRIMARY, _MODEL_FALLBACK), start=1):
        try:
            log.info("Gemini attempt %d — model: %s", attempt, model)
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=_GEMINI_CONFIG,
            )
            return response.text
        except Exception as exc:
            log.warning("Model %s failed (attempt %d): %s", model, attempt, exc)
            if attempt == 1:
                time.sleep(1.5)   # brief pause before fallback
    raise RuntimeError("Both Gemini models failed — see logs for details.")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  HTML SANITIZER  (Objective 1 — strict formatting)
# ──────────────────────────────────────────────────────────────────────────────

# Tags that Telegram's HTML parser actually supports
_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}

def _strip_markdown(text: str) -> str:
    """
    Belt-and-suspenders: remove residual Markdown that Gemini might sneak in
    even after explicit instructions.
    """
    # Bold/italic asterisks
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    # Bold/italic underscores
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text, flags=re.DOTALL)
    # ATX headers (#, ##, ###)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Inline code backticks NOT inside <code> (rare but possible)
    text = re.sub(r"(?<!<code>)`([^`]+)`(?!</code>)", r"<code>\1</code>", text)
    return text


def _ensure_disclaimer(text: str) -> str:
    """If the model forgot the disclaimer, prepend it."""
    disclaimer = (
        "<i><b>DISCLAIMER: This is an AI-generated summary for informational "
        "purposes only. It is NOT medical advice. Always consult a qualified "
        "doctor.</b></i>\n\n"
    )
    if "<i><b>DISCLAIMER" not in text:
        text = disclaimer + text
    return text


def sanitize_for_telegram(text: str) -> str:
    """Full pipeline: strip Markdown → guarantee disclaimer."""
    text = _strip_markdown(text)
    text = _ensure_disclaimer(text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SAFE REPLY HELPERS  (Objective 3 — error handling / chat hygiene)
# ──────────────────────────────────────────────────────────────────────────────

_MAX_MSG_LEN = 4096   # Telegram hard limit


def _send_long(chat_id: int, text: str, reply_to: int | None = None) -> None:
    """Split long responses and send as multiple messages."""
    chunks = [text[i : i + _MAX_MSG_LEN] for i in range(0, len(text), _MAX_MSG_LEN)]
    for idx, chunk in enumerate(chunks):
        kwargs = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        if idx == 0 and reply_to:
            kwargs["reply_to_message_id"] = reply_to
        try:
            bot.send_message(**kwargs)
        except telebot.apihelper.ApiTelegramException as e:
            log.error("Telegram send error: %s", e)
            # Retry once without parse_mode in case of malformed HTML
            bot.send_message(chat_id=chat_id, text=chunk)


def _edit_or_send(chat_id: int, message_id: int, text: str) -> None:
    """Edit an existing message; fall back to new message if edit fails."""
    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode="HTML",
        )
    except Exception as e:
        log.warning("edit_message_text failed (%s), sending new message.", e)
        _send_long(chat_id, text)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  COMMAND HANDLERS
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    welcome_text = (
        "👋 <b>Welcome to Med-Clear!</b>\n\n"
        "I am your AI Medical Jargon Interpreter, powered by Gemini.\n\n"
        "📸 <u><b>How to use me:</b></u>\n"
        "1. Take a clear photo of a medical report, lab result, or prescription.\n"
        "2. Send it here — or upload a <b>PDF</b>.\n"
        "3. I will explain complex terms in plain language and give you questions "
        "for your doctor.\n\n"
        "💬 You can also <b>type a question</b> about medical terminology.\n\n"
        "<i>⚠️ I am an AI assistant, not a licensed physician. My summaries are "
        "for informational purposes only and do not replace professional medical advice.</i>"
    )
    bot.reply_to(message, welcome_text, parse_mode="HTML")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PHOTO HANDLER  (Objective 2 — OCR)
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    status = bot.reply_to(message, "🔍 Analyzing your medical image…")
    try:
        file_info      = bot.get_file(message.photo[-1].file_id)
        img_bytes      = bot.download_file(file_info.file_path)
        img_part       = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        raw  = _gemini_generate([ANALYSIS_PROMPT, img_part])
        text = sanitize_for_telegram(raw)

        _edit_or_send(message.chat.id, status.message_id, text)

    except Exception as exc:
        log.error("handle_photo error: %s", exc)
        _edit_or_send(
            message.chat.id, status.message_id,
            "❌ Something went wrong while analyzing your image. Please try again."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DOCUMENT HANDLER  (Objective 2 — PDF + image-as-file)
# ──────────────────────────────────────────────────────────────────────────────

_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


@bot.message_handler(content_types=["document"])
def handle_document(message):
    doc       = message.document
    mime_type = doc.mime_type or ""

    # ── Route: image sent as a file ──────────────────────────────────────────
    if mime_type in _IMAGE_MIME_TYPES:
        status = bot.reply_to(message, "🔍 Analyzing your medical image…")
        try:
            file_info = bot.get_file(doc.file_id)
            img_bytes = bot.download_file(file_info.file_path)
            img_part  = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

            raw  = _gemini_generate([ANALYSIS_PROMPT, img_part])
            text = sanitize_for_telegram(raw)
            _edit_or_send(message.chat.id, status.message_id, text)
        except Exception as exc:
            log.error("handle_document (image) error: %s", exc)
            _edit_or_send(
                message.chat.id, status.message_id,
                "❌ Could not analyze that image file. Please try again."
            )
        return

    # ── Route: PDF ───────────────────────────────────────────────────────────
    if mime_type == "application/pdf":
        status = bot.reply_to(message, "📄 Reading your PDF report… this may take a moment.")
        try:
            file_info  = bot.get_file(doc.file_id)
            pdf_bytes  = bot.download_file(file_info.file_path)
            pdf_part   = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

            raw  = _gemini_generate([ANALYSIS_PROMPT, pdf_part])
            text = sanitize_for_telegram(raw)
            _edit_or_send(message.chat.id, status.message_id, text)
        except Exception as exc:
            log.error("handle_document (PDF) error: %s", exc)
            _edit_or_send(
                message.chat.id, status.message_id,
                "❌ I had trouble reading that PDF.\n\n"
                "Possible reasons:\n"
                "• The file is password-protected\n"
                "• The scan quality is very low\n"
                "• The file is corrupted\n\n"
                "Try sending a clear <b>photo</b> of each page instead.",
            )
        return

    # ── Unsupported type ─────────────────────────────────────────────────────
    bot.reply_to(
        message,
        "⚠️ Unsupported file type. Please send your report as a <b>PDF</b> or a <b>photo</b>.",
        parse_mode="HTML",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  TEXT / FOLLOW-UP Q&A HANDLER  (Objective 5 — guardrails)
# ──────────────────────────────────────────────────────────────────────────────

_TEXT_SYSTEM = """\
You are Med-Clear, a Medical Jargon Interpreter.
A user is asking a follow-up question about medical terminology or their report.

STRICT RULES:
1. NEVER provide a diagnosis or say "you have [condition]".
2. ALWAYS use hedging language: "may suggest", "can indicate", "your doctor will determine".
3. Format with Telegram HTML only (no Markdown).
4. End EVERY response with:
<i><b>Remember: This information is for educational purposes only. Please consult your doctor for personalized medical advice.</b></i>
"""

@bot.message_handler(func=lambda m: m.content_type == "text")
def handle_text(message):
    user_text = message.text.strip()
    if not user_text or user_text.startswith("/"):
        return

    status = bot.reply_to(message, "💬 Looking that up for you…")
    try:
        raw = _gemini_generate([
            {"role": "user", "parts": [_TEXT_SYSTEM + "\n\nUser question: " + user_text]}
        ])
        text = sanitize_for_telegram(raw)
        _edit_or_send(message.chat.id, status.message_id, text)
    except Exception as exc:
        log.error("handle_text error: %s", exc)
        _edit_or_send(
            message.chat.id, status.message_id,
            "❌ I couldn't process that question. Please rephrase and try again."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8.  FLASK KEEP-ALIVE  (Objective 4 — Cloud Run ready)
# ──────────────────────────────────────────────────────────────────────────────

server = Flask(__name__)


@server.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Med-Clear | AI Medical Jargon Interpreter</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 100%);
      color: #1e293b;
      padding: 20px;
    }
    .card {
      background: #ffffff;
      border-radius: 24px;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.12);
      max-width: 480px;
      width: 100%;
      padding: 48px 36px 40px;
      text-align: center;
    }
    .logo {
      width: 88px; height: 88px;
      background: linear-gradient(135deg, #0088cc, #00b4d8);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 42px;
      margin: 0 auto 24px;
      box-shadow: 0 8px 24px rgba(0,136,204,0.3);
    }
    h1 { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; }
    .tagline { font-size: 1.05rem; color: #64748b; line-height: 1.6; margin-bottom: 32px; }
    .features {
      background: #f8fafc;
      border-radius: 16px;
      padding: 20px 24px;
      text-align: left;
      margin-bottom: 32px;
    }
    .features h2 { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .08em; color: #94a3b8; margin-bottom: 14px; }
    .feature {
      display: flex; align-items: center; gap: 12px;
      font-size: 0.95rem; color: #334155;
      padding: 6px 0;
    }
    .feature-icon { font-size: 1.2rem; flex-shrink: 0; }
    .btn {
      display: block;
      background: linear-gradient(135deg, #0088cc, #00b4d8);
      color: #ffffff;
      text-decoration: none;
      padding: 18px 32px;
      border-radius: 14px;
      font-weight: 700;
      font-size: 1.05rem;
      transition: transform 0.18s, box-shadow 0.18s;
      box-shadow: 0 6px 20px rgba(0,136,204,0.35);
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,136,204,0.4); }
    .disclaimer {
      margin-top: 28px;
      padding-top: 22px;
      border-top: 1px solid #f1f5f9;
      font-size: 0.72rem;
      color: #94a3b8;
      line-height: 1.5;
    }
    .status-dot {
      display: inline-block; width: 8px; height: 8px;
      background: #22c55e; border-radius: 50%;
      animation: pulse 2s infinite;
      margin-right: 6px; vertical-align: middle;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🔍</div>
    <h1>Med-Clear</h1>
    <p class="tagline">Understand your medical reports in plain English — instantly.</p>

    <div class="features">
      <h2>What I can do</h2>
      <div class="feature"><span class="feature-icon">📸</span> Analyze photos of lab reports &amp; prescriptions</div>
      <div class="feature"><span class="feature-icon">📄</span> Process PDF documents (multi-page &amp; tables)</div>
      <div class="feature"><span class="feature-icon">🔬</span> Flag out-of-range lab values automatically</div>
      <div class="feature"><span class="feature-icon">💬</span> Answer follow-up medical terminology questions</div>
      <div class="feature"><span class="feature-icon">🔒</span> Private &amp; secure — no data stored</div>
    </div>

    <a href="https://t.me/med_clear_bot?start=welcome" class="btn">🚀 Open Med-Clear on Telegram</a>

    <p class="disclaimer">
      <span class="status-dot"></span><strong>Bot is online</strong><br><br>
      <strong>Medical Disclaimer:</strong> Med-Clear provides AI-generated summaries
      for informational purposes only. It is not a substitute for professional medical
      advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
    </p>
  </div>
</body>
</html>"""


@server.route("/health")
def health():
    """Health-check endpoint for Cloud Run / Render."""
    return {"status": "ok", "service": "med-clear"}, 200


def _run_flask():
    port = int(os.environ.get("PORT", 10000))
    # use_reloader=False is critical — avoids double-start in threaded environments
    server.run(host="0.0.0.0", port=port, use_reloader=False)


def keep_alive():
    t = threading.Thread(target=_run_flask, name="flask-keepalive", daemon=True)
    t.start()
    log.info("Flask keep-alive started.")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  POLLING WATCHDOG  (survives Render Free Tier cold-starts & Error 104)
# ──────────────────────────────────────────────────────────────────────────────

_polling_lock = threading.Lock()   # prevents two polling loops racing each other
_WATCHDOG_INTERVAL = 30            # seconds between liveness checks


def _start_polling():
    """
    Start infinity_polling inside a dedicated daemon thread.
    Safe to call multiple times — the lock prevents overlapping loops.
    """
    if not _polling_lock.acquire(blocking=False):
        log.warning("Polling already running — skipping duplicate start.")
        return

    def _poll():
        try:
            log.info("Polling thread started.")
            bot.infinity_polling(
                none_stop=True,
                interval=1,
                timeout=25,
                long_polling_timeout=25,
                logger_level=logging.WARNING,
                restart_on_change=False,
            )
        except Exception as exc:
            log.error("Polling loop crashed: %s", exc)
        finally:
            log.warning("Polling thread exited — watchdog will restart it.")
            _polling_lock.release()

    t = threading.Thread(target=_poll, name="bot-polling", daemon=True)
    t.start()


def _watchdog():
    """
    Runs forever in its own thread.
    Every WATCHDOG_INTERVAL seconds it checks whether the bot is still
    connected (get_me succeeds) and whether a polling thread is alive.
    If either check fails it calls _start_polling() to revive the bot.
    """
    # Give the first polling thread time to initialize
    time.sleep(10)

    while True:
        try:
            bot.get_me()                         # lightweight Telegram API ping
            log.info("Watchdog: bot connection OK.")
        except Exception as exc:
            log.warning("Watchdog: connection check failed (%s) — restarting polling.", exc)
            bot.stop_polling()                   # clean up any zombie state
            time.sleep(2)
            _start_polling()

        # Also restart if the polling lock was released (thread died quietly)
        if not _polling_lock.locked():
            log.warning("Watchdog: polling thread not running — restarting.")
            _start_polling()

        time.sleep(_WATCHDOG_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    keep_alive()

    # 1. Start the first polling loop
    _start_polling()

    # 2. Start the watchdog that resurrects polling after idle/cold-start drops
    wd = threading.Thread(target=_watchdog, name="bot-watchdog", daemon=True)
    wd.start()
    log.info("Watchdog started. Med-Clear is live.")

    # 3. Keep the main thread alive (required — if main exits, all daemons die)
    while True:
        time.sleep(60)
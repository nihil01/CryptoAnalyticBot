import logging
import time

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
import os

from api_requests import summarize_crypto

logging.basicConfig(
    filename="logger.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
load_dotenv()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я шарю за крипту, ты можешь послать команду /analyze (НАЗВАНИЕ_МОНЕТЫ, например, BTC, XRP, XLM ..),"
                                    " и я отправлю тебе сводку по этой монете, чтобы ты не был лошком")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if message is None or message.text is None or len(message.text) == 0:
        await update.message.reply_text("Ты че, клоун? Отправь название монеты, мудила -_-")
        return

    await update.message.reply_text("Ща подумаю ...")
    result = summarize_crypto(message.text[len("/analyze"):].strip())

    if result is None:
        await update.message.reply_text("Получилась хуйня !!!")
        return

    await update.message.reply_text(result, parse_mode="Markdown", disable_web_page_preview=True)

def main():
    token: str = os.getenv("TG_BOT_API_KEY")
    application = Application.builder().token(token).build()
    logger.log(msg=f"APPLICATION STARTED WITH TOKEN {token} on {time.asctime(time.localtime(time.time()))}", level=logging.INFO)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze))

    application.run_polling()

if __name__ == "__main__":
    main()

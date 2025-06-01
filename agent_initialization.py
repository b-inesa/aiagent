from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import logging, os
from data_processing import Data
from llm_connection import LLM
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

BOT_KEY = os.getenv("BOT_KEY")
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST")
WEBHOOK_PATH = f"/webhook/{BOT_KEY}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

bot = ApplicationBuilder().token(BOT_KEY).build()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent:
    def __init__(self, data: Data):
        self.data = data
        self.users = {}        

    def get_user_chain(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = LLM().create_chain()
        return self.users[user_id]

    async def start(self, update: Update):
        await update.message.reply_text("Привіт! Я допоможу тобі знайти відповіді на питання щодо вступу до КПІ на ФІОТ. \n? напиши своє питання:")
    
    async def handle_question(self, update: Update):
        user_id = update.message.chat.id
        question = update.message.text
        chain = self.get_user_chain(user_id)

        try:
            doc_context = await self.data.get_context(question)    
            prompt = question + doc_context
            answer = await chain.ainvoke({"human_input": prompt})
            await update.message.reply_text(answer["text"])
        except Exception as e:
            logging.error(f"Виникла помилка при обробці запитання: {question} - {e}")
            await update.message.reply_text("Сталася помилка. Спробуйте ще раз.")

data = Data()
data.get_pdf_data()
data.vectorize_documents()
agent = Agent(data)

bot.add_handler(CommandHandler("start", agent.start))
bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agent.handle_question))

@asynccontextmanager
async def lifespan(app: FastAPI):    
    await bot.initialize()
    await bot.start()
    await bot.bot.set_webhook(WEBHOOK_URL)
    print(f"✅ Webhook встановлено: {WEBHOOK_URL}")
    yield

app = FastAPI(lifespan=lifespan)

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot.bot)
    await bot.update_queue.put(update)
    return {"ok": True}
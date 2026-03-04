import os
import logging
import aiohttp
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_helper import RAGSearch

# ── Настройки ────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# ── Логирование ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Загрузка RAG базы знаний ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rag = RAGSearch(os.path.join(BASE_DIR, "knowledge_base_rag.json"))
logger.info("✅ База знаний загружена")

# ── Системный промпт ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты — помощник службы поддержки платформы ATI.SU (биржа грузоперевозок).

Твоя задача — отвечать на вопросы пользователей СТРОГО на основе предоставленного контекста из базы знаний.

Правила:
1. Отвечай только по контексту. Если ответа в контексте нет — скажи: «К сожалению, у меня нет информации по этому вопросу. Пожалуйста, обратитесь в поддержку ATI.SU.»
2. Отвечай на русском языке, вежливо и по делу.
3. Не придумывай информацию, не дополняй от себя.
4. Если вопрос касается технических проблем с аккаунтом — предложи обратиться в поддержку.
5. Структурируй ответ: используй нумерованные списки или абзацы где уместно."""


# ── Запрос к Яндекс GPT ──────────────────────────────────────────────────────
async def ask_yandex_gpt(question: str, context: str) -> str:
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }

    user_message = f"""Контекст из базы знаний ATI.SU:
{context}

Вопрос пользователя: {question}"""

    payload = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 2000
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": user_message}
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(YANDEX_GPT_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Yandex GPT error {resp.status}: {error_text}")
                return "Произошла ошибка при обращении к ИИ. Попробуйте позже."
            data = await resp.json()
            return data["result"]["alternatives"][0]["message"]["text"]


# ── Обработчики Telegram ─────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я помощник ATI.SU.\n\n"
        "Задайте мне вопрос о работе платформы — о грузах, транспорте, "
        "лицензиях, индексе ставок, АТИ-Доках и других сервисах.\n\n"
        "Я отвечу на основе официальной базы знаний ATI.SU."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Я могу ответить на вопросы о:\n"
        "• Грузах и транспорте на бирже\n"
        "• Индексе ATI.SU и средних ставках\n"
        "• АТИ-Доках (электронный документооборот)\n"
        "• Лицензиях и тарифах\n"
        "• Тендерах и площадках\n"
        "• Рейтинге и репутации участников\n\n"
        "Просто напишите свой вопрос!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    question = update.message.text.strip()

    if not question:
        return

    logger.info(f"Вопрос от {user.id} (@{user.username}): {question[:100]}")

    # Показать "печатает..."
    await update.message.chat.send_action("typing")

    try:
        # 1. Найти релевантные чанки
        context_text = rag.get_context(question, top_k=8)

        # 2. Отправить в Яндекс GPT
        answer = await ask_yandex_gpt(question, context_text)

        # 3. Ответить пользователю
        await update.message.reply_text(answer)
        logger.info(f"Ответ отправлен пользователю {user.id}")

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Произошла ошибка. Пожалуйста, попробуйте ещё раз или обратитесь в поддержку ATI.SU."
        )


# ── Запуск ───────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env")
    if not YANDEX_API_KEY:
        raise ValueError("YANDEX_API_KEY не задан в .env")
    if not YANDEX_FOLDER_ID:
        raise ValueError("YANDEX_FOLDER_ID не задан в .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Бот запущен")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

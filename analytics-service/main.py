from fastapi import FastAPI
from dotenv import load_dotenv
from routes import analysis, chat

load_dotenv()
app = FastAPI(
    title="Terrafy Analytics",
    description=(
        "Agronomic analytics & AI service. "
        "Historic data, alerts, and chatbot for hydroponic growing systems."
    ),
    version="2.0.0",
)

app.include_router(analysis.router, prefix="/analysis")
app.include_router(chat.router,       prefix="/ai")


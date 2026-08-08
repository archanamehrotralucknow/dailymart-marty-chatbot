from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = Exception
    BaseModel = object

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))


class Intent:
    GREETING = "greeting"
    PRODUCT_SEARCH = "product_search"
    PRICE_DEAL = "price_deal"
    AFFILIATE_FAQ = "affiliate_faq"
    ORDER_TRACKING = "order_tracking"
    HUMAN_HANDOFF = "human_handoff"
    SMALLTALK = "smalltalk"
    FALLBACK = "fallback"


PATTERNS = {
    Intent.GREETING: re.compile(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", re.I),
    Intent.ORDER_TRACKING: re.compile(r"\b(order status|track order|where.*order|order.*tracking|shipment|delivery status|where is my order)\b", re.I),
    Intent.AFFILIATE_FAQ: re.compile(r"\b(affiliate|commission|cashback|cash back|partner site|partner merchant|earn.*commission|affiliate.*link)\b", re.I),
    Intent.HUMAN_HANDOFF: re.compile(r"\b(agent|human|support|customer service|someone else|contact support|real person)\b", re.I),
    Intent.PRICE_DEAL: re.compile(r"\b(price|deal|discount|cheapest|cheaper|compare|best price|sale|offer|offers|budget)\b", re.I),
    Intent.PRODUCT_SEARCH: re.compile(r"\b(search|find|look for|looking for|recommend|show me|want|need|buy|purchase|available|stock)\b", re.I),
    Intent.SMALLTALK: re.compile(r"\b(thanks|thank you|bye|goodbye|see you|nice|cool|how are you|what's up)\b", re.I),
}

SYNONYM_MAP = {
    "ac": "air conditioner",
    "aircon": "air conditioner",
    "tv": "television",
}

STOPWORDS = {"i", "want", "need", "search", "find", "show", "me", "for", "the", "a", "an", "in", "on", "of", "to", "buy", "purchase"}

PRODUCT_CATALOG = [
    {"title": "Noise Cancelling Wireless Earbuds", "price": "₹3,499", "merchant": "Amazon", "affiliate_url": "https://affiliate.amazon.in/noise-earbuds", "rating": "4.5/5"},
    {"title": "Smart LED TV 43-inch Full HD", "price": "₹19,999", "merchant": "Flipkart", "affiliate_url": "https://affiliate.flipkart.com/43-inch-tv", "rating": "4.3/5"},
    {"title": "Fitness Running Shoes for Men", "price": "₹2,199", "merchant": "Amazon", "affiliate_url": "https://affiliate.amazon.in/running-shoes", "rating": "4.2/5"},
    {"title": "Espresso Coffee Maker", "price": "₹7,499", "merchant": "Flipkart", "affiliate_url": "https://affiliate.flipkart.com/espresso-maker", "rating": "4.6/5"},
    {"title": "USB-C Fast Charger 65W", "price": "₹1,299", "merchant": "Amazon", "affiliate_url": "https://affiliate.amazon.in/65w-charger", "rating": "4.4/5"},
    {"title": "Bluetooth Soundbar with Subwoofer", "price": "₹5,299", "merchant": "Flipkart", "affiliate_url": "https://affiliate.flipkart.com/soundbar-subwoofer", "rating": "4.1/5"},
    {"title": "Smartphone Glass Screen Protector", "price": "₹249", "merchant": "Amazon", "affiliate_url": "https://affiliate.amazon.in/screen-protector", "rating": "4.0/5"},
    {"title": "Noise Cancelling Over-Ear Headphones", "price": "₹4,999", "merchant": "Flipkart", "affiliate_url": "https://affiliate.flipkart.com/over-ear-headphones", "rating": "4.4/5"},
    {"title": "1.5 Ton Split Air Conditioner", "price": "₹34,999", "merchant": "Amazon", "affiliate_url": "https://affiliate.amazon.in/split-ac", "rating": "4.3/5"},
    {"title": "Window Air Conditioner 1 Ton", "price": "₹24,499", "merchant": "Flipkart", "affiliate_url": "https://affiliate.flipkart.com/window-ac", "rating": "4.1/5"},
]

SYSTEM_PROMPT = (
    "You are Marty, Dailymart's shopping assistant. Be friendly, concise, and transparent. "
    "Always disclose that Dailymart earns commission when sharing affiliate links. "
    "Do not invent prices, stock, or inventory. Ground your answers only in the provided catalog data or FAQ guidance. "
    "If a user asks about order tracking, explain that tracking is handled by the partner merchant site."
)


def classify_intent(message: str) -> str:
    normalized = (message or "").strip().lower()
    if not normalized:
        return Intent.FALLBACK
    for intent in (
        Intent.ORDER_TRACKING,
        Intent.AFFILIATE_FAQ,
        Intent.HUMAN_HANDOFF,
        Intent.PRICE_DEAL,
        Intent.PRODUCT_SEARCH,
        Intent.GREETING,
        Intent.SMALLTALK,
    ):
        if PATTERNS[intent].search(normalized):
            return intent
    return Intent.FALLBACK


def query_variants(query: str) -> List[str]:
    normalized = (query or "").strip().lower()
    if not normalized:
        return []
    variants = [normalized]
    for token in re.findall(r"\w+", normalized):
        mapped = SYNONYM_MAP.get(token)
        if mapped and mapped not in variants:
            variants.append(mapped)
    return variants


def search_products(query: str) -> List[Dict[str, Any]]:
    variants = query_variants(query)
    if not variants:
        return PRODUCT_CATALOG[:5]
    tokens = {t for t in re.findall(r"\w+", " ".join(variants)) if len(t) > 2 and t not in STOPWORDS}
    scored = []
    for item in PRODUCT_CATALOG:
        haystack = f"{item['title']} {item['merchant']}".lower()
        score = sum(3 for v in variants if re.search(r"\b" + re.escape(v) + r"\b", haystack))
        score += sum(1 for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", haystack))
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored[:5]]


def available_models() -> List[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [model["name"] for model in data.get("models", []) if model.get("name")]
    except Exception:
        return []


def ask_ollama(messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[str]:
    model = model or OLLAMA_MODEL
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.7}}
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip() or None
    except urllib.error.HTTPError as error:
        if error.code == 404 and model == OLLAMA_MODEL:
            fallback = next(iter(available_models()), None)
            if fallback and fallback != model:
                return ask_ollama(messages, fallback)
        return None
    except Exception:
        return None


def generate_reply(history: List[Dict[str, str]], context: str, user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context: {context or 'No recent catalog or FAQ context available.'}"},
    ]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})
    reply = ask_ollama(messages)
    if reply:
        return reply
    return "I can't reach the local AI right now. Please make sure Ollama is running with a model pulled, then try again."


@dataclass
class ProductCard:
    title: str
    price: str
    merchant: str
    rating: Optional[str]
    affiliate_url: str


class ChatBot:
    def __init__(self, max_memory: int = 6):
        self.max_memory = max_memory
        self.memory: List[Dict[str, str]] = []

    def handle_message(self, message: str) -> Dict[str, Any]:
        intent = classify_intent(message)
        product_cards: List[ProductCard] = []
        affiliate_disclosure: Optional[str] = None

        if intent in (Intent.GREETING, Intent.AFFILIATE_FAQ, Intent.ORDER_TRACKING, Intent.HUMAN_HANDOFF, Intent.SMALLTALK):
            reply_text = self._template_reply(intent)
        elif intent in (Intent.PRODUCT_SEARCH, Intent.PRICE_DEAL):
            product_cards = self._product_cards(message)
            if product_cards:
                affiliate_disclosure = "These are affiliate links. Dailymart earns commission when you click through and buy from a partner merchant."
                reply_text = self._product_reply(product_cards, intent)
            else:
                reply_text = generate_reply(self.memory, self._no_match_context(), message)
        else:
            reply_text = generate_reply(self.memory, self._context(message), message)

        self._remember("user", message)
        self._remember("assistant", reply_text)
        return {
            "intent": intent,
            "reply_text": reply_text,
            "product_cards": [card.__dict__ for card in product_cards],
            "affiliate_disclosure": affiliate_disclosure,
        }

    def _template_reply(self, intent: str) -> str:
        replies = {
            Intent.GREETING: "Hi! I'm Marty, Dailymart's shopping assistant. I can help you discover products, compare partner deals, and explain affiliate links.",
            Intent.AFFILIATE_FAQ: "Dailymart curates product listings from partner merchants like Amazon and Flipkart. When you click an affiliate link and complete a purchase on the partner site, Dailymart earns a commission. We always disclose that it's an affiliate link.",
            Intent.ORDER_TRACKING: "Order tracking lives on the partner merchant's site because Dailymart does not sell or ship products directly. If you need help finding your Amazon or Flipkart order status, I can guide you to the right place.",
            Intent.HUMAN_HANDOFF: "I'm happy to help with product discovery. For account or order support, please contact the merchant's customer service team.",
            Intent.SMALLTALK: "I'm here to help you find the best product deals from partner merchants. What are you shopping for today?",
        }
        return replies.get(intent, "How can I help you with shopping or deals today?")

    def _product_cards(self, query: str) -> List[ProductCard]:
        return [
            ProductCard(
                title=item["title"],
                price=item["price"],
                merchant=item["merchant"],
                rating=item.get("rating"),
                affiliate_url=item["affiliate_url"],
            )
            for item in search_products(query)
        ]

    def _product_reply(self, cards: List[ProductCard], intent: str) -> str:
        if intent == Intent.PRICE_DEAL:
            return f"I found {len(cards)} deals for you. Here are current offers from partner merchants. Each result includes an affiliate link, and Dailymart earns commission if you purchase."
        return f"I found {len(cards)} matching products. These affiliate links take you to the merchant site where you can complete the purchase."

    def _no_match_context(self) -> str:
        titles = ", ".join(item["title"] for item in PRODUCT_CATALOG)
        return f"The affiliate catalog has no item matching the request. Only these products are available: {titles}. Do not invent products; if nothing fits, say so and suggest the closest available category."

    def _context(self, message: str) -> str:
        recent = [turn["content"] for turn in self.memory if turn["role"] == "assistant"][-2:]
        return f"Recent assistant context: {' '.join(recent)}. User asked: {message}."

    def _remember(self, role: str, content: str) -> None:
        if not content:
            return
        self.memory.append({"role": role, "content": content})
        limit = self.max_memory * 2
        if len(self.memory) > limit:
            self.memory = self.memory[-limit:]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    reply_text: str
    product_cards: List[Dict[str, Any]] = []
    affiliate_disclosure: Optional[str] = None


app = FastAPI(title="Dailymart Marty Single") if FastAPI else None
bot = ChatBot()

if app is not None:
    @app.post("/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest) -> ChatResponse:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required")
        return ChatResponse(**bot.handle_message(request.message))


def run_cli() -> None:
    print("Dailymart Marty local CLI. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if not message:
            continue
        response = bot.handle_message(message)
        print(f"Marty: {response['reply_text']}\n")
        for index, card in enumerate(response["product_cards"], start=1):
            print(f"{index}. {card['title']} | {card['merchant']} | {card['price']} | {card['rating']} | {card['affiliate_url']}")
        if response["affiliate_disclosure"]:
            print(f"\n{response['affiliate_disclosure']}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"serve", "api"}:
        try:
            import uvicorn
            uvicorn.run("marty_single:app", host="127.0.0.1", port=8000, reload=True)
        except ImportError:
            print("uvicorn not available. Run: uvicorn marty_single:app --reload")
    else:
        run_cli()

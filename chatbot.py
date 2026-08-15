from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = Exception
    BaseModel = object


def _load_env_file(path: Path) -> None:
    """Minimal .env loader so the CLI works without python-dotenv installed."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file(Path(__file__).resolve().parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))

_client_cache: Dict[tuple, genai.Client] = {}


def get_client() -> Optional[genai.Client]:
    """Return a cached client for the current key, or None if unconfigured."""
    if not GEMINI_API_KEY:
        return None
    cache_key = (GEMINI_API_KEY, GEMINI_TIMEOUT)
    client = _client_cache.get(cache_key)
    if client is None:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            # google-genai expects milliseconds.
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000),
        )
        _client_cache[cache_key] = client
    return client


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


def _model_names(client: genai.Client) -> List[str]:
    # The API returns fully qualified names like "models/gemini-3.7-flash".
    return sorted(
        (model.name or "").removeprefix("models/")
        for model in client.models.list()
        if model.name
    )


def available_models() -> List[str]:
    client = get_client()
    if client is None:
        return []
    try:
        return _model_names(client)
    except Exception:
        return []


def check_connection() -> Dict[str, Any]:
    """Probe the Gemini API with the configured key. Returns reachability, models, and a reason."""
    client = get_client()
    if client is None:
        return {"ok": False, "models": [], "reason": "No GEMINI_API_KEY configured"}
    try:
        return {"ok": True, "models": _model_names(client), "reason": ""}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "models": [], "reason": str(error)}


LAST_ERROR = ""


def to_gemini_contents(messages: List[Dict[str, str]]) -> List[types.Content]:
    """Convert the internal role/content turns to Gemini contents ('assistant' becomes 'model')."""
    contents = []
    for message in messages:
        text = message.get("content") or ""
        if not text:
            continue
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def ask_gemini(
    messages: List[Dict[str, str]],
    system_instruction: str = SYSTEM_PROMPT,
    model: Optional[str] = None,
) -> Optional[str]:
    global LAST_ERROR
    client = get_client()
    if client is None:
        LAST_ERROR = "No GEMINI_API_KEY configured"
        return None
    try:
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=to_gemini_contents(messages),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
        )
    except Exception as error:  # noqa: BLE001
        LAST_ERROR = str(error)
        return None
    LAST_ERROR = ""
    return (response.text or "").strip() or None


def generate_reply(history: List[Dict[str, str]], context: str, user_message: str) -> str:
    # Gemini takes system text separately, so the context line rides along with the persona.
    system_instruction = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context: {context or 'No recent catalog or FAQ context available.'}"
    )
    messages = list(history[-6:])
    messages.append({"role": "user", "content": user_message})
    reply = ask_gemini(messages, system_instruction)
    if reply:
        return reply
    if not GEMINI_API_KEY:
        return "I'm not configured yet — set GEMINI_API_KEY in .env (or Streamlit secrets) and try again."
    if "RESOURCE_EXHAUSTED" in LAST_ERROR or "429" in LAST_ERROR:
        return "My Gemini quota is exhausted right now. Check your usage limits at aistudio.google.com, then try again."
    if "API_KEY_INVALID" in LAST_ERROR or "API key not valid" in LAST_ERROR:
        return "My Gemini API key was rejected. Please check GEMINI_API_KEY and try again."
    if "NOT_FOUND" in LAST_ERROR or "not found" in LAST_ERROR:
        return f"The model '{GEMINI_MODEL}' isn't available to this API key. Pick a different model in Settings."
    return "I can't reach Gemini right now. Please check your network and try again."


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

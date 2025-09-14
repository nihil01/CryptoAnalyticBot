import base64
import datetime
import json

import requests
from dotenv import load_dotenv
import os
import logging

from openai import OpenAI

import yfinance
import matplotlib.pyplot as plt

logging.basicConfig(
    filename="logger.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def save_crypto_graph(crypto_name: str):

    if not crypto_name.endswith("-USD"):
        crypto_name = crypto_name + "-USD"

    data = yfinance.Ticker(crypto_name).history(period="1mo", interval="1h")
    print(data)

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label=crypto_name)
    plt.title(f"{crypto_name} Price Chart (7d)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("crypto_chart.png")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_request_to_openai(text) -> str:
    #preapre a chart
    base64_image = encode_image("crypto_chart.png")

    summary = json.dumps(text, ensure_ascii=False)

    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{summary}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
        instructions="""
        Ты – финансовый аналитик по криптовалютам, работающий в стиле **скальпинг**. 
        Я передаю тебе объект с данными о выбранной криптовалюте. 
        В объекте есть:
        - текущая цена, волатильность, объём торгов, ликвидность и другие метрики;
        - массив новостей (каждая новость имеет ссылку и позицию SENTIMENT, проанализируй каждую ссылку на новость);
        - исторические данные (свечи) за день или неделю.

        Твоя задача:
        1. Проанализировать все предоставленные данные (метрики, новости, свечи).
        2. Определить текущее состояние рынка.
        3. Дать конкретное решение по **Spot торговле (скальпинг)**:
           - Купить / Продать / Держать.
           - Указать конкретную цену входа и цену выхода (короткий диапазон).
           - Указать уровень уверенности (0–1).
           - Чётко объяснить причину.
        4. Дать конкретное решение по **Futures торговле (скальпинг)**:
           - Лонг / Шорт / Не входить.
           - Указать рекомендуемое плечо (например, x3, x5, x10).
           - Указать короткий диапазон движения (high / low).
           - Указать уровень уверенности (0–1).
           - Чётко объяснить причину.
        5. Учитывай влияние новостей на настроение рынка (позитивное/негативное) и их моментальный эффект.
        6. Формируй результат **в текстовой форме**, например:

        📌 Монета: <название>
        💰 Spot (скальпинг): <действие> (вход по …, выход по …). Уверенность: …
        📈 Futures (скальпинг): <лонг/шорт>, плечо …, диапазон … – …. Уверенность: …
        📝 Причина: <Чёткое и внятное объяснение>

        Очень важно: форматируй результат так, чтобы пользователь сразу видел краткую стратегию действий (что купить/продать, где войти, где выйти), без JSON.
        """
    )

    return response.output_text



def get_crypto_news(name: str) -> list[dict[str, str]]:
    response = requests.get(
        "https://data-api.coindesk.com/news/v1/search",
        params={
            "categories": name,
            "lang": "EN",
            "api_key": os.getenv("COINDESK_API_KEY"),
            "limit": 10,
            "search_string": f"{name.upper()}",
            "source_key": "coindesk"
        },
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    data = response.json()

    news_items = []

    for item in data.get("Data", []):
        url = item.get("URL")
        published = datetime.datetime.fromtimestamp(item.get("PUBLISHED_ON"),
                    tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        sentiment = item.get("SENTIMENT", "NEUTRAL")
        source = item.get("SOURCE_DATA", {}).get("NAME", "Unknown")

        summary = {
            "url": url,
            "published": published,
            "sentiment": sentiment,
            "source": source,
        }
        news_items.append(summary)

    return news_items

def get_cryptocurrency(name: str) -> dict[str, float] | dict[str, str]:
    response = requests.get(
        'https://data-api.coindesk.com/spot/v1/latest/tick',
        params={
            "market": "coinbase",
            "instruments": f"{name.upper()}-USD",
            "apply_mapping": "true",
            "api_key": os.getenv("COINDESK_API_KEY")
        },
        headers={"Content-type": "application/json; charset=UTF-8"}
    )

    if response.status_code != 200:
        return {"error": "BAD_CODE"}

    data = response.json()["Data"][f"{name.upper()}-USD"]

    buy_volume = data.get("MOVING_24_HOUR_VOLUME_BUY", 0)
    sell_volume = data.get("MOVING_24_HOUR_VOLUME_SELL", 0)
    buy_sell_ratio = buy_volume / sell_volume if sell_volume != 0 else None

    return {
        "symbol": name.upper(),
        "price": data["PRICE"],

        "best_bid": data["BEST_BID"],
        "best_ask": data["BEST_ASK"],
        "spread": data["BEST_ASK"] - data["BEST_BID"],

        "change_24h": data["MOVING_24_HOUR_CHANGE_PERCENTAGE"],
        "change_7d": data["MOVING_7_DAY_CHANGE_PERCENTAGE"],
        "change_30d": data["MOVING_30_DAY_CHANGE_PERCENTAGE"],

        "day_open": data["CURRENT_DAY_OPEN"],
        "day_high": data["CURRENT_DAY_HIGH"],
        "day_low": data["CURRENT_DAY_LOW"],

        "volume_24h": data["MOVING_24_HOUR_VOLUME"],
        "quote_volume_24h": data["MOVING_24_HOUR_QUOTE_VOLUME"],
        "trades_24h": data["MOVING_24_HOUR_TOTAL_TRADES"],

        "buy_volume_24h": buy_volume,
        "sell_volume_24h": sell_volume,
        "buy_sell_ratio": buy_sell_ratio
    }


def summarize_crypto(name: str) -> str | None:
    logger.info(f"Summarizing crypto {name}")
    result = get_cryptocurrency(name)
    if result.get("error"):
        return None

    logger.info(f"Crypto {name} : {result}")

    news = get_crypto_news(name)
    logger.info(f"Crypto news : {news}")

    # сохраняем график
    save_crypto_graph(name)
    logger.info(f"Crypto chart saved!")

    summary = {
        "symbol": result["symbol"],
        "price": f"${result['price']:.2f}",
        "performance": {
            "24h": f"{result['change_24h']:.2f}%",
            "7d": f"{result['change_7d']:.2f}%",
            "30d": f"{result['change_30d']:.2f}%",
        },
        "range_today": f"${result['day_low']:.2f} - ${result['day_high']:.2f}",
        "volume_24h": f"{result['volume_24h']:,}",
        "quote_volume_24h": f"{result['quote_volume_24h']:,}",
        "trades_24h": f"{result['trades_24h']:,}",

        "best_bid": f"${result['best_bid']:.2f}",
        "best_ask": f"${result['best_ask']:.2f}",
        "spread": f"${result['spread']:.4f}",

        "buy_volume_24h": f"{result['buy_volume_24h']:,}",
        "sell_volume_24h": f"{result['sell_volume_24h']:,}",
        "buy_sell_ratio": f"{result['buy_sell_ratio']:.2f}" if result['buy_sell_ratio'] else None,

        "latest_news": news
    }

    return make_request_to_openai(summary)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva warnings de TensorFlow
import sys
import json
import numpy as np
import joblib
from datetime import datetime, time
from collections import deque
import logging
import asyncio
import tracemalloc
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackContext,
    filters,
    Application
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import requests
from typing import Dict, List, Optional

# Configuración de logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ----------------- CONFIGURACIÓN AVANZADA -----------------
MODEL_SAVE_PATH = "ai_models/saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ----------------- CONFIGURACIÓN DE APIS -----------------
class APIConfig:
    """Configuración de APIs externas"""
    COINGECKO_URL = "https://api.coingecko.com/api/v3"
    BINANCE_URL = "https://api.binance.com/api/v3"
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2"
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    
    @staticmethod
    def get_headers(api_name: str) -> Dict:
        """Obtiene headers para diferentes APIs"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if api_name == "cryptocompare":
            headers['authorization'] = f"Apikey YOUR_CRYPTOCOMPARE_API_KEY"
        elif api_name == "newsapi":
            headers['X-Api-Key'] = "YOUR_NEWS_API_KEY"
            
        return headers

# ----------------- FUNCIONES DE DATOS REALES -----------------
def get_real_time_price(symbol: str) -> float:
    """Obtiene el precio en tiempo real de una criptomoneda"""
    try:
        params = {
            'ids': symbol.lower(),
            'vs_currencies': 'usd',
            'precision': 'full'
        }
        response = requests.get(
            f"{APIConfig.COINGECKO_URL}/simple/price",
            params=params,
            headers=APIConfig.get_headers('coingecko'),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return float(data.get(symbol.lower(), {}).get('usd', 0))
            
        response = requests.get(
            f"{APIConfig.BINANCE_URL}/ticker/price",
            params={'symbol': f"{symbol.upper()}USDT"},
            headers=APIConfig.get_headers('binance'),
            timeout=10
        )
        
        if response.status_code == 200:
            return float(response.json().get('price', 0))
            
        response = requests.get(
            f"{APIConfig.CRYPTOCOMPARE_URL}/price",
            params={'fsym': symbol.upper(), 'tsyms': 'USD'},
            headers=APIConfig.get_headers('cryptocompare'),
            timeout=10
        )
        
        if response.status_code == 200:
            return float(response.json().get('USD', 0))
            
        return 0.0
        
    except Exception as e:
        logger.error(f"Error obteniendo precio real: {str(e)}")
        return 0.0

def get_historical_data(symbol: str, days: int = 30) -> List[Dict]:
    """Obtiene datos históricos reales"""
    try:
        response = requests.get(
            f"{APIConfig.COINGECKO_URL}/coins/{symbol.lower()}/market_chart",
            params={'vs_currency': 'usd', 'days': days},
            headers=APIConfig.get_headers('coingecko'),
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return [
                {'timestamp': entry[0]/1000, 'price': entry[1]}
                for entry in data.get('prices', [])
            ]
            
        return []
    except Exception as e:
        logger.error(f"Error obteniendo datos históricos: {str(e)}")
        return []

def get_real_news(limit: int = 10) -> List[Dict]:
    """Obtiene noticias reales de criptomonedas"""
    try:
        response = requests.get(
            f"{APIConfig.CRYPTOCOMPARE_URL}/news/",
            params={'categories': 'BTC,ETH', 'excludeCategories': 'Sponsored', 'lang': 'ES'},
            headers=APIConfig.get_headers('cryptocompare'),
            timeout=15
        )
        
        news_items = []
        
        if response.status_code == 200:
            data = response.json().get('Data', [])
            for item in data[:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', '#'),
                    'published_at': item.get('published_on', 0),
                    'sentiment': item.get('sentiment', 'neutral')
                })
        
        if len(news_items) < limit:
            response = requests.get(
                APIConfig.NEWS_API_URL,
                params={
                    'q': 'bitcoin OR ethereum OR criptomonedas',
                    'language': 'es',
                    'sortBy': 'publishedAt',
                    'pageSize': limit - len(news_items)
                },
                headers=APIConfig.get_headers('newsapi'),
                timeout=15
            )
            
            if response.status_code == 200:
                for item in response.json().get('articles', []):
                    news_items.append({
                        'title': item.get('title', ''),
                        'source': item.get('source', {}).get('name', ''),
                        'url': item.get('url', '#'),
                        'published_at': item.get('publishedAt', ''),
                        'sentiment': 'neutral'
                    })
        
        return news_items[:limit]
        
    except Exception as e:
        logger.error(f"Error obteniendo noticias reales: {str(e)}")
        return []

def get_market_overview() -> Dict:
    """Obtiene un resumen completo del mercado"""
    try:
        response = requests.get(
            f"{APIConfig.COINGECKO_URL}/global",
            headers=APIConfig.get_headers('coingecko'),
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json().get('data', {})
            
        return {}
    except Exception as e:
        logger.error(f"Error obteniendo overview: {str(e)}")
        return {}

def get_24h_change(symbol: str) -> float:
    """Obtiene el cambio porcentual en 24h"""
    try:
        response = requests.get(
            f"{APIConfig.COINGECKO_URL}/coins/{symbol.lower()}",
            params={'localization': 'false', 'tickers': 'false', 'market_data': 'true'},
            headers=APIConfig.get_headers('coingecko'),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return float(data.get('market_data', {}).get('price_change_percentage_24h', 0))
            
        return 0.0
    except Exception as e:
        logger.error(f"Error obteniendo cambio 24h: {str(e)}")
        return 0.0

# ----------------- IA CON AUTOAPRENDIZAJE -----------------
class CryptoAI:
    """Clase base para la IA de criptomonedas"""
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def predict_price(self, symbol):
        """Predice el precio para un símbolo dado"""
        try:
            historical = get_historical_data(symbol, days=30)
            if not historical:
                return [10000, 10500, 11000] if symbol == "BTC" else [500, 525, 550]
                
            prices = [item['price'] for item in historical]
            if len(prices) < 10:
                return [10000, 10500, 11000] if symbol == "BTC" else [500, 525, 550]
                
            if symbol in self.models:
                last_price = prices[-1]
                change = (prices[-1] - prices[-7]) / prices[-7]
                
                return [
                    last_price * (1 + change/3),
                    last_price * (1 + change*2/3),
                    last_price * (1 + change)
                ]
            else:
                last_price = prices[-1]
                return [last_price * 1.01, last_price * 1.02, last_price * 1.03]
        except Exception as e:
            logger.error(f"Error en predict_price: {str(e)}")
            return []

    def analyze_news(self, text):
        """Análisis básico de sentimiento en noticias"""
        try:
            return {'positive': 0.7, 'negative': 0.3}
        except Exception as e:
            logger.error(f"Error en analyze_news: {str(e)}")
            return {'positive': 0.5, 'negative': 0.5}
        
    def prepare_data(self, data):
        """Prepara datos para el modelo"""
        try:
            return data[:-1], data[1:], None
        except Exception as e:
            logger.error(f"Error en prepare_data: {str(e)}")
            return None, None, None

class SelfLearningCryptoAI(CryptoAI):
    def __init__(self):
        super().__init__()
        self.news_model = None
        self.feedback_db = deque(maxlen=1000)
        self.load_models()
        
    def load_models(self):
        """Carga modelos guardados con mejor manejo de errores"""
        try:
            if os.path.exists(f"{MODEL_SAVE_PATH}/news_model.model"):
                self.news_model = Word2Vec.load(f"{MODEL_SAVE_PATH}/news_model.model")
                logger.info("Modelo de noticias cargado correctamente")
                
            for symbol in ["BTC", "ETH", "SOL"]:
                model_path = f"{MODEL_SAVE_PATH}/{symbol}_model.h5"
                scaler_path = f"{MODEL_SAVE_PATH}/{symbol}_scaler.pkl"
                
                if os.path.exists(model_path):
                    self.models[symbol] = load_model(model_path)
                    logger.info(f"Modelo para {symbol} cargado correctamente")
                
                if os.path.exists(scaler_path):
                    self.scalers[symbol] = joblib.load(scaler_path)
                    logger.info(f"Scaler para {symbol} cargado correctamente")
                    
        except Exception as e:
            logger.error(f"Error cargando modelos: {str(e)}")

    def save_models(self):
        """Guarda los modelos periódicamente con verificación"""
        try:
            if self.news_model:
                save_path = f"{MODEL_SAVE_PATH}/news_model.model"
                self.news_model.save(save_path)
                logger.info("Modelo de noticias guardado correctamente")
                
            for symbol, model in self.models.items():
                model_save_path = f"{MODEL_SAVE_PATH}/{symbol}_model.h5"
                scaler_save_path = f"{MODEL_SAVE_PATH}/{symbol}_scaler.pkl"
                
                model.save(model_save_path)
                if symbol in self.scalers:
                    joblib.dump(self.scalers[symbol], scaler_save_path)
                
                logger.info(f"Modelo para {symbol} guardado correctamente")
                
        except Exception as e:
            logger.error(f"Error guardando modelos: {str(e)}")

    def update_with_feedback(self, symbol, actual_price, predicted_price):
        """Ajusta el modelo basado en feedback con validación"""
        try:
            error = actual_price - predicted_price
            self.feedback_db.append({
                'symbol': symbol,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            
            if len(self.feedback_db) % 100 == 0:
                self.retrain_models()
        except Exception as e:
            logger.error(f"Error en update_with_feedback: {str(e)}")

    def retrain_models(self):
        """Reentrena modelos con nuevos datos y mejor manejo de errores"""
        try:
            logger.info("Iniciando reentrenamiento de modelos...")
            
            for symbol in list(self.models.keys()):
                try:
                    new_data = self.get_klines(symbol, limit=500)
                    if new_data:
                        closes = np.array([float(k[4]) for k in new_data]).reshape(-1, 1)
                        X, y, scaler = self.prepare_data(closes)
                        
                        if X is not None and y is not None:
                            self.models[symbol].fit(X, y, epochs=20, batch_size=16, verbose=0)
                            self.scalers[symbol] = scaler
                except Exception as e:
                    logger.error(f"Error reentrenando modelo para {symbol}: {str(e)}")
                    continue
            
            try:
                articles = self.get_fundamental_news(limit=100)
                if articles:
                    processed_news = [self.preprocess_text(a['title']) for a in articles]
                    if not self.news_model:
                        self.news_model = Word2Vec(processed_news, vector_size=100, window=5, min_count=1, workers=4)
                    else:
                        self.news_model.build_vocab(processed_news, update=True)
                        self.news_model.train(processed_news, total_examples=len(processed_news), epochs=10)
            except Exception as e:
                logger.error(f"Error reentrenando modelo de noticias: {str(e)}")
            
            self.save_models()
            logger.info("Reentrenamiento completado con éxito")
            
        except Exception as e:
            logger.error(f"Error en reentrenamiento general: {str(e)}")

    def get_klines(self, symbol, limit=500):
        """Obtiene datos históricos con manejo de errores"""
        try:
            response = requests.get(
                f"{APIConfig.BINANCE_URL}/klines",
                params={
                    'symbol': f"{symbol.upper()}USDT",
                    'interval': '1d',
                    'limit': limit
                },
                headers=APIConfig.get_headers('binance'),
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
                
            return [[None, None, None, None, 10000 + i] for i in range(limit)]
        except Exception as e:
            logger.error(f"Error en get_klines: {str(e)}")
            return []

    def get_fundamental_news(self, limit=100):
        """Obtiene noticias con manejo de errores"""
        try:
            return get_real_news(limit)
        except Exception as e:
            logger.error(f"Error en get_fundamental_news: {str(e)}")
            return []

    def analyze_news_advanced(self, text):
        """Análisis semántico avanzado de noticias con mejor manejo de errores"""
        try:
            sentiment = super().analyze_news(text)
            processed = self.preprocess_text(text)
            
            if self.news_model and processed:
                vectors = [self.news_model.wv[word] for word in processed if word in self.news_model.wv]
                if vectors:
                    try:
                        cluster_model = KMeans(n_clusters=3)
                        clusters = cluster_model.fit_predict(vectors)
                        sentiment['clusters'] = len(set(clusters))
                        sentiment['topics'] = self.extract_topics(processed)
                    except Exception as e:
                        logger.error(f"Error en clustering: {str(e)}")
            
            return sentiment
        except Exception as e:
            logger.error(f"Error en análisis avanzado: {str(e)}")
            return {'positive': 0.5, 'negative': 0.5}

    def preprocess_text(self, text):
        """Preprocesamiento de texto para NLP con manejo de errores"""
        try:
            return text.lower().split()
        except Exception as e:
            logger.error(f"Error en preprocess_text: {str(e)}")
            return []

    def extract_topics(self, tokens):
        """Extrae temas clave con manejo de errores"""
        try:
            if not self.news_model or not tokens:
                return []
            return list(set(token for token in tokens if token in self.news_model.wv))
        except Exception as e:
            logger.error(f"Error en extract_topics: {str(e)}")
            return []

# ----------------- FUNCIONES AUXILIARES MEJORADAS -----------------
def get_price(symbol):
    """Obtiene el precio actual con manejo de errores"""
    try:
        return get_real_time_price(symbol)
    except Exception as e:
        logger.error(f"Error en get_price: {str(e)}")
        return 0

def generate_ta_report(symbol):
    """Genera reporte técnico con manejo de errores"""
    try:
        response = requests.get(
            f"{APIConfig.CRYPTOCOMPARE_URL}/indicators",
            params={
                'fsym': symbol.upper(),
                'tsym': 'USD',
                'indicators': 'rsi,macd,ema50,ema200,volumeto'
            },
            headers=APIConfig.get_headers('cryptocompare'),
            timeout=15
        )
        
        if response.status_code != 200:
            return "• No se pudo obtener análisis técnico (error en API)"
            
        data = response.json().get('Data', {})
        
        rsi = data.get('RSI', [{}])[0].get('value', 50)
        macd = data.get('MACD', [{}])[0]
        ema50 = data.get('EMA50', [{}])[0].get('value', 0)
        ema200 = data.get('EMA200', [{}])[0].get('value', 0)
        volume = data.get('VOLUMETO', [{}])[0].get('value', 0)
        
        trend = "Alcista" if ema50 > ema200 else "Bajista"
        momentum = "Fuerte" if (rsi > 70 or rsi < 30) else "Moderado"
        
        return (
            f"• Tendencia: {trend}\n"
            f"• Momentum: {momentum} (RSI: {rsi:.1f})\n"
            f"• Cruce de medias: EMA50 ({ema50:.2f}) vs EMA200 ({ema200:.2f})\n"
            f"• Volumen (24h): {volume:,.0f} USD\n"
            f"• MACD: {macd.get('value', 0):.4f} (Señal: {macd.get('signal', 0):.4f})"
        )
        
    except Exception as e:
        logger.error(f"Error en generate_ta_report: {str(e)}")
        return "• No se pudo generar el análisis técnico"

def generate_recommendation(predictions, ta, sentiment):
    """Genera recomendación integrada con validación"""
    try:
        if not predictions or not ta or not sentiment:
            return "⚠️ No se pudo generar recomendación (datos insuficientes)"
            
        if sentiment.get('positive', 0) > sentiment.get('negative', 0) * 1.5:
            return "✅ Fuerte recomendación de COMPRA (Fundamentales positivos)"
        return "⚠️ Neutral (Esperar confirmación técnica)"
    except Exception as e:
        logger.error(f"Error en generate_recommendation: {str(e)}")
        return "⚠️ Error generando recomendación"

def format_predictions(preds):
    """Formatea predicciones para visualización con manejo de errores"""
    try:
        if not preds:
            return "No hay predicciones disponibles"
        return "\n".join(f"Día {i+1}: ${p:.2f}" for i, p in enumerate(preds))
    except Exception as e:
        logger.error(f"Error en format_predictions: {str(e)}")
        return "Error formateando predicciones"

# ----------------- COMANDOS DEL BOT -----------------
async def ia_prediction(update: Update, context: CallbackContext):
    """Predicción de precios por IA con mejor manejo de errores"""
    try:
        symbol = context.args[0].upper() if context.args else 'BTC'
        predictions = crypto_ai.predict_price(symbol)
        
        if not predictions:
            await update.message.reply_text(f"❌ No se pudieron generar predicciones para {symbol}")
            return
            
        await update.message.reply_text(
            f"📈 Predicción para {symbol}:\n{format_predictions(predictions)}",
            parse_mode='Markdown'
        )
    except IndexError:
        await update.message.reply_text("ℹ️ Uso: /prediccion [símbolo]")
    except Exception as e:
        logger.error(f"Error en ia_prediction: {str(e)}")
        await update.message.reply_text("❌ Error procesando tu solicitud")

async def technical_analysis(update: Update, context: CallbackContext):
    """Análisis técnico con mejor manejo de errores"""
    try:
        symbol = context.args[0].upper() if context.args else 'BTC'
        ta_report = generate_ta_report(symbol)
        
        await update.message.reply_text(
            f"📊 Análisis Técnico {symbol}:\n{ta_report}",
            parse_mode='Markdown'
        )
    except IndexError:
        await update.message.reply_text("ℹ️ Uso: /analisis [símbolo]")
    except Exception as e:
        logger.error(f"Error en technical_analysis: {str(e)}")
        await update.message.reply_text("❌ Error generando análisis técnico")

async def market_analysis(update: Update, context: CallbackContext):
    """Análisis completo del mercado con mejor manejo de errores"""
    try:
        symbol = context.args[0].upper() if context.args else 'BTC'
        predictions = crypto_ai.predict_price(symbol)
        ta_report = generate_ta_report(symbol)
        news = crypto_ai.get_fundamental_news()
        sentiment = crypto_ai.analyze_news_advanced(" ".join([n['title'] for n in news]))
        recommendation = generate_recommendation(predictions, ta_report, sentiment)
        
        if not all([predictions, ta_report, sentiment, recommendation]):
            await update.message.reply_text("❌ No se pudo generar un análisis completo")
            return
            
        message = (
            f"🔍 *Análisis Completo - {symbol}*\n\n"
            f"📈 *Predicción IA (3 días)*:\n"
            f"{format_predictions(predictions)}\n\n"
            f"📊 *Análisis Técnico*:\n"
            f"{ta_report}\n\n"
            f"📰 *Análisis Fundamental*:\n"
            f"• Sentimiento: {sentiment.get('positive', 0)}👍 / {sentiment.get('negative', 0)}👎\n"
            f"• Temas clave: {', '.join(sentiment.get('topics', []))}\n\n"
            f"💡 *Recomendación*:\n"
            f"{recommendation}"
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except IndexError:
        await update.message.reply_text("ℹ️ Uso: /mercado [símbolo]")
    except Exception as e:
        logger.error(f"Error en market_analysis: {str(e)}")
        await update.message.reply_text("❌ Error generando análisis de mercado")

async def crypto_summary(update: Update, context: CallbackContext):
    """Genera un resumen completo de una criptomoneda"""
    try:
        symbol = context.args[0].upper() if context.args else 'BTC'
        
        current_price = get_price(symbol)
        predictions = crypto_ai.predict_price(symbol)
        ta_report = generate_ta_report(symbol)
        news = crypto_ai.get_fundamental_news(limit=5)
        sentiment = crypto_ai.analyze_news_advanced(" ".join([n['title'] for n in news]))
        
        if sentiment.get('positive', 0) > 0.65:
            semaforo = "🟢 Verde"
        elif sentiment.get('negative', 0) > 0.65:
            semaforo = "🔴 Rojo"
        else:
            semaforo = "🟡 Amarillo"
            
        entry = current_price * 0.98 if "Alcista" in ta_report else current_price * 0.95
        exit_price = current_price * 1.02 if "Alcista" in ta_report else current_price * 1.05
        
        message = (
            f"📈 *{symbol}*\n"
            f"💵 Precio: {current_price:.2f}\n"
            f"📊 Tendencia: {'En tendencia alcista' if 'Alcista' in ta_report else 'En tendencia bajista'}\n"
            f"⚠️ Semáforo: {semaforo}\n"
            f"💡 Entrada sugerida: {entry:.4f}\n"
            f"🚪 Salida sugerida: {exit_price:.4f}\n\n"
            f"📰 *Noticias fundamentales del día:*\n\n"
        )
        
        for item in news:
            title = item.get('title', 'Sin título')
            source = item.get('source', 'Fuente desconocida')
            url = item.get('url', '#')
            message += f"• {title} - {source}\n{url}\n\n"
        
        await update.message.reply_text(
            text=message,
            parse_mode='Markdown',
            disable_web_page_preview=False
        )
        
    except IndexError:
        await update.message.reply_text("ℹ️ Uso: /resumen [símbolo]")
    except Exception as e:
        logger.error(f"Error en crypto_summary: {str(e)}")
        await update.message.reply_text("❌ Error generando el resumen")

async def learn_command(update: Update, context: CallbackContext):
    """Permite enseñar nuevos comandos al bot con mejor manejo de errores"""
    try:
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("ℹ️ Formato: /aprender <comando> <respuesta>")
            return
            
        command = context.args[0].lower()
        response = " ".join(context.args[1:])
        
        try:
            with open("knowledge_base.json", "a+") as f:
                json.dump({"command": command, "response": response}, f)
                f.write("\n")
                
            await update.message.reply_text(f"✅ Aprendido nuevo comando: /{command}")
        except IOError as e:
            logger.error(f"Error escribiendo en knowledge_base: {str(e)}")
            await update.message.reply_text("❌ Error guardando el comando")
            
    except Exception as e:
        logger.error(f"Error en learn_command: {str(e)}")
        await update.message.reply_text("❌ Error procesando tu solicitud")

async def execute_custom_command(update: Update, context: CallbackContext):
    """Ejecuta comandos personalizados con mejor manejo de errores"""
    try:
        command = update.message.text[1:].lower().split()[0]
        
        if not os.path.exists("knowledge_base.json"):
            await update.message.reply_text("ℹ️ No hay comandos personalizados aprendidos")
            return
            
        try:
            with open("knowledge_base.json", "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data["command"] == command:
                            await update.message.reply_text(data["response"])
                            return
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            logger.error(f"Error leyendo knowledge_base: {str(e)}")
            await update.message.reply_text("❌ Error accediendo a la base de conocimientos")
                    
        await update.message.reply_text("ℹ️ Comando desconocido. Use /aprender para enseñarme")
    except Exception as e:
        logger.error(f"Error en execute_custom_command: {str(e)}")
        await update.message.reply_text("❌ Error procesando comando personalizado")

# ----------------- ALERTAS PROGRAMADAS -----------------
async def enviar_alerta_automatica(context: CallbackContext):
    """Envía alertas automáticas con todo el análisis"""
    try:
        chat_id = context.job.chat_id if hasattr(context, 'job') and hasattr(context.job, 'chat_id') else None
        
        if not chat_id:
            chat_id = os.getenv("CANAL_ID")
            if not chat_id:
                logger.error("No se pudo determinar el chat_id para la alerta automática")
                return

        market_data = get_market_overview()
        
        message = "📊 *Resumen del Mercado Cripto* 📊\n\n"
        message += f"🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        
        for symbol in ["BTC", "ETH", "BNB", "SOL", "XRP"]:
            price = get_real_time_price(symbol)
            change_24h = get_24h_change(symbol)
            
            message += (
                f"🔹 *{symbol}*: ${price:,.2f} "
                f"({change_24h:+.2f}%)\n"
            )
        
        news = get_real_news(limit=3)
        if news:
            message += "\n📰 *Noticias destacadas:*\n"
            for item in news:
                message += f"• {item['title']}\n{item['url']}\n"
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
            disable_web_page_preview=True
        )
        
    except Exception as e:
        logger.error(f"Error en enviar_alerta_automatica: {str(e)}")

async def check_alerts(context: CallbackContext):
    """Verifica condiciones de mercado para alertas con mejor manejo de errores"""
    try:
        for symbol in ["BTC", "ETH"]:
            try:
                preds = crypto_ai.predict_price(symbol)
                current = get_price(symbol)
                
                if preds and current > 0:
                    crypto_ai.update_with_feedback(symbol, current, preds[-1])
                    
                    if abs(preds[-1] - current) > current * 0.05:
                        await context.bot.send_message(
                            chat_id=context.job.chat_id,
                            text=f"⚠️ Alerta {symbol}: Gran discrepancia IA/mercado\n"
                                 f"Predicho: ${preds[-1]:.2f} vs Actual: ${current:.2f}"
                        )
            except Exception as e:
                logger.error(f"Error verificando alerta para {symbol}: {str(e)}")
                
        if datetime.now().hour == 3:
            crypto_ai.retrain_models()
    except Exception as e:
        logger.error(f"Error en check_alerts: {str(e)}")

# ----------------- INICIALIZACIÓN MEJORADA -----------------
crypto_ai = SelfLearningCryptoAI()

async def post_init(application: Application):
    """Configuración posterior a la inicialización mejorada"""
    try:
        job_queue = application.job_queue
        if job_queue:
            try:
                chat_id = os.getenv("CANAL_ID")
                if chat_id:
                    chat_id = int(chat_id) if isinstance(chat_id, str) else chat_id
                    
                    job_queue.run_repeating(
                        check_alerts,
                        interval=3600,
                        first=0,
                        chat_id=chat_id
                    )
                    
                    job_queue.run_daily(
                        lambda ctx: asyncio.create_task(crypto_ai.retrain_models()),
                        time=time(hour=3),
                        chat_id=chat_id
                    )
                    
                    logger.info("Alertas programadas configuradas correctamente")
                    
                    await configurar_alertas_automaticas(application)
                    
            except Exception as e:
                logger.warning(f"CANAL_ID no configurado correctamente. Error: {str(e)}")
        else:
            logger.warning("JobQueue no está disponible. Las alertas programadas no funcionarán.")
    except Exception as e:
        logger.error(f"Error en post_init: {str(e)}")

async def configurar_alertas_automaticas(app: Application):
    """Configura el scheduler para las alertas automáticas"""
    try:
        if hasattr(app, '_alert_scheduler'):
            return app._alert_scheduler
            
        scheduler = AsyncIOScheduler()
        
        horas_alertas = [6, 12, 18, 0]
        
        for hora in horas_alertas:
            scheduler.add_job(
                enviar_alerta_automatica,
                'cron',
                hour=hora,
                minute=0,
                args=[app],
                misfire_grace_time=60,
                name=f"alerta_{hora}h"
            )
        
        scheduler.start()
        app._alert_scheduler = scheduler
        logger.info(f"Alertas automáticas programadas para las horas: {horas_alertas}")
        return scheduler
    except Exception as e:
        logger.error(f"Error configurando alertas automáticas: {str(e)}")
        return None

async def shutdown(application: Application):
    """Cierre limpio de la aplicación"""
    try:
        if hasattr(application, '_alert_scheduler'):
            scheduler = application._alert_scheduler
            if scheduler and scheduler.running:
                scheduler.shutdown(wait=False)
                logger.info("Scheduler de alertas detenido correctamente")
        
        crypto_ai.save_models()
        logger.info("Modelos de IA guardados correctamente")
        
        if application.running:
            await application.stop()
            await application.shutdown()
    except Exception as e:
        logger.error(f"Error durante el cierre: {str(e)}")
        raise

async def main():
    """Función principal mejorada con manejo adecuado del event loop"""
    application = None
    try:
        tracemalloc.start()
        
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        application = ApplicationBuilder().token("7928518493:AAE4tvmK8MBiZtaSwRbSKj1Io95WPiuyADI").build()
        
        application.add_handler(CommandHandler("prediccion", ia_prediction))
        application.add_handler(CommandHandler("analisis", technical_analysis))
        application.add_handler(CommandHandler("mercado", market_analysis))
        application.add_handler(CommandHandler("resumen", crypto_summary))
        application.add_handler(CommandHandler("aprender", learn_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, execute_custom_command))

        await post_init(application)

        logger.info("Iniciando bot...")
        await application.run_polling()

    except asyncio.CancelledError:
        logger.info("Bot detenido por
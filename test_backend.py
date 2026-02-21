import asyncio
import websockets
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

async def listen_to_backend():
    uri = "ws://localhost:8000/ws"
    
    while True:
        try:
            logging.info(f"Connecting to Backend WebSocket at {uri}...")
            async with websockets.connect(uri) as websocket:
                logging.info("✅ Connected to Backend! Listening for live data...")
                logging.info("--------------------------------------------------")
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if "prediction" in data and data["prediction"]:
                        pred = data["prediction"]
                        dir_str = pred.get("direction", "WAITING")
                        conf = pred.get("confidence", 0)
                        price = pred.get("current_price", 0)
                        
                        logging.info(f"🎯 SIGNAL OVER WIRE: {dir_str} ({conf:.1f}%) | BTC Price: ${price:.2f}")
                    else:
                        logging.info("⏳ Received ping/tick, but no prediction data attached yet...")
                        
        except websockets.exceptions.ConnectionClosed:
            logging.error("❌ Connection closed by the server. Retrying in 5s...")
            await asyncio.sleep(5)
        except ConnectionRefusedError:
            logging.error("❌ Connection refused. Server not ready. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"❌ Error: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(listen_to_backend())
    except KeyboardInterrupt:
        print("\nExiting...")

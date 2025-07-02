# websocket_server.py
# streams/websocket_server.py

import asyncio
import websockets
import json
import os

WS_FEED_FILE = "data/live_websocket_feed.json"

async def handle_socket(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            os.makedirs(os.path.dirname(WS_FEED_FILE), exist_ok=True)
            with open(WS_FEED_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WebSocket Error] {e}")

def start_ws_server(host="0.0.0.0", port=6789):
    print(f"🔌 WebSocket Server running on ws://{host}:{port}")
    loop = asyncio.get_event_loop()
    server = websockets.serve(handle_socket, host, port)
    loop.run_until_complete(server)
    loop.run_forever()


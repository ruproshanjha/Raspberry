import cv2
import asyncio
import websockets
import numpy as np

async def stream(websocket):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Encode frame as JPEG
        _, jpg = cv2.imencode('.jpg', frame)
        data = jpg.tobytes()

        # Send binary data
        await websocket.send(data)

        await asyncio.sleep(0.03)  # ~30 FPS

async def main():
    async with websockets.serve(stream, "0.0.0.0", 8765, max_size=None):
        print("WebSocket Server running at ws://0.0.0.0:8765")
        await asyncio.Future()

asyncio.run(main())

import asyncio
import logging
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect

def singleton(cls):
    """Make SocketManager a singleton class."""
    
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class SocketManager:
    """Class to track websocket connection by tracker_id and safe broadcast."""
    
    def __init__(self):
        # store connection here, mapping a room_id(tracker_id) to a list of clients.
        self.active_sessions: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self.log = logging.getLogger(self.__class__.__name__)
    
    async def connect(self, websocket: WebSocket, tracker_id:str):
        """Accept a client connection and add it to a room(tracker_id)."""

        await websocket.accept()
        self.log.info(f"Connection accepted for tracker_id: {tracker_id}, client: {websocket.client.host}:{websocket.client.port}")
        async with self._lock:
            connections = self.active_sessions.get(tracker_id)
            if connections is None:
                self.active_sessions[tracker_id] = [websocket]
                self.log.info(f"Created new room for tracker_id: {tracker_id}")
            else:
                if websocket not in connections:
                    connections.append(websocket)

    async def disconnect(self, websocket: WebSocket, tracker_id:str):
        """Remove a client connection from the room and clean up if empty."""

        self.log.info(f"Disconnecting client for tracker_id: {tracker_id}, client: {websocket.client.host}:{websocket.client.port}")
        async with self._lock:
            connections = self.active_sessions.get(tracker_id)
            if not connections:
                self.log.warning(f"Attempted to disconnect from non-existent room: {tracker_id}")
                return
            try:
                connections.remove(websocket)
                self.log.info(f"Removed connection for tracker_id: {tracker_id}")
            except ValueError:
                self.log.warning(f"Connection not found in room: {tracker_id} during disconnect")
                pass
            if not connections:
                del self.active_sessions[tracker_id]
                self.log.info(f"Removed empty room for tracker_id: {tracker_id}")

    async def broadcast(self, event: str, data:dict, tracker_id:str):
        """Send a JSON message to all clients in a room."""

        message = {
            "event": event,
            "data": data
            }
        # Snapshot connections so we don't mutate while iterating
        async with self._lock:
            connections = list(self.active_sessions.get(tracker_id, []))
        if not connections:
            self.log.debug(f"No active connections in room: {tracker_id} for broadcast")
            return
        self.log.debug(f"Broadcasting to {len(connections)} connections in room: {tracker_id}")
        await asyncio.gather(
            *(self._send_json_safe(conn, message, tracker_id) for conn in connections),
            return_exceptions=True,
        )

    async def _send_json_safe(self, connection: WebSocket, message: dict, tracker_id: str):
        try:
            await connection.send_json(message)

        except WebSocketDisconnect:
            self.log.info(f"WebSocket disconnected during send to room: {tracker_id}")
            await self.disconnect(connection, tracker_id)

        except Exception as e:
            self.log.error(f"Uexpected error sending message to room: {tracker_id}, error: {str(e)}")
            await self.disconnect(connection, tracker_id)

####################################
# singleton socket manager instance
ws_manager = SocketManager()
####################################
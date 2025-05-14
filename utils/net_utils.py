
from websockets.sync.server import serve, ServerConnection
from threading import Thread
import threading
import time
from queue import Queue
from collections import deque
import pickle, torch
from scene.cameras import MiniCam

Q_out = Queue(1000)
Q_in = deque(maxlen=2)
clients = set()

def new_client(websocket: ServerConnection):
    global Q_out, Q_in
    if len(clients) != 0: 
        websocket.close()
        print(f"Reject new client")
        return
    
    Q_out = Queue(1000)
    Q_in.clear()
    clients.add(websocket)
    print(f"New client connected")

def client_left(websocket):
    clients.clear()
    print(f"Client disconnected")

def keep_send(websocket: ServerConnection):
    global Q_out
    try:
        while True:
            data = Q_out.get(block=True)
            websocket.send(data)
    except Exception as e:
        pass

def handler(websocket: ServerConnection):
    global Q_in
    new_client(websocket)
    Thread(target=keep_send, args=(websocket,), daemon=True).start()
    try:
        while True:
            message = websocket.recv()
            Q_in.append(message)
    except Exception as e:
        print(e)
        client_left(websocket)
    
def run_server(ip, port):
    with serve(handler, ip, port, compression=None, server_header=None) as server:
        print(f'WebSocket server is running on ws://{ip}:{port}')
        server.serve_forever()

#---------------------------------

def net_init(ip, port):
    Thread(target=run_server, args=(ip, port), daemon=True).start()

def recv():
    global Q_in
    try:
        data = Q_in.popleft()
    except IndexError:
        data = None
    return data

def send(data):
    global Q_out
    Q_out.put(data, block=True)

def is_connected():
    return len(clients) > 0

def wait_connection():
    status = 'alive'
    while True:
        if is_connected():
            return status
        status = 'connected'
        time.sleep(0.1)

##################


def receive():
    message = recv()
    if message is None: return None

    message = pickle.loads(message)
    width = message["resolution_x"]
    height = message["resolution_y"]
    if width == 0 or height == 0: return None

    fovy = message["fov_y"]
    fovx = message["fov_x"]
    znear = message["z_near"]
    zfar = message["z_far"]
    world_view_transform = torch.tensor(message["view_matrix"]).cuda(non_blocking=True)
    full_proj_transform = torch.tensor(message["view_projection_matrix"]).cuda(non_blocking=True)
    custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)

    data = message
    del data['resolution_x'], data['resolution_y'], data['fov_x'], data['fov_y'], data['z_near'], data['z_far'], data['view_matrix'], data['view_projection_matrix']

    data['camera'] = custom_cam
    data['do_training'] = True
    data['keep_alive'] = True

    return data


def main():
    net_init('127.0.0.1', 12346)
    while True:
        data = recv()
        if data is None: continue
        message = data
        send(message)

if __name__ == "__main__":
    main()

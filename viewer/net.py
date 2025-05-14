
import websocket
import threading
from queue import Queue
from collections import deque

Q_out = Queue(1000)
Q_in = deque(maxlen=100)
connect_status = False

def on_message(ws, message):
    global Q_in
    Q_in.append(message)

def on_error(ws, error):
    global connect_status
    connect_status = False
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    global connect_status
    connect_status = False
    print("Connection closed")

def keep_send(ws):
    global Q_out
    try:
        while True:
            data_to_send = Q_out.get(block=True) 
            ws.send_bytes(data_to_send)
    except Exception as e:
        print(e)

def on_open(ws):
    global Q_in, Q_out, connect_status
    connect_status = True
    print('Open connnection')
    Q_out = Queue(1000)
    Q_in = deque(maxlen=10)
    threading.Thread(target=keep_send, args=(ws,), daemon=True).start()

# --------------------------------

conn = None

def net_init(ip, port):
    ws = websocket.WebSocketApp(f'ws://{ip}:{port}',
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    def ws_run():
        ws.run_forever(reconnect=2)
    threading.Thread(target=ws_run, daemon=True).start()
    global conn
    conn = ws

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
    return connect_status

def close():
    conn.close()

#######

def main():
    net_init('127.0.0.1', 12345)
    for i in range(100000):
        send(i.to_bytes(10))
        data = recv()

if __name__ == "__main__":
    main()
    print('end')

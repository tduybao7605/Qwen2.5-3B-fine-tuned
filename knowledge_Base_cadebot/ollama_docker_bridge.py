import socket
import threading
import sys

LISTEN_IP = "172.17.0.1"
LISTEN_PORT = 11434
TARGET_IP = "127.0.0.1"
TARGET_PORT = 11434

def handle_client(client_socket):
    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        target_socket.connect((TARGET_IP, TARGET_PORT))
    except Exception as e:
        client_socket.close()
        return

    def forward(src, dst):
        try:
            while True:
                data = src.recv(8192)
                if not data:
                    break
                dst.sendall(data)
        except:
            pass
        finally:
            try:
                src.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                dst.shutdown(socket.SHUT_RDWR)
            except:
                pass
            src.close()
            dst.close()

    t1 = threading.Thread(target=forward, args=(client_socket, target_socket), daemon=True)
    t2 = threading.Thread(target=forward, args=(target_socket, client_socket), daemon=True)
    t1.start()
    t2.start()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((LISTEN_IP, LISTEN_PORT))
        server.listen(128)
        print(f"✅ Ollama Docker Bridge running on {LISTEN_IP}:{LISTEN_PORT} -> {TARGET_IP}:{TARGET_PORT}")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Socket bind error: {e}")
        sys.exit(1)

    while True:
        try:
            client, addr = server.accept()
            threading.Thread(target=handle_client, args=(client,), daemon=True).start()
        except KeyboardInterrupt:
            break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()

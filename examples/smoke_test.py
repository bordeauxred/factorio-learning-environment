import sys
from factorio_rcon import RCONClient

def smoke_test(host="localhost", port=27000, password="factorio"):
    print(f"Connecting to Factorio at {host}:{port}...")
    try:
        client = RCONClient(host, port, password)
        client.connect()
        
        print("SUCCESS: RCON connected!")
        
        response = client.send_command("/sc rcon.print('Hello from Smoke Test')")
        print(f"Server Response: {response}")
        
        players = client.send_command("/players")
        print(f"Players: {players}")
        
        client.close()
        
    except Exception as e:
        print(f"FAILURE: Could not connect or interact. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 27000
    password = sys.argv[3] if len(sys.argv) > 3 else "factorio"
    smoke_test(host, port, password)

import sys
from fle.env.instance import FactorioInstance

def smoke_test(host="localhost", port=27000):
    print(f"Connecting to Factorio at {host}:{port}...")
    try:
        # Initialize FactorioInstance (this connects via RCON)
        # We use a dummy instance configuration since we just want to test connection
        instance = FactorioInstance(
            address=host,
            tcp_port=port,
            num_agents=1,
            skip_setup=True # We might need to add this flag or handle init carefully
        )
        
        # Since FactorioInstance does a lot of setup in __init__, we might want to just use RCONClient directly
        # for a pure smoke test if FactorioInstance is too heavy/expects specific state.
        # But let's try to use the class to verify FLE compatibility.
        
        # Actually, looking at FactorioInstance.__init__, it connects and then does setup.
        # If we want a raw connectivity test:
        from factorio_rcon import RCONClient
        client = RCONClient(host, port, "factorio")
        client.connect()
        
        print("SUCCESS: RCON connected!")
        
        response = client.send_command("/sc rcon.print('Hello from Smoke Test')")
        print(f"Server Response: {response}")
        
        players = client.send_command("/players")
        print(f"Players: {players}")
        
    except Exception as e:
        print(f"FAILURE: Could not connect or interact. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 27000
    smoke_test(host, port)

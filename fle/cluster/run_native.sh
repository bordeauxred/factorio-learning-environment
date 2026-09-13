#!/bin/bash

# Configuration
FACTORIO_VERSION="1.1.110"
FACTORIO_URL="https://www.factorio.com/get-download/${FACTORIO_VERSION}/headless/linux64"
INSTALL_DIR="factorio_server"
SCENARIOS_DIR="scenarios" # Relative to script location
CONFIG_DIR="config"       # Relative to script location

# Function to download and install Factorio
install_factorio() {
    if [ -d "$INSTALL_DIR" ]; then
        echo "Factorio server already installed in $INSTALL_DIR"
        return
    fi

    echo "Downloading Factorio Headless Server ${FACTORIO_VERSION}..."
    wget -O factorio_headless.tar.xz "$FACTORIO_URL"

    echo "Extracting..."
    mkdir -p "$INSTALL_DIR"
    tar -xJf factorio_headless.tar.xz -C "$INSTALL_DIR" --strip-components=1
    rm factorio_headless.tar.xz

    echo "Factorio installed successfully."
}

# Function to start instances
start_instances() {
    NUM_INSTANCES=${1:-1}
    SCENARIO=${2:-"default_lab_scenario"}

    echo "Starting $NUM_INSTANCES Factorio instances with scenario $SCENARIO..."

    # Create pids directory
    mkdir -p pids

    for i in $(seq 0 $(($NUM_INSTANCES - 1))); do
        UDP_PORT=$((34197 + i))
        TCP_PORT=$((27000 + i))
        INSTANCE_DIR="${INSTALL_DIR}_${i}"

        # Create instance directory by copying base install (using symlinks to save space/time would be better but copy is safer for config isolation)
        # Actually, we can just use the same binary but different write-data directories?
        # Standard way: copy the whole thing or use --write-data
        
        # Let's use a simpler approach: One install, multiple write-data dirs? 
        # Or just copy the install for isolation simplicity as per Docker logic.
        # Docker logic isolates everything. Let's replicate that structure.
        
        if [ ! -d "$INSTANCE_DIR" ]; then
            echo "Creating instance directory $INSTANCE_DIR..."
            cp -r "$INSTALL_DIR" "$INSTANCE_DIR"
        fi

        # 1. Setup Config
        mkdir -p "$INSTANCE_DIR/config"
        cp -r "$CONFIG_DIR"/* "$INSTANCE_DIR/config/"

        # 2. Setup Scenarios
        mkdir -p "$INSTANCE_DIR/scenarios"
        cp -r "$SCENARIOS_DIR"/* "$INSTANCE_DIR/scenarios/"

        # 3. Setup Mods (if any exist in ../mods)
        if [ -d "../mods" ]; then
            echo "Copying mods from ../mods to instance $i..."
            mkdir -p "$INSTANCE_DIR/mods"
            cp -r "../mods"/* "$INSTANCE_DIR/mods/"
        fi

        # Start the server
        echo "Starting instance $i on port $UDP_PORT (UDP) / $TCP_PORT (RCON TCP)..."
        
        # Build command arguments
        # Note: We use relative paths inside the instance dir or absolute paths
        # It's safer to use absolute paths or paths relative to the binary?
        # The binary is in bin/x64/factorio. PWD will be the script dir.
        # Let's use absolute paths for config to be safe, or relative to where we run the command.
        # We will run the command from the script dir, so we point to the instance dir files.
        
        ARGS="--start-server-load-scenario $SCENARIO"
        ARGS="$ARGS --port $UDP_PORT"
        ARGS="$ARGS --server-settings $INSTANCE_DIR/config/server-settings.json"
        ARGS="$ARGS --map-gen-settings $INSTANCE_DIR/config/map-gen-settings.json"
        ARGS="$ARGS --map-settings $INSTANCE_DIR/config/map-settings.json"
        ARGS="$ARGS --server-banlist $INSTANCE_DIR/config/server-banlist.json"
        ARGS="$ARGS --server-whitelist $INSTANCE_DIR/config/server-whitelist.json"
        ARGS="$ARGS --server-adminlist $INSTANCE_DIR/config/server-adminlist.json"
        ARGS="$ARGS --rcon-port $TCP_PORT"
        ARGS="$ARGS --rcon-password factorio"
        
        # Run in background
        nohup "$INSTANCE_DIR/bin/x64/factorio" $ARGS > "factorio_${i}.log" 2>&1 &
        PID=$!
        echo $PID > "pids/factorio_${i}.pid"
        echo "Started instance $i (PID $PID)"
    done
}

# Function to stop instances
stop_instances() {
    echo "Stopping Factorio instances..."
    if [ -d "pids" ]; then
        for pid_file in pids/factorio_*.pid; do
            if [ -f "$pid_file" ]; then
                PID=$(cat "$pid_file")
                if kill -0 "$PID" 2>/dev/null; then
                    echo "Stopping PID $PID..."
                    kill "$PID"
                else
                    echo "PID $PID not running."
                fi
                rm "$pid_file"
            fi
        done
    else
        echo "No pids directory found."
    fi
}

# Main logic
case "$1" in
    install)
        install_factorio
        ;;
    start)
        install_factorio
        start_instances "${2:-1}" "${3:-default_lab_scenario}"
        ;;
    stop)
        stop_instances
        ;;
    *)
        echo "Usage: $0 {install|start [num_instances] [scenario]|stop}"
        exit 1
        ;;
esac

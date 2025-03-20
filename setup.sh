#!/bin/bash

# Function to display messages
log() {
    echo -e "\n\033[1;32m$1\033[0m\n"
}

# Function to replace 'cuda' with 'cpu' in specified files
replace_cuda_with_cpu() {
    local file=$1
    if [ -f "$file" ]; then
        log "Updating $file to replace all occurrences of 'cuda' with 'cpu'..."
        sed -i 's/\bcuda\b/cpu/g' "$file"
        log "Finished updating $file."
    else
        echo "File $file not found! Skipping."
    fi
}

# Step 1: Install dependencies from requirements.txt
log "Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Step 2: Update and install system dependencies
log "Updating system and installing libgl1..."
sudo apt update && sudo apt install -y libgl1 wget

# Step 3: Install Segment Anything model
log "Installing Segment Anything model..."
pip3 install git+https://github.com/facebookresearch/segment-anything.git

# Step 4: Clone, install, and update GroundingDINO in final_submission
log "Installing GroundingDINO in final_submission folder..."
cd src/final_submission
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
else
    log "GroundingDINO folder already exists. Skipping clone step."
fi
cd GroundingDINO
pip install -e .
replace_cuda_with_cpu "groundingdino/util/inference.py"
replace_cuda_with_cpu "groundingdino/util/misc.py"
cd ../../..

# Step 5: Clone, install, and update GroundingDINO in wallpaper folder
log "Installing GroundingDINO in wallpaper folder..."
cd src/wallpaper
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
else
    log "GroundingDINO folder already exists. Skipping clone step."
fi
cd GroundingDINO
pip install -e .
replace_cuda_with_cpu "groundingdino/util/inference.py"
replace_cuda_with_cpu "groundingdino/util/misc.py"
cd ../../..

# Step 6: Download weights for Segment Anything model
log "Setting up weights directory and downloading weights..."
WEIGHTS_DIR="src/final_submission/weights"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

log "Checking if the weights directory exists..."
if [ ! -d "$WEIGHTS_DIR" ]; then
    log "Weights directory not found. Creating $WEIGHTS_DIR..."
    mkdir -p "$WEIGHTS_DIR"
else
    log "Weights directory already exists. Skipping creation."
fi

log "Downloading weights file..."
# Server setup
wget -P "$WEIGHTS_DIR" "$WEIGHTS_URL" && log "Weights downloaded successfully!" || log "Failed to download weights."

# Loacl Setup
# curl -o "$WEIGHTS_DIR/sam_vit_h_4b8939.pth" "$WEIGHTS_URL" && log "Weights downloaded successfully!" || log "Failed to download weights."

log "Setup complete!"

# To free cache used by installation process
sync; echo 1 > /proc/sys/vm/drop_caches

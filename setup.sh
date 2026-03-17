#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Project Setup...${NC}"

# Backend Setup
echo -e "${BLUE}Setting up Backend...${NC}"
cd backend
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${YELLOW}Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Frontend Setup
echo -e "${BLUE}Setting up Frontend...${NC}"
cd frontend
echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
npm install
cd ..

echo -e "${GREEN}Setup complete!${NC}"
echo -e "You can now start both services by running: ${BLUE}./run.sh${NC}"

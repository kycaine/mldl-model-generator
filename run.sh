#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}Starting ML Platform...${NC}"

# Start Backend in the background
echo -e "${BLUE}Starting FastAPI Backend...${NC}"
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend in the background
echo -e "${BLUE}Starting React Frontend...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}Both services are running!${NC}"
echo -e "Backend: http://localhost:8000"
echo -e "Frontend: http://localhost:5173"
echo -e "Press Ctrl+C to stop both services."

# Trap Ctrl+C to stop both processes
trap "echo -e '\n${GREEN}Stopping services...${NC}'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Keep script running
wait

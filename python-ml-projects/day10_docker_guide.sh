#!/bin/bash

echo "======================================================================"
echo "  DOCKER DEPLOYMENT GUIDE - Titanic ML API"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}Step 1: Building Docker Image${NC}"
echo "----------------------------------------------------------------------"
docker build -t titanic-ml-api:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image built successfully!${NC}"
else
    echo -e "${YELLOW}❌ Build failed. Check error messages above.${NC}"
    exit 1
fi

echo -e "\n${BLUE}Step 2: Image Information${NC}"
echo "----------------------------------------------------------------------"
docker images titanic-ml-api

echo -e "\n${BLUE}Step 3: Running Container${NC}"
echo "----------------------------------------------------------------------"
docker run -d \
    --name titanic-api \
    -p 5000:5000 \
    -v $(pwd)/models:/app/models:ro \
    titanic-ml-api:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container started successfully!${NC}"
else
    echo -e "${YELLOW}❌ Container failed to start.${NC}"
    exit 1
fi

echo -e "\n${BLUE}Step 4: Container Status${NC}"
echo "----------------------------------------------------------------------"
docker ps | grep titanic-api

echo -e "\n${BLUE}Step 5: Waiting for API to be ready...${NC}"
echo "----------------------------------------------------------------------"
sleep 5

echo -e "\n${BLUE}Step 6: Testing API${NC}"
echo "----------------------------------------------------------------------"

# Test health endpoint
echo "Testing /health endpoint..."
curl -s http://localhost:5000/health | python3 -m json.tool

echo -e "\n\nTesting /predict endpoint..."
curl -s -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "Pclass": 1,
        "Name": "Miss. Jane Smith",
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C",
        "Cabin": "C85",
        "Ticket": "12345"
    }' | python3 -m json.tool

echo -e "\n\n${GREEN}======================================================================"
echo "  🎉 DOCKER DEPLOYMENT COMPLETE!"
echo "======================================================================${NC}"

echo -e "\n${BLUE}Useful Docker Commands:${NC}"
echo "  View logs:      docker logs titanic-api"
echo "  Stop container: docker stop titanic-api"
echo "  Start container: docker start titanic-api"
echo "  Remove container: docker rm -f titanic-api"
echo "  Remove image:   docker rmi titanic-ml-api"
echo "  Shell access:   docker exec -it titanic-api /bin/bash"
echo ""
echo -e "${BLUE}API Available at:${NC} http://localhost:5000"
echo ""

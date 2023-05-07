# Docker setup
docker build . -t apud 

docker run -d apud    

This will launch docker container with Flask server

### Test
http://localhost:5000/houses
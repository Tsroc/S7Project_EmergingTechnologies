# S7Project_EmergingTechnologies-

set FLASK_APP=webserver.py
python -m flask run

docker build . -t webserver .
docker run -d -p 5000:5000 webserver
FROM ubuntu:latest
WORKDIR /app
RUN apt-get update -y && apt-get install python3-pip -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app/
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
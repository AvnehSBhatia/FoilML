# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.11

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /

# Install PyTorch CPU-only first
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*


COPY . .

EXPOSE 8080

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]

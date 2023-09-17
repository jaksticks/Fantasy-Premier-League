FROM python:3.10.6
COPY . .
RUN pip install -r requirements_23-24.txt

FROM python:3.7-slim

RUN mkdir /front

COPY requirements.txt /front

WORKDIR /front

RUN pip install -r requirements.txt

COPY . /front

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
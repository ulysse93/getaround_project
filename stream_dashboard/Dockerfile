FROM continuumio/miniconda3

WORKDIR /home/app
COPY requirements.txt .

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install altair pandas numpy streamlit pydeck matplotlib seaborn plotly.express plotly.graph_objs
COPY . /home/app
CMD ["streamlit", "run", "--server.port=8501", "app.py"]

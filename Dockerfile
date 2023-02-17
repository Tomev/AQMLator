FROM continuumio/anaconda3
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install tox
RUN pip install tox-current-env
RUN pip install -r requirements_dev.txt

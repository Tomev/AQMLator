FROM continuumio/anaconda3
RUN pip install requirements.txt
RUN pip install tox
RUN pip install tox-current-env
RUN pip install requirements_dev.txt

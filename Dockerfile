FROM continuumio/anaconda3
RUN conda install scikit-learn
RUN conda install dill
# Test requirements
RUN pip install tox-current-env
RUN pip install black
RUN pip install flake8
RUN pip install mypy
RUN pip install pylint
RUN pip install tox

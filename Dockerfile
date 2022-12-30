FROM continuumio/anaconda3
RUN conda install scikit-learn
RUN conda install dill
RUN pip install pennylane
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN conda install matplotlib
RUN pip install optuna
# Test requirements
RUN pip install tox
RUN pip install tox-current-env
RUN pip install black
RUN pip install flake8
RUN pip install mypy
RUN pip install pylint


FROM jupyter/scipy-notebook AS stage1

COPY requirements.txt ./requirements.txt
COPY crnn_data.py ./crnn_data.py
COPY crnn_model.py ./crnn_model.py
COPY utils ./utils
COPY dataset ./dataset

RUN pip install -r requirements.txt

COPY train.py ./train.py

RUN python3 train.py

FROM scratch AS export-stage

COPY --from=stage1 /home/jovyan/model/ .

# Adapted from repository https://github.com/anibali/docker-pytorch
FROM nvcr.io/nvidia/pytorch:20.03-py3

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install ray

CMD ["bash"]
FROM continuumio/anaconda3

COPY websensors-ocec-glove50d-tsne-osvm-v1.py /websensors-ocec-glove50d-tsne-osvm-v1.py
COPY entrypoint.sh /entrypoint.sh
COPY resources /resources
RUN mkdir /iexec_out/
RUN chmod 777 /iexec_out/
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

FROM python:3.7
COPY ./find-similarities.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./short-wiki.csv /deploy/
COPY ./model-indexes/ /deploy/
WORKDIR /deploy/
RUN pip install --quiet -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "find-similarities.py"]
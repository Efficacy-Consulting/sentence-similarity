FROM python:3.7 as base

RUN mkdir /work/
WORKDIR /work/

COPY ./requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt

COPY ./find-similarities.py /work/

ENV FLASK_APP=find-similarities.py

###########START NEW IMAGE : DEBUGGER ###################
FROM base as debug
RUN pip install ptvsd

WORKDIR /work/
CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait --multiprocess -m flask run -h 0.0.0 -p 1975

###########START NEW IMAGE: PRODUCTION ###################
FROM base as prod

CMD flask run -h 0.0.0 -p 1975

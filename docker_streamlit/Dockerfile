# base image
FROM python:3.6.9

# # virtualenv
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# ENV PATH="/opt/venv/bin:$PATH"

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /usr/src/app


# add and install requirements
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.1/en_core_web_md-2.3.1.tar.gz#egg=en_core_web_md

# copying all files over
COPY . . 

# cmd to launch app when container is run
CMD streamlit run src/app.py
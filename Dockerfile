FROM osgeo/gdal:ubuntu-small-latest

RUN apt-get update
RUN apt install -y python3-pip
WORKDIR /work
COPY requirements.txt ./
RUN pip3 install -r requirements.txt



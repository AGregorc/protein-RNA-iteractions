FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app

# Copy the code
COPY . .

# install from requirements.txt
RUN conda update conda \
    && conda install --file requirements-cpu.txt -c pytorch -c dglteam -c conda-forge -y

# Install msms
RUN mkdir /tmp_msms \
    && cd /tmp_msms \
    && curl http://mgltools.scripps.edu/downloads/tars/releases/MSMSRELEASE/REL2.6.1/msms_i86_64Linux2_2.6.1.tar.gz -o msms.tar.gz \
    && sudo mkdir /usr/local/lib/msms \
    && cd /usr/local/lib/msms \
    && tar zxvf /tmp_msms/msms.tar.gz  \
    && rm -rf /tmp_msms* \
    && sudo ln -s /usr/local/lib/msms/msms.x86_64Linux2.2.6.1 /usr/local/bin/msms \
    && sudo ln -s /usr/local/lib/msms/pdb_to_xyzr* /usr/local/bin

ENV DGLBACKEND=pytorch

WORKDIR /app/src

CMD ["update_dataset_and_preprocess.py"]
ENTRYPOINT ["python"]

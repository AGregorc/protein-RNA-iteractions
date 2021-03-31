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

ENV DGLBACKEND=pytorch

WORKDIR /app/src/web/flask

RUN chmod +x ./start.sh
# Run the API server using uwsgi
CMD ["./start.sh"]

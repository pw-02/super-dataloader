# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

# Use Alpine Linux as the base image
FROM ubuntu:22.04

ENV PYTHON_VERSION=3.10

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN apt update \
    && apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   zip \
                   #redis-server \ 
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda \
       python=$PYTHON_VERSION \
    && /opt/conda/bin/conda clean -ya

# Install requirements
COPY .devcontainer/requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements-dev.txt

# Set PYTHONPATH
ENV PYTHONPATH=/workspaces/super-dataloader
EXPOSE 50051

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
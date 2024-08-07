# Created by: George Corrêa de Araújo (george.gcac@gmail.com)
# ==================================================================

# FROM python:latest
FROM python:3.11

ARG GROUPID=901
ARG GROUPNAME=fasttext
ARG USERID=901
ARG USERNAME=user

# Environment variables

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install --upgrade" && \

# ==================================================================
# Create a system group with name deeplearning and id 901 to avoid
#    conflict with existing uids on the host system
# Create a system user with id 901 that belongs to group deeplearning
# ------------------------------------------------------------------

    groupadd -r $GROUPNAME -g $GROUPID && \
    # useradd -u $USERID -r -g $GROUPNAME $USERNAME && \
    useradd -u $USERID -m -g $GROUPNAME $USERNAME && \

# ==================================================================
# libraries via apt-get
# ------------------------------------------------------------------

    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        curl \
        git \
        locales \
        wget && \

# ==================================================================
# python libraries via pip
# ------------------------------------------------------------------

    $PIP_INSTALL \
        pip \
        wheel && \
    $PIP_INSTALL \
        colorama \
        comet-ml \
        fasttext \
        ipdb \
        ipython \
        numpy \
        pandas \
        prettytable \
        pyarrow \
        scikit-learn \
        top2vec \
        tqdm \
        unidecode \
        ydata-profiling && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sed -i -e 's/# pt_BR.UTF-8 UTF-8/pt_BR.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LC_ALL pt_BR.UTF-8
# handle bug with top2vec
# RuntimeError: cannot cache function 'rdist': no locator available for file '/usr/local/lib/python3.10/site-packages/umap/layouts.py'
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

USER $USERNAME

FROM nvcr.io/nvidia/pytorch:21.09-py3 
 
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN conda install -c conda-forge opencv
#RUN pip3 install opencv-python
#RUN pip3 install opencv-contlib-python
RUN pip3 install scikit-image
RUN pip3 install fsutils

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
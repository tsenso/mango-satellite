FROM continuumio/miniconda3

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# https://docs.python.org/3/using/cmdline.html
# We dont need to store byte code that used to speedup secondary runs in case of docker container
ENV PYTHONDONTWRITEBYTECODE 1
# We want to see stack trace when app fails
ENV PYTHONFAULTHANDLER 1
# WE don't want to bufferize app output, we want to see live logs
ENV PYTHONUNBUFFERED 1

ENV CONDA_ENV_NAME="new_gee_python" 
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "new_gee_python", "/bin/bash", "-c"]

# # Demonstrate the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

# The code to run when container is started:
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "new_gee_python", "python"]
CMD ["mhdp_server.py"]

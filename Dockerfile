FROM dlbase


COPY . /app
WORKDIR /app
RUN pip install datasets
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all



ENTRYPOINT ["python3"]
CMD ["-u","train.py"]
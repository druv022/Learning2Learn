FROM docker-rcods-projects-local.rt.artifactory.tio.systems/dl_base/dlbase:1


COPY . /app
WORKDIR /app

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all



ENTRYPOINT ["python3"]
CMD ["-u","train.py"]
FROM pytorch/pytorch:latest

# copy files
ADD scripts /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel

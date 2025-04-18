FROM spcleth/xaas:mpich as mpich
FROM spcleth/xaas:cuda-12.2 as cuda
FROM spcleth/xaas:omp-finder as omp-finder

FROM spcleth/xaas:llvm-19

COPY --from=mpich /opt/mpich /opt/mpich
COPY --from=cuda /usr/local/cuda-12.2 /usr/local/cuda-12.2
COPY --from=omp-finder /tools /tools

#USER root

RUN pip install tqdm

ENV PATH=/usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
ENV MPI_HOME=/opt/mpich

#RUN useradd -r docker_user && mkdir /build && chown docker_user:docker_user /build 
#USER docker_user
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ADD entrypoint.sh .
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]

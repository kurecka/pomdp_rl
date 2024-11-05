# ARG base=randriu/paynt
ARG base=xkurecka/paynt:rl
FROM $base

RUN pip install tqdm dill matplotlib pandas seaborn

RUN pip3 install jupyterhub
RUN pip install jupyterlab


WORKDIR /tmp
RUN jupyter server --generate-config
RUN jupyter lab clean

EXPOSE 8888

CMD ["start-notebook.py"]

COPY start-notebook.py start-notebook.sh start-singleuser.py start-singleuser.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-notebook.sh
RUN chmod +x /usr/local/bin/start-notebook.py
RUN chmod +x /usr/local/bin/start-singleuser.sh
RUN chmod +x /usr/local/bin/start-singleuser.py
COPY jupyter_server_config.py docker_healthcheck.py /etc/jupyter/
RUN chmod +x /etc/jupyter/docker_healthcheck.py

COPY copy-data.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/copy-data.sh

USER root
RUN chown -R 1000:1000 /etc/jupyter/


HEALTHCHECK --interval=3s --timeout=1s --start-period=3s --retries=3 \
CMD /etc/jupyter/docker_healthcheck.py || exit 1

WORKDIR /home/jovyan
ENV HOME=/home/jovyan

RUN chown -R 1000:1000 /home/jovyan
RUN chown -R 1000:1000 /home/ubuntu
USER 1000

# ENTRYPOINT [ "copy-data.sh" ]

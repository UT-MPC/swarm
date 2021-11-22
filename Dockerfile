FROM sethlee0111/private_swarm:server
EXPOSE 50051
WORKDIR swarm/dist_swarm
ENTRYPOINT ["python3", "simulate_device_server.py"]
# ENTRYPOINT ["/bin/bash", "-t", "-l", "-c", "./run_server.sh"]
RUN ["echo", "hello, device"]



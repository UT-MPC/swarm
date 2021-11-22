"""
resides in user side
"""

from kubernetes import client, config
import logging

from aws_settings import REGION

class K8sController():
    """
    k8s controller for distributed swarm simulation. Uses AWS EKS.
    Currently, one pod per node, which is called a worker
    """
    def __init__(self, eks_controller) -> None:
        self.eks_controller = eks_controller

        # Configs can be set in Configuration class directly or using helper utility
        config.load_kube_config()

        v1 = client.CoreV1Api()
        logging.info("Listing pods with their IPs:")
        ret = v1.list_namespaced_pod(namespace=self.eks_controller.get_namespace(), label_selector='app=dist-swarm')
        self.pods = {}
        for i in ret.items:
            self.pods[i.metadata.name] = i.status.pod_ip
            print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
        logging.info('Total {} pods read'.format(len(self.pods)))

        ret = v1.list_namespaced_pod(namespace=self.eks_controller.get_namespace(), label_selector='role=controller')
        for i in ret.items:
            logging.info("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

    def deploy_workers(self, tag, num):
        """
        tag: corresponds to cluster name in k8s
        """
        logging.info('deploying workers. This may take a while...')


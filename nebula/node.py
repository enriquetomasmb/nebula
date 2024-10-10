import os
import sys
import time
import random
import warnings
import numpy as np
import torch

torch.multiprocessing.set_start_method("spawn", force=True)

# Ignore CryptographyDeprecationWarning (datatime issues with cryptography library)
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nebula.config.config import Config
import logging
from nebula.core.datasets.mnist.mnist import MNISTDataset
from nebula.core.datasets.fashionmnist.fashionmnist import FashionMNISTDataset
from nebula.core.datasets.syscall.syscall import SYSCALLDataset
from nebula.core.datasets.cifar10.cifar10 import CIFAR10Dataset
from nebula.core.datasets.militarysar.militarysar import MilitarySARDataset
from nebula.core.datasets.datamodule import DataModule

from nebula.core.training.lightning import Lightning
from nebula.core.training.siamese import Siamese
from nebula.core.models.cifar10.dualagg import DualAggModel
from nebula.core.models.mnist.mlp import MNISTModelMLP
from nebula.core.models.mnist.cnn import MNISTModelCNN
from nebula.core.models.cifar10.cnnV3 import CIFAR10ModelCNN_V3
from nebula.core.models.fashionmnist.mlp import FashionMNISTModelMLP
from nebula.core.models.fashionmnist.cnn import FashionMNISTModelCNN
from nebula.core.models.syscall.mlp import SyscallModelMLP
from nebula.core.models.syscall.autoencoder import SyscallModelAutoencoder
from nebula.core.models.militarysar.cnn import MilitarySARModelCNN
from nebula.core.models.syscall.svm import SyscallModelSGDOneClassSVM
from nebula.core.engine import MaliciousNode, AggregatorNode, TrainerNode, ServerNode, IdleNode
from nebula.core.role import Role
from nebula.core.optimizations.communications.KD.models.cifar10.StudentCNN import StudentCIFAR10ModelCNN
from nebula.core.optimizations.communications.KD.models.cifar10.StudentResnet8 import StudentCIFAR10ModelResNet8
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentCNN import ProtoStudentCIFAR10ModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentResnet8 import ProtoStudentCIFAR10ModelResnet8
from nebula.core.optimizations.communications.KD.models.fashionmnist.StudentCNN import StudentFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoStudentCNN import ProtoStudentFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD.training.kdlightning import KDLightning
from nebula.core.optimizations.communications.KD_prototypes.training.protokdquantizationlightning import ProtoKDQuantizationLightning
from nebula.core.optimizations.communications.KD.models.mnist.StudentCNN import StudentMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoStudentCNN import ProtoStudentMNISTModelCNN
from nebula.core.research.FedGPD.models.cifar10.FedGPDCNN import FedGPDCIFAR10ModelCNN
from nebula.core.research.FedGPD.models.fashionmnist.FedGPDCNN import FedGPDFashionMNISTModelCNN
from nebula.core.research.FedGPD.models.mnist.FedGPDCNN import FedGPDMNISTModelCNN
from nebula.core.research.FedProto.models.cifar10.FedProtoCNN import FedProtoCIFAR10ModelCNN
from nebula.core.research.FedProto.models.cifar10.FedProtoResnet8 import FedProtoCIFAR10ModelResNet8
from nebula.core.research.FedProto.models.fashionmnist.FedProtoCNN import FedProtoFashionMNISTModelCNN
from nebula.core.research.FedProto.models.mnist.FedProtoCNN import FedProtoMNISTModelCNN
from nebula.core.research.FedProto.training.protolightning import ProtoLightning
from nebula.core.research.FML.models.mnist.FMLCombinedModel import FMLMNISTCombinedModelCNN
from nebula.core.research.FML.models.cifar10.FMLCombinedCNN import FMLCIFAR10CombinedModelCNN
from nebula.core.research.FML.models.cifar10.FMLCombinedResnet8 import FMLCIFAR10CombinedModelResNet8
from nebula.core.research.FML.models.cifar100.FMLCombinedResnet18 import FMLCIFAR100CombinedModelResNet18
from nebula.core.research.FML.models.fashionmnist.FMLCombinedModel import FMLFashionMNISTCombinedModelCNN
from nebula.core.research.FML.training.FMLlightning import FMLLightning
from nebula.core.datasets.cifar100.cifar100 import CIFAR100Dataset
from nebula.core.optimizations.communications.KD.models.cifar100.StudentResnet18 import StudentCIFAR100ModelResNet18
from nebula.core.optimizations.communications.KD_prototypes.models.cifar100.ProtoStudentResnet18 import ProtoStudentCIFAR100ModelResnet18
from nebula.core.research.FedGPD.models.cifar10.FedGPDResnet8 import FedGPDCIFAR10ModelResNet8
from nebula.core.research.FedGPD.models.cifar100.FedGPDResnet18 import FedGPDCIFAR100ModelResNet18
from nebula.core.research.FedProto.models.cifar100.FedProtoResnet18 import FedProtoCIFAR100ModelResNet18

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"


async def main():
    config_path = str(sys.argv[1])
    config = Config(entity="participant", participant_config_file=config_path)

    n_nodes = config.participant["scenario_args"]["n_nodes"]
    model_name = config.participant["model_args"]["model"]
    idx = config.participant["device_args"]["idx"]

    additional_node_status = config.participant["mobility_args"]["additional_node"]["status"]
    additional_node_round = config.participant["mobility_args"]["additional_node"]["round_start"]

    attacks = config.participant["adversarial_args"]["attacks"]
    poisoned_persent = config.participant["adversarial_args"]["poisoned_sample_percent"]
    poisoned_ratio = config.participant["adversarial_args"]["poisoned_ratio"]
    targeted = str(config.participant["adversarial_args"]["targeted"])
    target_label = config.participant["adversarial_args"]["target_label"]
    target_changed_label = config.participant["adversarial_args"]["target_changed_label"]
    noise_type = config.participant["adversarial_args"]["noise_type"]
    iid = config.participant["data_args"]["iid"]
    partition_selection = config.participant["data_args"]["partition_selection"]
    partition_parameter = np.array(config.participant["data_args"]["partition_parameter"], dtype=np.float64)
    label_flipping = False
    data_poisoning = False
    model_poisoning = False
    if attacks == "Label Flipping":
        label_flipping = True
        poisoned_ratio = 0
        if targeted == "true" or targeted == "True":
            targeted = True
        else:
            targeted = False
    elif attacks == "Sample Poisoning":
        data_poisoning = True
        if targeted == "true" or targeted == "True":
            targeted = True
        else:
            targeted = False
    elif attacks == "Model Poisoning":
        model_poisoning = True
    else:
        label_flipping = False
        data_poisoning = False
        targeted = False
        poisoned_persent = 0
        poisoned_ratio = 0

    # Adjust the total number of nodes and the index of the current node for CFL, as it doesn't require a specific partition for the server (not used for training)
    if config.participant["scenario_args"]["federation"] == "CFL":
        n_nodes -= 1
        if idx > 0:
            idx -= 1

    dataset = None
    dataset_str = config.participant["data_args"]["dataset"]
    num_workers = config.participant["data_args"]["num_workers"]
    model = None
    learner = None
    if dataset_str == "MNIST":
        dataset = MNISTDataset(
            num_classes=10,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        if model_name == "MLP":
            model = MNISTModelMLP()
        elif model_name == "CNN":
            model = MNISTModelCNN()
        elif model_name == "CNN KD":
            model = StudentMNISTModelCNN()
            learner = KDLightning
        elif model_name == "CNN KD-D":
            model = StudentMNISTModelCNN(decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN KD-S":
            model = StudentMNISTModelCNN(send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN KD-DS":
            model = StudentMNISTModelCNN(decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD":
            model = StudentMNISTModelCNN(mutual_distilation="MD")
            learner = KDLightning
        elif model_name == "CNN MD-D":
            model = StudentMNISTModelCNN(mutual_distilation="MD", decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN MD-S":
            model = StudentMNISTModelCNN(mutual_distilation="MD", send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD-DS":
            model = StudentMNISTModelCNN(mutual_distilation="MD", decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN FedProto":
            model = FedProtoMNISTModelCNN()
            learner = ProtoLightning
        elif model_name == "CNN FedGPD":
            model = FedGPDMNISTModelCNN()
        elif model_name == "CNN FML":
            model = FMLMNISTCombinedModelCNN()
            learner = FMLLightning
        elif model_name == "CNN PKD":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-S":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-D":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-A":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-DS":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD", send_logic="mixed_2rounds", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-AS":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="KD", send_logic="mixed_2rounds", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-S":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-D":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-A":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-DS":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD", send_logic="mixed_2rounds", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-AS":
            model = ProtoStudentMNISTModelCNN(knowledge_distilation="MD", send_logic="mixed_2rounds", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        else:
            raise ValueError(f"Model {model} not supported for dataset {dataset_str}")
    elif dataset_str == "FashionMNIST":
        dataset = FashionMNISTDataset(
            num_classes=10,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        if model_name == "MLP":
            model = FashionMNISTModelMLP()
        elif model_name == "CNN":
            model = FashionMNISTModelCNN()
        elif model_name == "CNN KD":
            model = StudentFashionMNISTModelCNN()
            learner = KDLightning
        elif model_name == "CNN KD-D":
            model = StudentFashionMNISTModelCNN(decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN KD-S":
            model = StudentFashionMNISTModelCNN(send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN KD-DS":
            model = StudentFashionMNISTModelCNN(decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD":
            model = StudentFashionMNISTModelCNN(mutual_distilation="MD")
            learner = KDLightning
        elif model_name == "CNN MD-D":
            model = StudentFashionMNISTModelCNN(mutual_distilation="MD", decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN MD-S":
            model = StudentFashionMNISTModelCNN(mutual_distilation="MD", send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD-DS":
            model = StudentFashionMNISTModelCNN(mutual_distilation="MD", decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN FedProto":
            model = FedProtoFashionMNISTModelCNN()
            learner = ProtoLightning
        elif model_name == "CNN FedGPD":
            model = FedGPDFashionMNISTModelCNN()
        elif model_name == "CNN FML":
            model = FMLFashionMNISTCombinedModelCNN()
            learner = FMLLightning
        elif model_name == "CNN PKD":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-D":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-A":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-S":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-DS":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-AS":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="MD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-D":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="MD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-A":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="MD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-S":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="MD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-DS":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-AS":
            model = ProtoStudentFashionMNISTModelCNN(knowledge_distilation="KD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        else:
            raise ValueError(f"Model {model} not supported for dataset {dataset_str}")
    elif dataset_str == "SYSCALL":
        dataset = SYSCALLDataset(
            num_classes=10,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        if model_name == "MLP":
            model = SyscallModelMLP()
        elif model_name == "SVM":
            model = SyscallModelSGDOneClassSVM()
        elif model_name == "Autoencoder":
            model = SyscallModelAutoencoder()
        else:
            raise ValueError(f"Model {model} not supported for dataset {dataset_str}")
    elif dataset_str == "CIFAR10":
        dataset = CIFAR10Dataset(
            num_classes=10,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        if model_name == "CNN":
            model = CIFAR10ModelCNN_V3()
        elif model_name == "CNN KD":
            model = StudentCIFAR10ModelCNN()
            learner = KDLightning
        elif model_name == "CNN KD-D":
            model = StudentCIFAR10ModelCNN(decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN KD-S":
            model = StudentCIFAR10ModelCNN(send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN KD-DS":
            model = StudentCIFAR10ModelCNN(decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD":
            model = StudentCIFAR10ModelCNN(mutual_distilation="MD")
            learner = KDLightning
        elif model_name == "CNN MD-D":
            model = StudentCIFAR10ModelCNN(mutual_distilation="MD", decreasing_beta=True)
            learner = KDLightning
        elif model_name == "CNN MD-S":
            model = StudentCIFAR10ModelCNN(mutual_distilation="MD", send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN MD-DS":
            model = StudentCIFAR10ModelCNN(mutual_distilation="MD", decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "CNN FedProto":
            model = FedProtoCIFAR10ModelCNN()
            learner = ProtoLightning
        elif model_name == "CNN FedGPD":
            model = FedGPDCIFAR10ModelCNN()
        elif model_name == "CNN FML":
            model = FMLCIFAR10CombinedModelCNN()
            learner = FMLLightning
        elif model_name == "CNN PKD":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-D":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-A":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-S":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-DS":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PKD-AS":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="KD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-D":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-A":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-S":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-DS":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "CNN PMD-AS":
            model = ProtoStudentCIFAR10ModelCNN(knowledge_distilation="MD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8":
            model = StudentCIFAR10ModelResNet8(mutual_distilation=None)
            learner = KDLightning
        elif model_name == "Resnet8 KD":
            model = StudentCIFAR10ModelResNet8()
            learner = KDLightning
        elif model_name == "Resnet8 KD-D":
            model = StudentCIFAR10ModelResNet8(decreasing_beta=True)
            learner = KDLightning
        elif model_name == "Resnet8 KD-S":
            model = StudentCIFAR10ModelResNet8(send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet8 KD-DS":
            model = StudentCIFAR10ModelResNet8(decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet8 MD":
            model = StudentCIFAR10ModelResNet8(mutual_distilation="MD")
            learner = KDLightning
        elif model_name == "Resnet8 MD-D":
            model = StudentCIFAR10ModelResNet8(mutual_distilation="MD", decreasing_beta=True)
            learner = KDLightning
        elif model_name == "Resnet8 MD-S":
            model = StudentCIFAR10ModelResNet8(mutual_distilation="MD", send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet8 MD-DS":
            model = StudentCIFAR10ModelResNet8(mutual_distilation="MD", decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet8 FedProto":
            model = FedProtoCIFAR10ModelResNet8()
            learner = ProtoLightning
        elif model_name == "Resnet8 FedGPD":
            model = FedGPDCIFAR10ModelResNet8()
        elif model_name == "Resnet8 FML":
            model = FMLCIFAR10CombinedModelResNet8()
            learner = FMLLightning
        elif model_name == "Resnet8 PKD":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PKD-D":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PKD-A":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PKD-S":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PKD-DS":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PKD-AS":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="KD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-D":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-A":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-S":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-DS":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-AS":
            model = ProtoStudentCIFAR10ModelResnet8(knowledge_distilation="MD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        else:
            raise ValueError(f"Model {model} not supported for dataset {dataset_str}")
    elif dataset_str == "MilitarySAR":
        dataset = MilitarySARDataset(
            num_classes=10,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        model = MilitarySARModelCNN()

    elif dataset_str == "CIFAR100":
        dataset = CIFAR100Dataset(
            num_classes=100,
            partition_id=idx,
            partitions_number=n_nodes,
            iid=iid,
            partition=partition_selection,
            partition_parameter=partition_parameter,
            seed=42,
            config=config,
        )
        if model_name == "Resnet18":
            model = StudentCIFAR100ModelResNet18(mutual_distilation=None)
            learner = KDLightning
        elif model_name == "Resnet18 KD":
            model = StudentCIFAR100ModelResNet18()
            learner = KDLightning
        elif model_name == "Resnet18 KD-D":
            model = StudentCIFAR100ModelResNet18(decreasing_beta=True)
            learner = KDLightning
        elif model_name == "Resnet18 KD-S":
            model = StudentCIFAR100ModelResNet18(send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet18 KD-DS":
            model = StudentCIFAR100ModelResNet18(decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet18 MD":
            model = StudentCIFAR100ModelResNet18(mutual_distilation="MD")
            learner = KDLightning
        elif model_name == "Resnet18 MD-D":
            model = StudentCIFAR100ModelResNet18(mutual_distilation="MD", decreasing_beta=True)
            learner = KDLightning
        elif model_name == "Resnet18 MD-S":
            model = StudentCIFAR100ModelResNet18(mutual_distilation="MD", send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet18 MD-DS":
            model = StudentCIFAR100ModelResNet18(mutual_distilation="MD", decreasing_beta=True, send_logic="mixed_2rounds")
            learner = KDLightning
        elif model_name == "Resnet18 FedProto":
            model = FedProtoCIFAR100ModelResNet18()
            learner = ProtoLightning
        elif model_name == "Resnet18 FedGPD":
            model = FedGPDCIFAR100ModelResNet18()
        elif model_name == "Resnet18 FML":
            model = FMLCIFAR100CombinedModelResNet18()
            learner = FMLLightning
        elif model_name == "Resnet18 PKD":
            model = ProtoStudentCIFAR100ModelResnet18()
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PKD-D":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="KD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PKD-A":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="KD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PKD-S":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="KD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PKD-DS":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="KD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PKD-AS":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="KD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PMD":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PMD-D":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD", weighting="decreasing")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PMD-A":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD", weighting="adaptative")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet8 PMD-S":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PMD-DS":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD", weighting="decreasing", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        elif model_name == "Resnet18 PMD-AS":
            model = ProtoStudentCIFAR100ModelResnet18(knowledge_distilation="MD", weighting="adaptative", send_logic="mixed_2rounds")
            learner = ProtoKDQuantizationLightning
        else:
            raise ValueError(f"Model {model} not supported")

    else:
        raise ValueError(f"Dataset {dataset_str} not supported")

    dataset = DataModule(
        train_set=dataset.train_set,
        train_set_indices=dataset.train_indices_map,
        test_set=dataset.test_set,
        test_set_indices=dataset.test_indices_map,
        local_test_set_indices=dataset.local_test_indices_map,
        num_workers=num_workers,
        partition_id=idx,
        partitions_number=n_nodes,
        batch_size=dataset.batch_size,
        label_flipping=label_flipping,
        data_poisoning=data_poisoning,
        poisoned_persent=poisoned_persent,
        poisoned_ratio=poisoned_ratio,
        targeted=targeted,
        target_label=target_label,
        target_changed_label=target_changed_label,
        noise_type=noise_type,
    )

    # - Import MNISTDatasetScikit (not torch component)
    # - Import scikit-learn model
    # - Import ScikitDataModule
    # - Import Scikit as trainer
    # - Import aggregation algorithm adapted to scikit-learn models (e.g. FedAvgSVM)
    if learner is not None:
        trainer = learner
    else:
        trainer_str = config.participant["training_args"]["trainer"]
        if trainer_str == "lightning":
            trainer = Lightning
        elif trainer_str == "scikit":
            raise NotImplementedError
        elif trainer_str == "siamese" and dataset_str == "CIFAR10":
            trainer = Siamese
            model = DualAggModel()
            config.participant["model_args"]["model"] = "DualAggModel"
            config.participant["data_args"]["dataset"] = "CIFAR10"
            config.participant["aggregator_args"]["algorithm"] = "DualHistAgg"
        else:
            raise ValueError(f"Trainer {trainer_str} not supported")

    if config.participant["device_args"]["malicious"]:
        node_cls = MaliciousNode
    else:
        if config.participant["device_args"]["role"] == Role.AGGREGATOR:
            node_cls = AggregatorNode
        elif config.participant["device_args"]["role"] == Role.TRAINER:
            node_cls = TrainerNode
        elif config.participant["device_args"]["role"] == Role.SERVER:
            node_cls = ServerNode
        elif config.participant["device_args"]["role"] == Role.IDLE:
            node_cls = IdleNode
        else:
            raise ValueError(f"Role {config.participant['device_args']['role']} not supported")

    VARIABILITY = 0.5

    def randomize_value(value, variability):
        min_value = max(0, value - variability)
        max_value = value + variability
        return random.uniform(min_value, max_value)

    config_keys = [
        ["reporter_args", "report_frequency"],
        ["discoverer_args", "discovery_frequency"],
        ["health_args", "health_interval"],
        ["health_args", "grace_time_health"],
        ["health_args", "check_alive_interval"],
        ["health_args", "send_alive_interval"],
        ["forwarder_args", "forwarder_interval"],
        ["forwarder_args", "forward_messages_interval"],
    ]

    for keys in config_keys:
        value = config.participant
        for key in keys[:-1]:
            value = value[key]
        value[keys[-1]] = randomize_value(value[keys[-1]], VARIABILITY)

    logging.info(f"Starting node {idx} with model {model_name}, trainer {trainer.__name__}, and as {node_cls.__name__}")

    node = node_cls(
        model=model,
        dataset=dataset,
        config=config,
        trainer=trainer,
        security=False,
        model_poisoning=model_poisoning,
        poisoned_ratio=poisoned_ratio,
        noise_type=noise_type,
    )
    await node.start_communications()
    await node.deploy_federation()

    # If it is an additional node, it should wait until additional_node_round to connect to the network
    # In order to do that, it should request the current round to the controller
    if additional_node_status:
        logging.info(f"Waiting for round {additional_node_round} to start")
        time.sleep(6000)  # DEBUG purposes
        import requests

        url = f'http://{node.config.participant["scenario_args"]["controller"]}/platform/{node.config.participant["scenario_args"]["name"]}/round'
        current_round = int(requests.get(url).json()["round"])
        while current_round < additional_node_round:
            logging.info(f"Waiting for round {additional_node_round} to start")
            time.sleep(10)
        logging.info(f"Round {additional_node_round} started, connecting to the network")

    if node.cm is not None:
        await node.cm.network_wait()


if __name__ == "__main__":
    if sys.platform == "win32":
        import asyncio

        asyncio.run(main(), debug=False)
    else:
        try:
            import uvloop 
            uvloop.run(main(), debug=False)
        except ImportError:
            logging.warning("uvloop not available, using default loop")
            import asyncio
            asyncio.run(main(), debug=False)

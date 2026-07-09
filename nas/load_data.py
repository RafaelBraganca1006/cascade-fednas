from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.standalone.superfednas.Server.ServerModel import ServerResnet_10_26

data_loader = load_partition_data_cifar10
(
    train_data_num,
    test_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    class_num,
) = data_loader(
    "cifar10",
    "data/cifar10",
    "hetero",
    100,
    1,
    64,
)

# dataset = [
#         train_data_num,
#         test_data_num,
#         train_data_global,
#         test_data_global,
#         train_data_local_num_dict,
#         train_data_local_dict,
#         test_data_local_dict,
#         class_num,
#     ]

dataset = test_data_local_dict[0]

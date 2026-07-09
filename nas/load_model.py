import torch
from fedml_api.standalone.superfednas.Server.ServerModel import ServerResnet_10_26

def custom_ofa_net(num_classes, config, client_num_in_total):
    params = dict()
    params["n_classes"] = num_classes
    if config is not None:
        if "d" in config.keys() and config["d"] is not None:
            params["depth_list"] = config["d"]
        if "e" in config.keys() and config["e"] is not None:
            params["expand_ratio_list"] = config["e"]
        if "w" in config.keys() and config["w"] is not None:
            params["width_mult_list"] = config["w"]

    model = ServerResnet_10_26(params, None, client_num_in_total, None, None,)
    return model

def load_model(ckpt_path, dataset):
    checkpoint = torch.load(ckpt_path)
    if "params" in checkpoint:
        server_model = custom_ofa_net(
            10,
            {"d":[0,1,2],
            "e":[0.1, 0.14, 0.18, 0.22, 0.25]},
            20
        )
        server_model.set_model_params(checkpoint["params"])
    else:
        raise ValueError("params doesn't exist in checkpoint")

    return server_model

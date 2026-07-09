from fedml_api.standalone.superfednas.Server.ServerTrainer.superfednas_trainer import FLOFA_Trainer


class ClientSupernetTrainer(FLOFA_Trainer):
    def __init__(
        self,
        server_model,
        dataset,
        client_trainer,
        args,
        lr_scheduler,
        teacher_model=None,
        start_round=0,
    ):
        super(ClientSupernetTrainer, self).__init__(
            server_model,
            dataset,
            client_trainer,
            args,
            lr_scheduler,
            teacher_model=teacher_model,
            start_round=start_round,
        )

    def _aggregate(self, w_locals):
        averaged_params = w_locals[0].state_dict()
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i].state_dict()
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]
        return averaged_params

import random
import numpy as np

__all__ = ["MobileNetArchEncoder", "ResNetArchEncoder"]

class ResNetArchEncoder:
    def __init__(
        self,
        image_size_list=None,
    ):
        self.image_size_list = [32] if image_size_list is None else image_size_list
        self.expand_list = [0.1, 0.14, 0.18, 0.22, 0.25]
        self.depth_list = [0, 1, 2]
        self.width_mult_list = ([0.65, 0.8, 1.0])
        self.base_depth_list =  [1, 1, 1, 1]

        """" build info dict """
        self.n_dim = 0

        # resolution
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")

        # input stem skip
        self.input_stem_d_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="input_stem_d")

        # width_mult
        self.width_mult_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="width_mult")

        # expand ratio
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")

    @property
    def n_stage(self):
        return len(self.base_depth_list)

    @property
    def max_n_blocks(self):
        return sum(self.base_depth_list) + self.n_stage * max(self.depth_list)

    def _build_info_dict(self, target):
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)

        elif target == "input_stem_d":
            target_dict = self.input_stem_d_info
            target_dict["L"].append(self.n_dim)
            for skip in [0, 1]:
                target_dict["val2id"][skip] = self.n_dim
                target_dict["id2val"][self.n_dim] = skip
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)

        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_list
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for e in choices:
                    target_dict["val2id"][i][e] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = e
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        d, e, r = (
            arch_dict["d"],
            arch_dict["e"],
            arch_dict["image_size"],
        )
        input_stem_skip = 1 if d[0] > 0 else 0
        feature = np.zeros(self.n_dim)
        feature[self.r_info["val2id"][r]] = 1
        feature[self.input_stem_d_info["val2id"][input_stem_skip]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d[i]
            for j in range(start_pt, start_pt + depth):
                feature[self.e_info["val2id"][j][e[j]]] = 1
            start_pt += max(self.depth_list) + base_depth

        return feature

    def feature2arch(self, feature):
        img_sz = self.r_info["id2val"][
            int(np.argmax(feature[self.r_info["L"][0] : self.r_info["R"][0]]))
            + self.r_info["L"][0]
        ]
        input_stem_skip = (
            self.input_stem_d_info["id2val"][
                int(
                    np.argmax(
                        feature[
                            self.input_stem_d_info["L"][0] : self.input_stem_d_info[
                                "R"
                            ][0]
                        ]
                    )
                )
                + self.input_stem_d_info["L"][0]
            ]
            * 2
        )
        assert img_sz in self.image_size_list
        arch_dict = {"d": [input_stem_skip], "e": [], "image_size": img_sz}

        d = 0
        skipped = 0
        stage_id = 0
        for i in range(self.max_n_blocks):
            skip = True
            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    skip = False
                    break
            if skip:
                arch_dict["e"].append(0)
                skipped += 1
            else:
                d += 1

            if (
                i + 1 == self.max_n_blocks
                or (skipped + d)
                % (max(self.depth_list) + self.base_depth_list[stage_id])
                == 0
            ):
                arch_dict["d"].append(d - self.base_depth_list[stage_id])
                d, skipped = 0, 0
                stage_id += 1

        return arch_dict

    def random_sample_arch(self):
        return {
            "d": [random.choice([0, 2])]
            + random.choices(self.depth_list, k=self.n_stage),
            "e": random.choices(self.expand_list, k=self.max_n_blocks),
            "image_size": random.choice(self.image_size_list),
        }

    def mutate_resolution(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["image_size"] = random.choice(self.image_size_list)
        return arch_dict

    def mutate_arch(self, arch_dict, mutate_prob):
        # input stem skip
        if random.random() < mutate_prob:
            arch_dict["d"][0] = random.choice([0, 2])

        # depth
        for i in range(1, len(arch_dict["d"])):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)

        # expand ratio
        for i in range(len(arch_dict["e"])):
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_list)

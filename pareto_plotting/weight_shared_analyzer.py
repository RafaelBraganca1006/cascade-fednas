import wandb
from collections import defaultdict
import pandas as pd
import os
from pprint import pprint
import math


class WeightSharedMetric:
    def __init__(
        self,
        df,
        metric_key,
        pareto_models,
        pareto_labels,
        larger_is_better=True,
        error_df=None,
    ):
        self.df = df
        self.error_df = error_df
        self.metric_key = metric_key
        self.pareto_models = pareto_models
        self.pareto_labels = pareto_labels
        self.larger_is_better = larger_is_better
        self._init_metric_columns()

    def _init_metric_columns(self):
        metric_columns = list()
        metrics_mapping = dict()
        self.largest_model, self.smallest_model = None, None
        self.largest_model_index, self.smallest_model_index = None, None
        for model in self.pareto_models:
            model_metric_key = self.metric_key.replace("{model}", f"{model}")
            metric_columns.append(model_metric_key)
            label = self.pareto_labels[f"{model}"]
            if label == "largest":
                self.largest_model = model_metric_key
                self.largest_model_index = len(metric_columns) - 1
            elif label == "smallest":
                self.smallest_model = model_metric_key
                self.smallest_model_index = len(metric_columns) - 1

            metrics_mapping[model_metric_key] = label
        self.metrics_mapping = metrics_mapping
        self.metric_columns = metric_columns

    def _select_columns(self, columns):
        if not isinstance(columns, list):
            columns = [columns]
        return self.df.columns.intersection(set(columns))

    def dataframe(self, model=None):
        if model is None:
            return self.df[self.metric_columns + ["round"]]
        model_key = self.metric_key.replace("{model}", f"{model}")
        return self.df[[model_key, "round"]]

    def get_best_pareto_curve(self, uptil_round=None, mode="avg"):
        row_index, column_index = None, None
        if uptil_round is None:
            df = self.df
            error_df = self.error_df
        else:
            df = self.df.loc[self.df["round"] <= uptil_round, :]
            if self.error_df is not None:
                error_df = self.error_df.loc[
                    self.error_df["round"] <= uptil_round, :
                ]
            else:
                error_df = None

        if mode == "avg":
            if self.larger_is_better:
                row_index = (
                    df[self._select_columns(self.metric_columns)]
                    .mean(axis=1)
                    .argmax()
                )
                column_index = self._select_columns(self.metric_columns)
            else:
                row_index = (
                    df[self._select_columns(self.metric_columns)]
                    .mean(axis=1)
                    .argmin()
                )
                column_index = self._select_columns(self.metric_columns)

        elif mode == "largest":
            if self.larger_is_better:
                row_index = (
                    df[
                        self._select_columns(
                            self.metric_columns[self.largest_model_index]
                        )
                    ]
                    .squeeze()
                    .argmax()
                )
                column_index = self._select_columns(self.metric_columns)

            else:
                row_index = (
                    df[
                        self._select_columns(
                            self.metric_columns[self.largest_model_index]
                        )
                    ]
                    .squeeze()
                    .argmin()
                )
                column_index = self._select_columns(self.metric_columns)

        elif mode == "smallest":
            if self.larger_is_better:
                row_index = (
                    df[
                        self._select_columns(
                            self.metric_columns[self.smallest_model_index]
                        )
                    ]
                    .squeeze()
                    .argmax()
                )
                column_index = self._select_columns(self.metric_columns)

            else:
                row_index = (
                    df[
                        self._select_columns(
                            self.metric_columns[self.smallest_model_index]
                        )
                    ]
                    .squeeze()
                    .argmin()
                )
                column_index = self._select_columns(self.metric_columns)
        #print(row_index, column_index)
        if error_df is not None:
            return (
                df.loc[row_index, column_index],
                error_df.loc[row_index, column_index],
            )
        return df.loc[row_index, column_index], None

    def plot_best_pareto_curve(self, uptil_round=None, mode="avg"):
        df, error_df = self.get_best_pareto_curve(
            uptil_round=uptil_round, mode=mode
        )
        df.rename(self.metrics_mapping).plot.line(yerr=error_df)

    def get_multi_best_pareto_curves(self, rounds=None, mode="avg"):
        if rounds is None or isinstance(rounds, int):
            self.plot_best_pareto_curve(self, uptil_round=rounds, mode=mode)

        pareto_values = []
        pareto_error_values = []
        for rnum in rounds:
            df, error_df = self.get_best_pareto_curve(rnum)
            pareto_values.append(df.rename(rnum))
            if error_df is not None:
                pareto_error_values.append(error_df.rename(rnum))
            else:
                pareto_error_values.append(None)

        if all([v is not None for v in pareto_error_values]):
            return pd.concat(pareto_values, axis=1), None

        return pd.concat(pareto_values, axis=1), pd.concat(
            pareto_error_values, axis=1
        )

    def plot_multi_best_pareto_curves(self, rounds, mode="avg"):
        df, error_df = self.get_multi_best_pareto_curves(
            rounds, mode=mode
        ).rename(index=self.metrics_mapping)
        df.plot.line(yerr=error_df)


class WeightSharedRun:
    def __init__(self, run_ids, pareto_models, pareto_labels, name=None):
        if isinstance(run_ids, list):
            self.ws_run = list()
            for run_id in run_ids:
                self.ws_run.append(wandb.Api().run(run_id))
        else:
            self.ws_run = [wandb.Api().run(run_ids)]
        self.pareto_models = pareto_models
        self.pareto_labels = pareto_labels
        self._create_df()
        self.metrics_dict = dict()
        self._name = name if name is not None else self.ws_run[0].name

    def _create_df(self):
        df_list = list()
        for run in self.ws_run:
            tracker = defaultdict(list)
            for i, row in run.history().iterrows():
                for index, value in row.items():
                    if not isinstance(value, dict) and not (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        if index == 'Test/Mean/PPL':
                            continue
                        elif index == 'Test/Mean/Acc':
                            continue
                        elif "parameters" in index:
                            continue
                        tracker[index].append(value)

            try:
                df_list.append(pd.DataFrame(tracker))
            except Exception as e:
                print(e)
                pprint({k: len(v) for k, v in tracker.items()})
                raise e
        final_df = pd.concat(df_list)
        group_final_df = final_df.groupby(final_df.index)
        self.df = group_final_df.mean()
        self.error_df = group_final_df.std().fillna(0)

    def metric(self, key, larger_is_better=True):
        if key not in self.metrics_dict:
            self.metrics_dict[key] = WeightSharedMetric(
                self.df,
                key,
                self.pareto_models,
                self.pareto_labels,
                larger_is_better,
                self.error_df,
            )
        return self.metrics_dict[key]

    def save(self, model_dir=""):
        filename = os.path.join(model_dir, self.ws_run.name)
        self.df.to_csv(filename)

    @property
    def name(self):
        return self._name


def get_pareto_curves_df(
    run_list,
    metric_key,
    larger_is_better=True,
    uptil_round=None,
    rename=False,
    mode="avg",
    axis=1,
    name=None,
):
    pareto_values = []
    pareto_error_values = []
    metrics_mapping = None
    for run in run_list:
        metric_info = run.metric(metric_key, larger_is_better)
        df, error_df = metric_info.get_best_pareto_curve(
            uptil_round, mode=mode
        )
        if metrics_mapping is None and rename:
            metrics_mapping = metric_info.metrics_mapping
        pareto_values.append(df.rename(run.name))
        if error_df is not None:
            pareto_error_values.append(error_df.rename(run.name))
        else:
            pareto_values.append(None)
    final_df = None
    final_error_df = None
    if rename:
        final_df = pd.concat(pareto_values, axis=axis).rename(
            index=metrics_mapping
        )
        if all([v is not None for v in pareto_error_values]):
            final_error_df = pd.concat(pareto_error_values, axis=axis).rename(
                index=metrics_mapping
            )
    else:
        final_df = pd.concat(pareto_values, axis=axis).rename(None)
        if all([v is not None for v in pareto_error_values]):
            final_error_df = pd.concat(pareto_error_values, axis=axis).rename(
                None
            )
    if name is not None:
        final_df = final_df.rename(name)
        if final_error_df is not None:
            final_error_df = final_error_df.rename(name)
    return final_df, final_error_df


def plot_pareto_curves_ws(
    run_list,
    metric_key,
    larger_is_better=True,
    uptil_round=None,
    title=None,
    mode="avg",
    axis=1,
    name=None,
):
    df, error_df = get_pareto_curves_df(
        run_list,
        metric_key,
        larger_is_better,
        uptil_round,
        rename=True,
        mode=mode,
        axis=axis,
        name=name,
    )
    if title is not None:
        df.plot.line(title=title, yerr=error_df)
    else:
        df.plot.line(yerr=error_df)

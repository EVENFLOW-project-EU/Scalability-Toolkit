import glob
import os
import re
from datetime import datetime, timedelta

import pandas as pd
from dateutil import parser


class EvenFlowLogsAndMetricsAgg:
    def __init__(self, metrics_path: str, logs_path: str, num_nodes: int) -> None:
        self.metrics_path = metrics_path
        self.logs_path = logs_path
        self.num_nodes = num_nodes

    def aggregate_logs(self) -> None:
        all_files: list[str] = glob.glob(os.path.join(self.logs_path, "*.log"))
        log_lines_ts: dict[float, str] = {}

        for file in all_files:
            with open(file, "r+", encoding="UTF-8") as f:
                for line in f.readlines():
                    try:
                        line = line.replace("\n", " ")
                        matches: list[str] = line.split(None, 3)
                        ts, worker_name, _, msg = matches
                        ts_dt: datetime = parser.parse(ts)

                        if ts_dt.timestamp() in log_lines_ts:
                            worker_id: int = (
                                re.match(r"trainer(\d+)", worker_name).groups()[0] if worker_name != "ps" else 999
                            )
                            ts_dt += timedelta(microseconds=int(worker_id))

                        log_lines_ts[ts_dt.timestamp()] = {"time": ts_dt, "name": worker_name, "message": msg}
                    except Exception:
                        pass

        df: pd.DataFrame = (
            pd.DataFrame.from_dict(log_lines_ts, orient="index").reset_index(drop=True).set_index("time").sort_index()
        )

        os.makedirs(f"{self.logs_path}/aggregated/", exist_ok=True)
        df.to_csv(f"{self.logs_path}/aggregated/all_logs.csv")

    def aggregate_metrics(self) -> None:
        all_files: list[str] = glob.glob(os.path.join(self.metrics_path, "*.csv"))

        all_files_df: list[pd.DataFrame] = []
        for file in all_files:
            try:
                _df: pd.DataFrame = pd.read_csv(file)
                all_files_df.append(_df)
            except BaseException:
                pass
        df: pd.DataFrame = pd.concat(all_files_df)
        df["date"] = pd.to_datetime(df["ts"], unit="s")
        df.drop("ts", axis="columns", inplace=True)

        df: pd.DataFrame = df.reset_index(drop=True).set_index("date").sort_index()

        os.makedirs(f"{self.metrics_path}/aggregated", exist_ok=True)
        df.to_csv(f"{self.metrics_path}/aggregated/all_metrics.csv", index=True)

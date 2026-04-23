import pandas as pd
import random
import time

from common.constants import (
    COMPONENTS,
    WINDOW_SIZE,
    ALL_LEVELS,
    ERROR_LEVELS,
    NORMAL_LEVELS,
    ANCHOR_FILE_PATH_TRAIN,
    POSITIVE_FILE_PATH_TRAIN,
    NEGATIVE_FILE_PATH_TRAIN,
    ANCHOR_FILE_PATH_VALID,
    POSITIVE_FILE_PATH_VALID,
    NEGATIVE_FILE_PATH_VALID,
)


class SyntheticLogGenerator:
    def __init__(self, start_time=None):
        if start_time is None:
            self.last_ts = time.time()

        self.batch_gap = (
            600.0  # 10 mins gap between windows to prevent cross-batch alerts
        )

    def generate_window(self, scenario="healthy"):
        logs = []
        # Start this window significantly AFTER the previous window ended
        current_time = self.last_ts + self.batch_gap

        # Select a 'target' component for the scenario
        target_comp = random.choice(COMPONENTS)
        other_comps = [c for c in COMPONENTS if c != target_comp]
        # Randomly pick where the pattern starts (0 to 8 so there's room for index+1)
        pattern_start = random.randint(0, WINDOW_SIZE - 2)
        seen_comps = {}
        prev_comp = None
        max_anomalies_count = random.choices(range(2, 7))[0]
        anomaly_count = 0
        prev_anomaly_loc = None

        for i in range(WINDOW_SIZE):
            # 1. SET DEFAULT "BACKGROUND NOISE" VALUES
            gap = random.uniform(1, 100)
            lvl = random.choices(ALL_LEVELS, weights=[70, 15, 10, 3, 2])[0]
            comp = random.choice(COMPONENTS)

            if scenario == "healthy_time":
                gap = random.uniform(301, 600)
                # Rule: Spread out logs, mostly low severity
                current_time += gap  # > 300s gap
                logs.append(
                    {
                        "timestamp": current_time,
                        "component": random.choice(COMPONENTS),
                        "level": random.choices(ALL_LEVELS, weights=[70, 20, 5, 3, 2])[
                            0
                        ],
                    }
                )
            elif scenario == "healthy_level":
                gap = random.uniform(1, 60)
                # Rule: Spread out logs, mostly low severity
                current_time += gap  # > 300s gap
                logs.append(
                    {
                        "timestamp": current_time,
                        "component": random.choice(COMPONENTS),
                        "level": random.choices(NORMAL_LEVELS, weights=[34, 33, 33])[0],
                    }
                )
            elif scenario == "single_fatal":
                # Rule: A single FATAL is NOT an abnormality
                # ensure that we have utmost single ERROR level for a comp, and if
                # there are multiple error levels for same comp ensure they are separated by > 300 seconds
                lvl = random.choices(ERROR_LEVELS)[0]
                if prev_comp is None:
                    current_comp = target_comp
                else:
                    current_comp = random.choices(
                        [c for c in COMPONENTS if c != prev_comp]
                    )[0]

                if current_comp not in seen_comps:
                    seen_comps[current_comp] = True
                    gap = random.uniform(10, 50)
                else:
                    gap = random.uniform(301, 600)
                current_time += gap
                prev_comp = current_comp
                logs.append(
                    {"timestamp": current_time, "component": current_comp, "level": lvl}
                )
            elif scenario == "cross_component_burst":
                # Rule: Two FATALs or SEVEREs within 300s from DIFFERENT components is NOT abnormal
                current_time += gap
                if i == pattern_start:
                    lvl = random.choices(ERROR_LEVELS, weights=[70, 30])[0]
                    comp = target_comp
                    gap = random.uniform(1.0, 5.0)
                elif i == pattern_start + 1:
                    lvl = random.choices(ERROR_LEVELS, weights=[60, 40])[0]
                    comp = random.choice(other_comps)  # Different Component
                    gap = random.uniform(0.1, 3.0)  # Very fast, but different comp
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "slow_fatals":
                # Rule: Two FATALs > 300s apart from SAME component is NOT abnormal
                current_time += gap
                if i == pattern_start:
                    lvl = random.choices(ERROR_LEVELS, weights=[70, 30])[0]
                    comp = target_comp
                    gap = random.uniform(1.0, 5.0)  # Random entry
                if i == pattern_start + 1:
                    lvl = random.choices(ERROR_LEVELS, weights=[60, 40])[0]
                    comp = target_comp
                    gap = random.uniform(301.0, 600.0)  # normal gap > 300.0
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "single_comp_all_good":
                current_time += gap
                lvl = random.choices(NORMAL_LEVELS, weights=[34, 33, 33])[0]
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "anomaly":
                # Rule: THE TRIGGER - Two are more high-severity logs, SAME component within 300s
                # within the batch. Not necessarily successive
                # even when error level is non-severe keeping gap < 299 so that when sorted
                # logs with severe level aren't always pused to the end
                gap = random.uniform(1.0, 299.0)
                if i == pattern_start:
                    lvl = random.choices(ERROR_LEVELS)[0]
                    comp = target_comp
                    anomaly_count += 1
                    prev_anomaly_loc = i
                else:
                    random_locs = [k for k in range(i + 1) if k != pattern_start]
                    v = random.randint(0, 1)
                    # randomly insert error severity log at start
                    if len(random_locs) == 1 and v:
                        lvl = random.choices(ERROR_LEVELS)[0]
                        comp = target_comp
                        anomaly_count += 1
                        prev_anomaly_loc = i

                    # randomizing so that next error level for same comp can come anywhere in the batch
                    if (
                        len(random_locs) > 1
                        and random.choice(random_locs) == i
                        and anomaly_count <= max_anomalies_count
                    ):
                        # ensure that we don't have logs with two error levels adjacent to each other all the time
                        if prev_anomaly_loc is None or (
                            prev_anomaly_loc + 1 == i and v
                        ):
                            lvl = random.choice(ERROR_LEVELS)
                            comp = target_comp
                            anomaly_count += 1
                            prev_anomaly_loc = i

                    # if we haven't added requisite anomalies till end of loop add one.
                    if i == (WINDOW_SIZE - 1) and anomaly_count <= max_anomalies_count:
                        lvl = random.choice(ERROR_LEVELS)
                        comp = target_comp
                        anomaly_count += 1
                        prev_anomaly_loc = i

                current_time += gap
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "single_comp_errors":
                # all logs from same component are of type ERROR
                gap = random.uniform(1.0, 299.0)
                lvl = random.choice(ERROR_LEVELS)
                current_time += gap
                logs.append(
                    {"timestamp": current_time, "component": target_comp, "level": lvl}
                )

        self.last_ts = current_time
        return logs


def create_balanced_triplets(n_triplets=1000):
    log_generator = SyntheticLogGenerator()
    triplets = []

    normal_scenarios = [
        "healthy_time",
        "healthy_level",
        "single_fatal",
        "cross_component_burst",
        "slow_fatals",
        "single_comp_all_good",
    ]

    abnormal_scenarios = ["anomaly", "single_comp_errors"]

    for _ in range(n_triplets):
        a_type = random.choice(normal_scenarios)
        p_type = random.choice(normal_scenarios)
        n_type = random.choice(abnormal_scenarios)

        anchor = log_generator.generate_window(scenario=a_type)
        positive = log_generator.generate_window(scenario=p_type)
        #
        # The Negative is always the actual rule violation
        negative = log_generator.generate_window(scenario=n_type)

        triplets.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "metadata": {"a_type": a_type, "p_type": p_type},
            }
        )
    return triplets


def generate_synthetic_logs(batch_count, validation=False):
    dataset = create_balanced_triplets(batch_count)
    anchors = []
    positives = []
    negatives = []
    for d in dataset:
        anchors.extend(d["anchor"])
        positives.extend(d["positive"])
        negatives.extend(d["negative"])

    if validation:
        file_path_mapping = {
            "anchors": (anchors, ANCHOR_FILE_PATH_VALID),
            "positive": (positives, POSITIVE_FILE_PATH_VALID),
            "negative": (negatives, NEGATIVE_FILE_PATH_VALID),
        }
    else:
        file_path_mapping = {
            "anchors": (anchors, ANCHOR_FILE_PATH_TRAIN),
            "positive": (positives, POSITIVE_FILE_PATH_TRAIN),
            "negative": (negatives, NEGATIVE_FILE_PATH_TRAIN),
        }


    for v in file_path_mapping.values():
        dfp = pd.DataFrame(v[0])
        assert len(dfp) > 0
        dfp = dfp.sort_values("timestamp")
        dfp.to_csv(v[1], index=False)


if __name__ == "__main__":
    generate_synthetic_logs(1000, True)

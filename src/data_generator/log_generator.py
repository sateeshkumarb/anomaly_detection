from typing import List

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
    NEGATIVE_FILE_PATH_TRAIN_0,
    NEGATIVE_FILE_PATH_TRAIN_1,
    NEGATIVE_FILE_PATH_TRAIN_2,
    NEGATIVE_FILE_PATH_TRAIN_3,
    ANCHOR_FILE_PATH_VALID,
    POSITIVE_FILE_PATH_VALID,
    NEGATIVE_FILE_PATH_VALID,
)

ANOMALY_SUB_TYPE_EASY = {
    "front_loaded": 0.35,
    "back_loaded": 0.35,
    "middle": 0.15,
    "triple": 0.15,
    "spread": 0.00,
    "multi_comp_distractor": 0.00,
    "boundary_time": 0.00,
}

ANOMALY_SUB_TYPE_MEDIUM = {
    "front_loaded": 0.25,
    "back_loaded": 0.25,
    "middle": 0.10,
    "triple": 0.10,
    "spread": 0.25,
    "multi_comp_distractor": 0.05,
    "boundary_time": 0.00,
}

ANOMALY_SUB_TYPE_HARD = {
    "front_loaded": 0.15,
    "back_loaded": 0.15,
    "middle": 0.05,
    "triple": 0.05,
    "spread": 0.30,
    "multi_comp_distractor": 0.10,
    "boundary_time": 0.20,
}

ANOMALY_SUB_TYPE_EXTREME = {
    "front_loaded": 0.10,
    "back_loaded": 0.10,
    "middle": 0.03,
    "triple": 0.02,
    "spread": 0.35,
    "multi_comp_distractor": 0.15,
    "boundary_time": 0.25,
}

ANOMALY_SUB_TYPE_FOR_VALIDATION = {
    "front_loaded": 0.05,
    "back_loaded": 0.05,
    "middle": 0.05,
    "triple": 0.05,
    "spread": 0.40,
    "multi_comp_distractor": 0.10,
    "boundary_time": 0.30,
}


class SyntheticLogGenerator:
    def __init__(self, start_time=None):
        if start_time is None:
            self.last_ts = time.time()
        else:
            self.last_ts = start_time

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
                        # field not used for training but only to analyze the training dataset mix
                        # "synthetic_type": scenario,
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
                elif i == pattern_start + 1:
                    lvl = random.choices(ERROR_LEVELS, weights=[60, 40])[0]
                    comp = random.choice(other_comps)  # Different Component
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "slow_fatals":
                # Rule: Two FATALs > 300s apart from SAME component is NOT abnormal
                current_time += gap
                if i == pattern_start:
                    lvl = random.choices(ERROR_LEVELS, weights=[70, 30])[0]
                    comp = target_comp
                if i == pattern_start + 1:
                    lvl = random.choices(ERROR_LEVELS, weights=[60, 40])[0]
                    comp = target_comp
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario == "single_comp_all_good":
                current_time += gap
                lvl = random.choices(NORMAL_LEVELS, weights=[34, 33, 33])[0]
                logs.append(
                    {"timestamp": current_time, "component": comp, "level": lvl}
                )
            elif scenario.startswith("anomaly_"):
                if scenario == "anomaly_easy":
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_EASY.keys()),
                        weights=list(ANOMALY_SUB_TYPE_EASY.values()),
                    )[0]
                elif scenario == "anomaly_medium":
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_MEDIUM.keys()),
                        weights=list(ANOMALY_SUB_TYPE_MEDIUM.values()),
                    )[0]
                elif scenario == "anomaly_hard":
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_HARD.keys()),
                        weights=list(ANOMALY_SUB_TYPE_HARD.values()),
                    )[0]
                elif scenario == "anomaly_extreme":
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_EXTREME.keys()),
                        weights=list(ANOMALY_SUB_TYPE_EXTREME.values()),
                    )[0]
                elif scenario == "anomaly_for_validation":
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_FOR_VALIDATION.keys()),
                        weights=list(ANOMALY_SUB_TYPE_FOR_VALIDATION.values()),
                    )[0]
                else:
                    # default to EASY
                    sub_type = random.choices(
                        list(ANOMALY_SUB_TYPE_EASY.keys()),
                        weights=list(ANOMALY_SUB_TYPE_EASY.values()),
                    )[0]

                return self._generate_anomaly_window(sub_type)

        self.last_ts = current_time
        return logs

    def _generate_anomaly_window(self, sub_type: str) -> List[dict]:
        """
        Each sub_type handles one specific anomaly pattern cleanly.
        Background noise logs are always interspersed to reflect reality.
        """
        logs = [None] * WINDOW_SIZE
        current_time = self.last_ts + self.batch_gap
        target_comp = random.choice(COMPONENTS)
        other_comps = [c for c in COMPONENTS if c != target_comp]

        # --- 1. Decide WHERE the violating events go in the batch ---
        if sub_type == "front_loaded":
            # Violations early in batch
            violation_positions = sorted(random.sample(range(0, 4), 2))

        elif sub_type == "back_loaded":
            # Violations late in batch
            violation_positions = sorted(random.sample(range(6, WINDOW_SIZE), 2))

        elif sub_type == "spread":
            # Violations at opposite ends — hardest for model to detect
            violation_positions = [
                random.randint(0, 2),
                random.randint(WINDOW_SIZE - 3, WINDOW_SIZE - 1),
            ]

        elif sub_type == "middle":
            violation_positions = sorted(random.sample(range(3, 7), 2))

        elif sub_type == "triple":
            # Three violations spread across batch
            violation_positions = sorted(random.sample(range(WINDOW_SIZE), 3))
        else:
            # fallback: random positions
            violation_positions = sorted(random.sample(range(WINDOW_SIZE), 2))

        # --- 2. Decide the TIME DELTA between violations ---
        if sub_type == "boundary_time":
            # Just inside the 300s rule — hardest temporal case
            total_violation_gap = random.uniform(250.0, 299.9)
        else:
            total_violation_gap = random.uniform(10.0, 250.0)

        # --- 3. Build timestamps for the full window ---
        # Distribute total_violation_gap across the gap between
        # first and last violation position
        first_vp = violation_positions[0]
        last_vp = violation_positions[-1]
        n_slots = last_vp - first_vp  # number of inter-log gaps to fill

        timestamps = {}
        t = current_time

        for i in range(WINDOW_SIZE):
            if i < first_vp:
                t += random.uniform(1.0, 20.0)  # background noise before violations
            elif i == first_vp:
                t += random.uniform(1.0, 10.0)
            elif i > first_vp and i <= last_vp:
                # Distribute total_violation_gap evenly across slots
                # with small jitter so it's not mechanical
                slot_gap = (total_violation_gap / n_slots) + random.uniform(-2.0, 2.0)
                slot_gap = max(0.5, slot_gap)  # never negative
                t += slot_gap
            else:
                t += random.uniform(1.0, 20.0)  # background noise after violations
            timestamps[i] = t

        # --- 4. Fill in log entries ---
        for i in range(WINDOW_SIZE):
            if i in violation_positions:
                logs[i] = {
                    "timestamp": timestamps[i],
                    "component": target_comp,
                    "level": random.choice(ERROR_LEVELS),
                }
            else:
                # Background noise — optionally add distractors
                if sub_type == "multi_comp_distractor" and random.random() < 0.3:
                    # Other components also show high severity — harder to isolate signal
                    logs[i] = {
                        "timestamp": timestamps[i],
                        "component": random.choice(other_comps),
                        "level": random.choice(ERROR_LEVELS),
                    }
                else:
                    logs[i] = {
                        "timestamp": timestamps[i],
                        "component": random.choice(COMPONENTS),
                        "level": random.choices(NORMAL_LEVELS, weights=[34, 33, 33])[0],
                    }

        self.last_ts = timestamps[WINDOW_SIZE - 1]
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

    for _ in range(n_triplets):
        a_type = random.choice(normal_scenarios)
        p_type = random.choice(normal_scenarios)

        anchor = log_generator.generate_window(scenario=a_type)
        positive = log_generator.generate_window(scenario=p_type)
        #
        # The Negative is always the actual rule violation
        negative_easy = log_generator.generate_window(scenario="anomaly_easy")
        negative_medium = log_generator.generate_window(scenario="anomaly_medium")
        negative_hard = log_generator.generate_window(scenario="anomaly_hard")
        negative_extreme = log_generator.generate_window(scenario="anomaly_extreme")
        negative_for_validation = log_generator.generate_window(
            scenario="anomaly_for_validation"
        )

        triplets.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative_easy": negative_easy,
                "negative_medium": negative_medium,
                "negative_hard": negative_hard,
                "negative_extreme": negative_extreme,
                "negative_for_validation": negative_for_validation,
            }
        )
    return triplets


def generate_synthetic_logs(batch_count, validation=False):
    dataset = create_balanced_triplets(batch_count)
    anchors = []
    positives = []
    negatives_easy = []
    negatives_medium = []
    negatives_hard = []
    negatives_extreme = []

    for d in dataset:
        anchors.extend(d["anchor"])
        positives.extend(d["positive"])
        negatives_easy.extend(d["negative_easy"])
        negatives_medium.extend(d["negative_medium"])
        negatives_hard.extend(d["negative_hard"])
        negatives_extreme.extend(d["negative_extreme"])

    if validation:
        file_path_mapping = {
            "anchors": (anchors, ANCHOR_FILE_PATH_VALID),
            "positive": (positives, POSITIVE_FILE_PATH_VALID),
            "negative_for_validation": (negatives_extreme, NEGATIVE_FILE_PATH_VALID),
        }
    else:
        file_path_mapping = {
            "anchors": (anchors, ANCHOR_FILE_PATH_TRAIN),
            "positive": (positives, POSITIVE_FILE_PATH_TRAIN),
            "negative_easy": (negatives_easy, NEGATIVE_FILE_PATH_TRAIN_0),
            "negative_medium": (negatives_medium, NEGATIVE_FILE_PATH_TRAIN_1),
            "negative_hard": (negatives_hard, NEGATIVE_FILE_PATH_TRAIN_2),
            "negative_extreme": (negatives_extreme, NEGATIVE_FILE_PATH_TRAIN_3),
        }

    for v in file_path_mapping.values():
        dfp = pd.DataFrame(v[0])
        assert len(dfp) > 0
        dfp = dfp.sort_values("timestamp")
        dfp.to_csv(v[1], index=False)


if __name__ == "__main__":
    generate_synthetic_logs(1000, True)

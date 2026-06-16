# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function
"""

import re
import string



def batch_reward_function(data_source, solution_str, ground_truth, extra_info=None):

    try:
        v = list(map(int, re.findall(r"<answer>(\d+)</answer>", solution_str)))
        if len(set(v)) != len(ground_truth):
            return 0
            
        labels = list(string.ascii_uppercase[:len(v)])
        sorted_labels = [label for _, label in sorted(zip(v, labels), reverse=True)]
        result = ''.join(sorted_labels)

        batch_reward = 0

        if ground_truth == "ABC":
            if result == "ABC": batch_reward = 1
            elif result in ["ACB", "BAC"]: batch_reward = 2/3
            elif result in ["BCA", "CAB"]: batch_reward = 1/3
            else: batch_reward = 0

        if ground_truth == "ACB":
            if result == "ACB": batch_reward = 1
            elif result in ["ABC", "CAB"]: batch_reward = 2/3
            elif result in ["CBA", "BAC"]: batch_reward = 1/3
            else: batch_reward = 0

        if ground_truth == "BAC":
            if result == "BAC": batch_reward = 1
            elif result in ["BCA", "ABC"]: batch_reward = 2/3
            elif result in ["ACB", "CBA"]: batch_reward = 1/3
            else: batch_reward = 0

        if ground_truth == "BCA":
            if result == "BCA": batch_reward = 1
            elif result in ["BAC", "CBA"]: batch_reward = 2/3
            elif result in ["ABC", "CAB"]: batch_reward = 1/3
            else: batch_reward = 0

        if ground_truth == "CAB":
            if result == "CAB": batch_reward = 1
            elif result in ["CBA", "ACB"]: batch_reward = 2/3
            elif result in ["BCA", "ABC"]: batch_reward = 1/3
            else: batch_reward = 0

        if ground_truth == "CBA":
            if result == "CBA": batch_reward = 1
            elif result in ["CAB", "BCA"]: batch_reward = 2/3
            elif result in ["ACB", "BAC"]: batch_reward = 1/3
            else: batch_reward = 0

        return batch_reward

    except Exception:
        print(ground_truth, solution_str)
        return 0



def pair_reward_function(solution_str, ground_truth):
    try:
        score = list(map(int, re.findall(r"<answer>(\d+)</answer>", solution_str)))
        
        if len(score) != 2:
            return 0
        
        if score[0] > score[1]: pred = "A"
        elif score[0] < score[1]: pred = "B"
        else: pred = "C"
        
        if ground_truth == pred:
            return 1
        else:
            return 0
        
    except Exception:
        print(ground_truth, solution_str)
        return 0

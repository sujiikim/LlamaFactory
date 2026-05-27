# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen
from .response_mask import mask_thought_in_target_ids_action, mask_thought_in_target_ids_toolcall


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):

    VALID_LEARN_TYPES = {"learn_action", "learn_param", "learn_full"}

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        learn_type: Optional[str] = None,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_label_ids = list(chosen_ids)
        rejected_label_ids = list(rejected_ids)
        if self.data_args.label_dynamic:
            if learn_type == "learn_action":
                chosen_label_ids = mask_thought_in_target_ids_action(chosen_label_ids, self.tokenizer)
                rejected_label_ids = mask_thought_in_target_ids_action(rejected_label_ids, self.tokenizer)
            elif learn_type == "learn_param":
                chosen_label_ids = mask_thought_in_target_ids_toolcall(chosen_label_ids, self.tokenizer)
                rejected_label_ids = mask_thought_in_target_ids_toolcall(rejected_label_ids, self.tokenizer)
            # learn_full: no masking, keep all labels

        # input_ids must remain valid token ids; IGNORE_INDEX belongs only in labels (same as SFT).
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_label_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_label_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            learn_type = None
            if self.data_args.label_dynamic:
                learn_type = examples["_learn_type"][i]
                if learn_type is None:
                    logger.warning_rank0("Skipped example with missing learn_type value.")
                    continue

                if learn_type == "":
                    raise ValueError("Found empty string for `learn_type`. Each example must have a valid learn_type.")

                if learn_type not in self.VALID_LEARN_TYPES:
                    raise ValueError(
                        f"Invalid learn_type `{learn_type}`. Must be one of {self.VALID_LEARN_TYPES}."
                    )

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                learn_type=learn_type,
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_ids = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_input_ids"]))
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_ids = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_input_ids"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            f"chosen_inputs:\n{self.tokenizer.decode(valid_chosen_ids, skip_special_tokens=False)}"
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            f"rejected_inputs:\n{self.tokenizer.decode(valid_rejected_ids, skip_special_tokens=False)}"
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
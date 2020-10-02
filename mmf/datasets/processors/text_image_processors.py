# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from mmf.utils.process_text_image import text_token_overlap_with_bbox
from mmf.utils.text import is_punctuation
from transformers.tokenization_auto import AutoTokenizer


class TextImageProcessor(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        text_tokenizer_config = config.text_processor.tokenizer_config
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_config.type, **text_tokenizer_config.params
        )

    def sample_traces(self, utterances, utterance_times, traces):
        tokens, index_tokens, token_weights, token_traces = [], [], [], []
        for idx, utterance in enumerate(utterances):
            if self.config.remove_punctuations:
                utterance = self.remove_punctuation(utterance)

            tokens_from_utterance = self.text_tokenizer.tokenize(utterance)
            indexed_tokens_from_utterance = self.text_tokenizer.convert_tokens_to_ids(
                tokens_from_utterance
            )
            (
                token_times,
                token_weights_from_utterance,
            ) = self.determine_token_weight_and_times(
                tokens_from_utterance,
                utterance_times[idx][0].item(),
                utterance_times[idx][1].item(),
            )
            token_traces_from_utterance = self.traces_from_token(
                token_times, traces, self.config.trace_num_samples
            )

            tokens += tokens_from_utterance
            index_tokens += indexed_tokens_from_utterance
            token_weights += token_weights_from_utterance
            token_traces += token_traces_from_utterance
        return (
            tokens,
            torch.IntTensor(index_tokens),
            torch.FloatTensor(token_weights),
            torch.FloatTensor(token_traces),
        )

    def determine_token_weight_and_times(self, tokens, start, end):
        return (
            torch.FloatTensor([start, end]).repeat(len(tokens), 1),
            torch.FloatTensor([1 / len(tokens)]).repeat(len(tokens)),
        )

    def traces_from_token(self, token_times, traces, num_samples):
        assert num_samples > 2, "Must sample at least the start and end traces."
        sampled_traces = []
        for start, end in token_times:
            assert start <= end, "End time must be greater than or equal to start time."

            start_trace = self.linearpolate(traces, start)
            end_trace = self.linearpolate(traces, end)

            # find intermediate interpolations
            trace_per_utter = [start_trace]
            interval = (end_trace[2] - start_trace[2]) / (num_samples - 1)
            for i in range(num_samples - 2):
                trace_per_utter.append(
                    self.linearpolate(traces, start + (i + 1) * interval)
                )

            trace_per_utter.append(end_trace)
            sampled_traces.append(trace_per_utter)

        return sampled_traces

    def linearpolate(self, traces, time_of_interest):
        prev_trace = None
        # interpolate
        for trace in traces:
            if prev_trace is None:
                prev_trace = trace
                continue
            if time_of_interest >= prev_trace[2] and time_of_interest <= trace[2]:
                return self.interpolate(prev_trace, trace, time_of_interest)
            prev_trace = trace

        # extrapolate
        if len(traces) > 1 and time_of_interest < traces[0][2]:
            return self.extrapolate(
                traces[0], traces[1], time_of_interest, is_left=True
            )
        elif len(traces) > 1 and time_of_interest >= traces[-1][2]:
            return self.extrapolate(
                traces[-2], traces[-1], time_of_interest, is_left=False
            )
        else:
            assert 0, "We should be able to either extrapolate or interpolate"

    def interpolate(self, coord1, coord2, time_of_interest):
        coords = np.array([coord1.numpy(), coord2.numpy()])
        tx = np.interp(time_of_interest, coords[:, 2], coords[:, 0])
        ty = np.interp(time_of_interest, coords[:, 2], coords[:, 1])

        return [max(0, min(tx, 1)), max(0, min(ty, 1)), time_of_interest]

    def extrapolate(self, coord1, coord2, time_of_interest, is_left=True):
        slope_x = (coord2[0] - coord1[0]) / (coord2[2] - coord1[2])
        slope_y = (coord2[1] - coord1[1]) / (coord2[2] - coord1[2])
        if is_left:
            anchor = coord1
        else:
            anchor = coord2
        tx = anchor[0] + slope_x * (time_of_interest - anchor[2])
        ty = anchor[1] + slope_y * (time_of_interest - anchor[2])
        return [max(0, min(tx, 1)), max(0, min(ty, 1)), time_of_interest]

    def remove_punctuation(self, word):
        chars = ""
        for char in word:
            if not is_punctuation(char):
                chars += char
        return chars


@registry.register_processor("text_token_overlap_with_image_bbox")
class TextTokenOverlapWithImageBboxProcessor(TextImageProcessor):
    def __call__(self, item):
        utterances = item["utterances"]
        utterance_times = item["utterance_times"]
        traces = item["traces"]
        image_info = item["image_info"]
        tokens, index_tokens, token_weights, token_traces = self.sample_traces(
            utterances, utterance_times, traces
        )
        percent_overlaps = text_token_overlap_with_bbox(
            token_traces, image_info["bbox"], self.config.trace_num_samples,
        )
        return {
            "tokens": tokens,
            "index_tokens": index_tokens,
            "token_weights": token_weights,
            "percent_overlaps": percent_overlaps,
        }


@registry.register_processor("text_token_closeness_with_image_bbox")
class TextTokenClosenessWithImageBboxProcessor(TextImageProcessor):
    pass

# Copyright (c) Facebook, Inc. and its affiliates.

import torch


def in_bbox(loc, bbox):
    return (
        loc[0] > bbox[0] and loc[0] < bbox[2] and loc[1] > bbox[1] and loc[1] < bbox[3]
    )


def text_token_overlap_with_bbox(traces, bboxes, num_samples):
    percent_overlaps = []
    for token_trace in traces:
        overlaps = []
        for bbox in bboxes:
            overlap = 0
            for trace in token_trace:
                if in_bbox(trace, bbox):
                    overlap += 1
            overlaps.append(overlap / num_samples)
        percent_overlaps.append(overlaps)
    return torch.FloatTensor(percent_overlaps)

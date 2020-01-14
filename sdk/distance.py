#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/1/14 14:00
# @Author   : chenkai
# @File     : distance.py

"""
import numpy as np


def calculate_distance(feature_1, feature_2) -> float:
    # return the distance between two feature
    similarity = np.dot(feature_1, feature_2)
    # TODO: replace the statement below with actual logic
    distance = 1 - similarity
    return distance

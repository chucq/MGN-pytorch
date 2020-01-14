#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/1/14 14:00
# @Author   : chenkai
# @File     : inference.py

"""
from sdk import distance


def initialize():
    # TODO: initialize the model
    return None


def infer(model, image_path):
    # TODO: given the model and image_path, get the feature
    return []


if __name__ == "__main__":
    mgn_model = initialize()
    image_1_path = ''
    image_2_path = ''
    feature_1 = infer(mgn_model, image_1_path)
    feature_2 = infer(mgn_model, image_2_path)
    dis = distance.calculate_distance(feature_1, feature_2)
    print(f'the distance of feature_1 and feature_2 is {dis}')


# -*- coding: utf-8 -*-

import math

def print_center(string: str, width: int = 80):
    string_len = len(string)
    padding = math.floor((width - string_len)/2)
    new_string = " "*padding + string + " "*padding
    print(new_string)
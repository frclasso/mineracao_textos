#!/usr/bin/env python3
# -*-encoding = utf-8 -*-

import re


pattern = re.compile(r'[A-Z][^A-Z]\w+')

with open('bras_cubas.txt', 'r', encoding='utf-8') as f:
    contents = f.read()
    matches = pattern.findall(contents)
    for match in matches:
        print(match)

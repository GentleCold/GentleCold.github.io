---
title: CMU15721-Spring2024课程笔记
category: [笔记]
date: 2024-11-19 15:11
tags: [笔记, CMU, 数据库]
---

some papers that worth to read:

// todo

## Overview

- data cubes -> data warehouses -> shared-disk -> lakehouse
- ETL tool
- push query to data / pull data to query
- shared-nothing / shared-disk

## Data Formats

- storage model:
  - n-ary: store all the attributes for a single tuple contiguously
  - decomposition: store a single attribute for all tuples contiguously
  - partition attributes across(PAX): hybrid, bertically partion attributes
    - using column chunks
- open-source: parquet / orc / arrow
- encoding:
  - dictionary compression for column
  - zstd for block compression
  - **zone maps** / bloom filters for filters
- nested data in columns:
  - shredding
  - length + presence

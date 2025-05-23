# Introduction

Oceanographic time series like any sequence of observational measures based on physical sensors is often missing data or has data that has failed a quality control check. Before this data can be used in subsequent data analysis workflow, it may be necessary to gap fill ("impute") the missing data. While for short gaps, simpler univariate methods are sufficient but when the gap is relatively long, machine learn methods are attractive option for estimating the missing data.

This series of notebooks documents the analysis and comparion of several machine learning method for data imputation of time series. The time series chosen for analysis is the Center for Marine Applied Research (CMAR) Water Quality datasets hosted at CIOOS Atlantic.

This project is an outcome of the Building Bridges Project (WPD2.2 Data Interpolation) from Canada's Oceans Supercluster.



#!/usr/bin/env python3
"""Data store object with data storage and basic functionality

Contains support for loading vehicle csv data into Pandas DataFrames using
functions from the HighD dataset source code, and adds support functions for
extracting data from vehicle id.

  Typical usage example:
  my_data_store = DataStore('./data')
  my_data = my_data_store.read_data_file('01_tracks.csv')
  vehicle_1_data = my_data.store.get_data_by_id(1)

"""

from sklearn.model_selection import train_test_split
import pandas as pd

from utils.read_csv import read_track_csv


class DataStore:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = {}

    # Returns vehicle data collated by vehicle id as a list of dicts
    def read_data_file(self, file_path: str):
        args = {'input_path' : self.data_path + file_path}
        tmp_data = read_track_csv(args)
        self.data = pd.DataFrame(tmp_data).set_index('id')
        #for vehicle in tmp_data:
        #    vehicle_id = vehicle['id']
        #    del vehicle['id']
        #    self.data[vehicle_id] = vehicle
        return self.data

    def get_data_by_id(self, vehicle_id: int):
        return self.data.loc[vehicle_id]

    def get_train_valid_split(self, proportion: float):
        return


if __name__ == '__main__':
    my_data_store = DataStore('~/Dropbox/exjobb/highd-dataset-v1.0/highD-dataset/Python/data')
    my_data_store.read_data_file('01_tracks.csv')
    print(my_data_store.get_data_by_id([1,5,7]).keys())


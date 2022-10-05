#!/usr/bin/env ruby

#####
#
# Usage:
#
# ./create_config PATH RECORDING
#
#####

files = ARGV

if files.length() < 2
  puts 'Insufficient number of arguments provided. Exiting...'
  exit 1
end

data_path = File.expand_path(ARGV[0])
recording = ARGV[1]

if recording.length == 1
  recording.prepend('0')
end

require 'yaml'

data_store = {
  'recordingMeta_file' => data_path + '/' + recording + '_recordingMeta.csv',
  'tracksMeta_file' => data_path + '/' + recording + '_tracksMeta.csv',
  'data_file' => data_path + '/' + recording + '_tracks.csv',
  'highway_image' => data_path + '/' + recording + '_highway.png'
}

File.open('data_store.yml', 'w') do |file|
  file.write data_store.to_yaml
end

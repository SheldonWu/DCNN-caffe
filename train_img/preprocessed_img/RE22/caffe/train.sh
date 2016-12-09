#!/usr/bin/env sh

set -e
TOOLS=/home/cmcc/caffe/build/tools
GLOG_logtostderr=0 GLOG_log_dir=../LOG \
$TOOLS/caffe train \
	 --solver=./solver.prototxt




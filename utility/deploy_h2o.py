#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This script deploys h2o in the server using deploy_h2o.sh

	This script deploys h2o in the server using deploy_h2o.sh. 
	If H2O is not running in the server, the script will start H2O.
	If H2O is already running in the server, the script will 
	connect to the existing session.
""" 

import sys
import os 
import time
import h2o 

def start():
	"""Starts h2o session if it is not running in the server.
	   If it already is running, then connect to it.
	"""
	try:
		print("[INFO] Checking if H2O is already running in the server.")
		h2o.connect()
	except:
		print("[INFO] H2O is not running. Starting H2O now.")
		os.system('../bin/deploy_h2o.sh')
		time.sleep(3)
		h2o.connect()
	return h2o


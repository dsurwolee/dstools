 #!/bin/sh
#
# Script Name: deploy_h2o.sh
#
# Author: Dan Lee
# Date: 12/15/18
# 
# Description: H2O deployment script

# Required configurations
JARPATH=/simility/lib/h2o.jar
IP_ADDRESS=127.0.0.1
PORT=54321
MEMORY=15
NTHREADS=-1

# Start h2o shell script
#function start_h2o ()
#{
#	nohup java -Xmx${MEMORY}g -jar ${JARPATH} -port ${PORT} -nthreads ${NTHREADS} -flatfile flatfile.txt &
#}

function start_h2o ()
{
	/usr/bin/java -ea -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -jar /home/simility/anaconda3/lib/python3.6/site-packages/h2o/backend/bin/h2o.jar -ip 127.0.0.1 -baseport54321 -ice_root /tmp/tmpm039c1xe -name H2O_from_python_simility_y7z4fu -log_level INFO
}

# Main function to execute
function main ()
{
	PROCESS=`ps aux | grep h2o.jar`
	# echo $PROCESS
	if [[ $PROCESS == *$JARPATH* ]]
	then
  		echo "H2O is already running in the server."
  	else
  		echo "Launching H2O..."
  		start_h2o
	fi
}

# Execute main 
main

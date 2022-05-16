export HADOOP_EXE='/usr/bin/hadoop'

alias hfs="$HADOOP_EXE fs"
alias spark-submit='python setup.py build; pip install .; PYSPARK_PYTHON=$(which python) /bin/spark-submit'
total=0;
for i in `seq 1 $2`;
do 
	second=`./test_auto_vectorize -t $1 | awk 'BEGIN{FS="sec"} NR==3{print $1}'`;
	echo $second;
	total=`echo "$total+$second" | bc -l`;
done
median=`echo "scale=5; $total/$2" | bc -l`;
echo "Running test$1()...";
echo "Median elapsed execution time of the loop in test$1():";
echo "$median sec (N: 1024, I: 20000000)";

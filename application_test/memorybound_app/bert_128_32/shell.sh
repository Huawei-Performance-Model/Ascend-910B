python ./get_fullgraph.py
gcc -c get_subgraph.cpp
gcc get_subgraph.o -o get_subgraph -lstdc++
./get_subgraph
python ./application_to_extime_forbert.py



#最好是搞一下clean操作
rm get_subgraph
rm get_subgraph.o

#python3 /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py  export summary -dir ./



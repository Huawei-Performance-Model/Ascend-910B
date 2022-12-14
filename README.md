  We can run **end_2_end.sh** that corresponds to **Fig. 10** and **memorybound.sh** that corresponds to **Fig. 11** in our paper.<br>
  For example:<br>
  To get the result in **Fig. 10**, you should run:<br>
  `bash end_2_end.sh`<br>
  To get the result in **Fig. 11**, you should run:<br>
  `bash memorybound.sh`<br>
  
  ***Attention:***<br>
  Before running these two shell, we need to unzip the following files first **for the limited storage space in github**:<br>
  file 1:<br>
  https://github.com/Huawei-Performance-Model/Ascend-910B/blob/main/application_test/final_test/vit64/graph.rar<br>
  file 2:<br>
  https://github.com/Huawei-Performance-Model/Ascend-910B/blob/main/application_test/final_test/vit128/graph.rar<br>
  file 3:<br>
  https://github.com/Huawei-Performance-Model/Ascend-910B/blob/main/application_test/final_test/vit256/graph.rar<br>
  file 4:<br>
  https://github.com/Huawei-Performance-Model/Ascend-910B/blob/main/application_test/memorybound_app/resnet50_64_224_224/runtime.rar<br>
  
  ***Besides：***<br>
  We need use runtime.db files in Releases：
  https://github.com/Huawei-Performance-Model/Ascend-910B/releases<br>
  Here we can find：<br>
  file 5:<br>
  resnet50_16_runtime.7z<br>
  file 6:<br>
  resnet50_32_runtime.rar<br>

  We need to unzip both of these two files, and move file 5 to:
  `application_test/memorybound_app/resnet50_16_224_224`<br>
  and move file 6 to:
  `application_test/memorybound_app/resnet50_32_224_224`<br>

	O!W?Yq?@O!W?Yq?@!O!W?Yq?@	?Sօ$l???Sօ$l??!?Sօ$l??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLO!W?Yq?@?z0)? @17?h?=?}@A?_vO??Id\qqԋi@Y?}8H????rEagerKernelExecute 0*	??QX??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??R%L@!??F*u?X@)??R%L@1??F*u?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??7?{ֹ?!?rV????)?k$	±?1??Oт??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchs????(??!???ά??)s????(??1???ά??:Preprocessing2F
Iterator::Model?E??ꎽ?!c?S@?9??)???=zÍ?1"
OGh??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??Q?L@! ?_-??X@)?#????n?1swGƀ{?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?29.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Sօ$l??I?_?@Q???2?7Q@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?z0)? @?z0)? @!?z0)? @      ??!       "	7?h?=?}@7?h?=?}@!7?h?=?}@*      ??!       2	?_vO???_vO??!?_vO??:	d\qqԋi@d\qqԋi@!d\qqԋi@B      ??!       J	?}8H?????}8H????!?}8H????R      ??!       Z	?}8H?????}8H????!?}8H????b      ??!       JGPUY?Sօ$l??b q?_?@y???2?7Q@
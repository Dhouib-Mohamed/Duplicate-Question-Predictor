©¹/
·
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeķout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint’’’’’’’’’
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ļ,
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0

Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

:@*
dtype0

Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

:@*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:@*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:@*
dtype0

Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	@*
dtype0

Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	@*
dtype0

Adam/v/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/lstm/lstm_cell/bias

.Adam/v/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/bias*
_output_shapes	
:*
dtype0

Adam/m/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/lstm/lstm_cell/bias

.Adam/m/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/bias*
_output_shapes	
:*
dtype0
Ŗ
&Adam/v/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&Adam/v/lstm/lstm_cell/recurrent_kernel
£
:Adam/v/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/v/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0
Ŗ
&Adam/m/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&Adam/m/lstm/lstm_cell/recurrent_kernel
£
:Adam/m/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/m/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

Adam/v/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/v/lstm/lstm_cell/kernel

0Adam/v/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/kernel*
_output_shapes
:	*
dtype0

Adam/m/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/m/lstm/lstm_cell/kernel

0Adam/m/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/kernel*
_output_shapes
:	*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:*
dtype0

lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!lstm/lstm_cell/recurrent_kernel

3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
z
serving_default_input_2Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
ß
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2lstm/lstm_cell/kernellstm/lstm_cell/biaslstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_269185

NoOpNoOp
ĶO
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*O
valuežNBūN BōN
Ŗ
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer_with_weights-2
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
Į
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec*
„
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator* 
„
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator* 

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
¦
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias*
„
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator* 
¦
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias*
5
U0
V1
W2
D3
E4
S5
T6*
5
U0
V1
W2
D3
E4
S5
T6*
* 
°
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
]trace_0
^trace_1
_trace_2
`trace_3* 
6
atrace_0
btrace_1
ctrace_2
dtrace_3* 
* 

e
_variables
f_iterations
g_learning_rate
h_index_dict
i
_momentums
j_velocities
k_update_step_xla*

lserving_default* 
* 
* 
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

rtrace_0* 

strace_0* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

ytrace_0* 

ztrace_0* 

U0
V1
W2*

U0
V1
W2*
* 
 

{states
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
ė
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

Ukernel
Vrecurrent_kernel
Wbias*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

trace_0
 trace_1* 

”trace_0
¢trace_1* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
„metrics
 ¦layer_regularization_losses
§layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Øtrace_0* 

©trace_0* 

D0
E1*

D0
E1*
* 

Ŗnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

Ætrace_0* 

°trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 “layer_regularization_losses
µlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

¶trace_0
·trace_1* 

øtrace_0
¹trace_1* 
* 

S0
T1*

S0
T1*
* 

ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

ætrace_0* 

Ątrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

Į0
Ā1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

f0
Ć1
Ä2
Å3
Ę4
Ē5
Č6
É7
Ź8
Ė9
Ģ10
Ķ11
Ī12
Ļ13
Š14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
Ć0
Å1
Ē2
É3
Ė4
Ķ5
Ļ6*
<
Ä0
Ę1
Č2
Ź3
Ģ4
Ī5
Š6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

U0
V1
W2*

U0
V1
W2*
* 

Ńnon_trainable_variables
Ņlayers
Ómetrics
 Ōlayer_regularization_losses
Õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ötrace_0
×trace_1* 

Ųtrace_0
Łtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ś	variables
Ū	keras_api

Ütotal

Żcount*
M
Ž	variables
ß	keras_api

ątotal

įcount
ā
_fn_kwargs*
ga
VARIABLE_VALUEAdam/m/lstm/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/lstm/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/lstm/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/lstm/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ü0
Ż1*

Ś	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ą0
į1*

Ž	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
į
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcountConst*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_272442
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcount*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_272533µÖ+


d
E__inference_dropout_2_layer_call_and_return_conditional_losses_268610

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ń
c
*__inference_dropout_2_layer_call_fn_271952

inputs
identity¢StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ę
s
G__inference_concatenate_layer_call_and_return_conditional_losses_271927
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’:’’’’’’’’’:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0

D
(__inference_dropout_layer_call_fn_271870

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268896a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ć
while_cond_268332
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_268332___redundant_placeholder04
0while_while_cond_268332___redundant_placeholder14
0while_while_cond_268332___redundant_placeholder24
0while_while_cond_268332___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter


b
C__inference_dropout_layer_call_and_return_conditional_losses_268556

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
£
F
*__inference_dropout_1_layer_call_fn_271897

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268902a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Å+
“
A__inference_model_layer_call_and_return_conditional_losses_268630
input_1
input_2
lstm_268533:	
lstm_268535:	
lstm_268537:

dense_268593:	@
dense_268595:@ 
dense_1_268624:@
dense_1_268626:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢lstm/StatefulPartitionedCall_1æ
reshape_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143»
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_268158
lstm/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_268533lstm_268535lstm_268537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268532
lstm/StatefulPartitionedCall_1StatefulPartitionedCall reshape/PartitionedCall:output:0lstm_268533lstm_268535lstm_268537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268532č
dropout/StatefulPartitionedCallStatefulPartitionedCall'lstm/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268556
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268570
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_268579
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_268593dense_268595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_268592
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268610
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_268624dense_1_268626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_268623w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’²
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
lstm/StatefulPartitionedCall_1lstm/StatefulPartitionedCall_12<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1


±
&__inference_model_layer_call_fn_269225
inputs_0
inputs_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	@
	unknown_3:@
	unknown_4:@
	unknown_5:
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_269010o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0
ćD
§
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267982

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpS
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::ķĻT
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’U
ones_like_1/ShapeShapestates*
T0*
_output_shapes
::ķĻV
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:’’’’’’’’’S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’]
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’]
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’]
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ķ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­
ó

lstm_while_1_body_270390*
&lstm_while_1_lstm_while_1_loop_counter0
,lstm_while_1_lstm_while_1_maximum_iterations
lstm_while_1_placeholder
lstm_while_1_placeholder_1
lstm_while_1_placeholder_2
lstm_while_1_placeholder_3'
#lstm_while_1_lstm_strided_slice_5_0e
alstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0I
6lstm_while_1_lstm_cell_split_readvariableop_resource_0:	G
8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0:	D
0lstm_while_1_lstm_cell_readvariableop_resource_0:

lstm_while_1_identity
lstm_while_1_identity_1
lstm_while_1_identity_2
lstm_while_1_identity_3
lstm_while_1_identity_4
lstm_while_1_identity_5%
!lstm_while_1_lstm_strided_slice_5c
_lstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensorG
4lstm_while_1_lstm_cell_split_readvariableop_resource:	E
6lstm_while_1_lstm_cell_split_1_readvariableop_resource:	B
.lstm_while_1_lstm_cell_readvariableop_resource:
¢%lstm/while_1/lstm_cell/ReadVariableOp¢'lstm/while_1/lstm_cell/ReadVariableOp_1¢'lstm/while_1/lstm_cell/ReadVariableOp_2¢'lstm/while_1/lstm_cell/ReadVariableOp_3¢+lstm/while_1/lstm_cell/split/ReadVariableOp¢-lstm/while_1/lstm_cell/split_1/ReadVariableOp
>lstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   É
0lstm/while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0lstm_while_1_placeholderGlstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
&lstm/while_1/lstm_cell/ones_like/ShapeShape7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻk
&lstm/while_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
 lstm/while_1/lstm_cell/ones_likeFill/lstm/while_1/lstm_cell/ones_like/Shape:output:0/lstm/while_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
(lstm/while_1/lstm_cell/ones_like_1/ShapeShapelstm_while_1_placeholder_2*
T0*
_output_shapes
::ķĻm
(lstm/while_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ć
"lstm/while_1/lstm_cell/ones_like_1Fill1lstm/while_1/lstm_cell/ones_like_1/Shape:output:01lstm/while_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’·
lstm/while_1/lstm_cell/mulMul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¹
lstm/while_1/lstm_cell/mul_1Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¹
lstm/while_1/lstm_cell/mul_2Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¹
lstm/while_1/lstm_cell/mul_3Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’h
&lstm/while_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :£
+lstm/while_1/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_while_1_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ē
lstm/while_1/lstm_cell/splitSplit/lstm/while_1/lstm_cell/split/split_dim:output:03lstm/while_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split”
lstm/while_1/lstm_cell/MatMulMatMullstm/while_1/lstm_cell/mul:z:0%lstm/while_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_1MatMul lstm/while_1/lstm_cell/mul_1:z:0%lstm/while_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_2MatMul lstm/while_1/lstm_cell/mul_2:z:0%lstm/while_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_3MatMul lstm/while_1/lstm_cell/mul_3:z:0%lstm/while_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’j
(lstm/while_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : £
-lstm/while_1/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ż
lstm/while_1/lstm_cell/split_1Split1lstm/while_1/lstm_cell/split_1/split_dim:output:05lstm/while_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split®
lstm/while_1/lstm_cell/BiasAddBiasAdd'lstm/while_1/lstm_cell/MatMul:product:0'lstm/while_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_1BiasAdd)lstm/while_1/lstm_cell/MatMul_1:product:0'lstm/while_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_2BiasAdd)lstm/while_1/lstm_cell/MatMul_2:product:0'lstm/while_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_3BiasAdd)lstm/while_1/lstm_cell/MatMul_3:product:0'lstm/while_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_4Mullstm_while_1_placeholder_2+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_5Mullstm_while_1_placeholder_2+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_6Mullstm_while_1_placeholder_2+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_7Mullstm_while_1_placeholder_2+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while_1/lstm_cell/ReadVariableOpReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while_1/lstm_cell/strided_sliceStridedSlice-lstm/while_1/lstm_cell/ReadVariableOp:value:03lstm/while_1/lstm_cell/strided_slice/stack:output:05lstm/while_1/lstm_cell/strided_slice/stack_1:output:05lstm/while_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask­
lstm/while_1/lstm_cell/MatMul_4MatMul lstm/while_1/lstm_cell/mul_4:z:0-lstm/while_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ŗ
lstm/while_1/lstm_cell/addAddV2'lstm/while_1/lstm_cell/BiasAdd:output:0)lstm/while_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while_1/lstm_cell/SigmoidSigmoidlstm/while_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_1StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_1:value:05lstm/while_1/lstm_cell/strided_slice_1/stack:output:07lstm/while_1/lstm_cell/strided_slice_1/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_5MatMul lstm/while_1/lstm_cell/mul_5:z:0/lstm/while_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_1AddV2)lstm/while_1/lstm_cell/BiasAdd_1:output:0)lstm/while_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’
 lstm/while_1/lstm_cell/Sigmoid_1Sigmoid lstm/while_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_8Mul$lstm/while_1/lstm_cell/Sigmoid_1:y:0lstm_while_1_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
.lstm/while_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_2StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_2:value:05lstm/while_1/lstm_cell/strided_slice_2/stack:output:07lstm/while_1/lstm_cell/strided_slice_2/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_6MatMul lstm/while_1/lstm_cell/mul_6:z:0/lstm/while_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_2AddV2)lstm/while_1/lstm_cell/BiasAdd_2:output:0)lstm/while_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’x
lstm/while_1/lstm_cell/TanhTanh lstm/while_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_9Mul"lstm/while_1/lstm_cell/Sigmoid:y:0lstm/while_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/add_3AddV2 lstm/while_1/lstm_cell/mul_8:z:0 lstm/while_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
.lstm/while_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm/while_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_3StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_3:value:05lstm/while_1/lstm_cell/strided_slice_3/stack:output:07lstm/while_1/lstm_cell/strided_slice_3/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_7MatMul lstm/while_1/lstm_cell/mul_7:z:0/lstm/while_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_4AddV2)lstm/while_1/lstm_cell/BiasAdd_3:output:0)lstm/while_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’
 lstm/while_1/lstm_cell/Sigmoid_2Sigmoid lstm/while_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’z
lstm/while_1/lstm_cell/Tanh_1Tanh lstm/while_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while_1/lstm_cell/mul_10Mul$lstm/while_1/lstm_cell/Sigmoid_2:y:0!lstm/while_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’y
7lstm/while_1/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm/while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_1_placeholder_1@lstm/while_1/TensorArrayV2Write/TensorListSetItem/index:output:0!lstm/while_1/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅT
lstm/while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm/while_1/addAddV2lstm_while_1_placeholderlstm/while_1/add/y:output:0*
T0*
_output_shapes
: V
lstm/while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm/while_1/add_1AddV2&lstm_while_1_lstm_while_1_loop_counterlstm/while_1/add_1/y:output:0*
T0*
_output_shapes
: n
lstm/while_1/IdentityIdentitylstm/while_1/add_1:z:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_1Identity,lstm_while_1_lstm_while_1_maximum_iterations^lstm/while_1/NoOp*
T0*
_output_shapes
: n
lstm/while_1/Identity_2Identitylstm/while_1/add:z:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_3IdentityAlstm/while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_4Identity!lstm/while_1/lstm_cell/mul_10:z:0^lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/Identity_5Identity lstm/while_1/lstm_cell/add_3:z:0^lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’×
lstm/while_1/NoOpNoOp&^lstm/while_1/lstm_cell/ReadVariableOp(^lstm/while_1/lstm_cell/ReadVariableOp_1(^lstm/while_1/lstm_cell/ReadVariableOp_2(^lstm/while_1/lstm_cell/ReadVariableOp_3,^lstm/while_1/lstm_cell/split/ReadVariableOp.^lstm/while_1/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_while_1_identity_1 lstm/while_1/Identity_1:output:0";
lstm_while_1_identity_2 lstm/while_1/Identity_2:output:0";
lstm_while_1_identity_3 lstm/while_1/Identity_3:output:0";
lstm_while_1_identity_4 lstm/while_1/Identity_4:output:0";
lstm_while_1_identity_5 lstm/while_1/Identity_5:output:0"7
lstm_while_1_identitylstm/while_1/Identity:output:0"b
.lstm_while_1_lstm_cell_readvariableop_resource0lstm_while_1_lstm_cell_readvariableop_resource_0"r
6lstm_while_1_lstm_cell_split_1_readvariableop_resource8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_while_1_lstm_cell_split_readvariableop_resource6lstm_while_1_lstm_cell_split_readvariableop_resource_0"H
!lstm_while_1_lstm_strided_slice_5#lstm_while_1_lstm_strided_slice_5_0"Ä
_lstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensoralstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2R
'lstm/while_1/lstm_cell/ReadVariableOp_1'lstm/while_1/lstm_cell/ReadVariableOp_12R
'lstm/while_1/lstm_cell/ReadVariableOp_2'lstm/while_1/lstm_cell/ReadVariableOp_22R
'lstm/while_1/lstm_cell/ReadVariableOp_3'lstm/while_1/lstm_cell/ReadVariableOp_32N
%lstm/while_1/lstm_cell/ReadVariableOp%lstm/while_1/lstm_cell/ReadVariableOp2Z
+lstm/while_1/lstm_cell/split/ReadVariableOp+lstm/while_1/lstm_cell/split/ReadVariableOp2^
-lstm/while_1/lstm_cell/split_1/ReadVariableOp-lstm/while_1/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :WS

_output_shapes
: 
9
_user_specified_name!lstm/while_1/maximum_iterations:Q M

_output_shapes
: 
3
_user_specified_namelstm/while_1/loop_counter
ēĒ
ź
__inference__traced_save_272442
file_prefix6
#read_disablecopyonread_dense_kernel:	@1
#read_1_disablecopyonread_dense_bias:@9
'read_2_disablecopyonread_dense_1_kernel:@3
%read_3_disablecopyonread_dense_1_bias:A
.read_4_disablecopyonread_lstm_lstm_cell_kernel:	L
8read_5_disablecopyonread_lstm_lstm_cell_recurrent_kernel:
;
,read_6_disablecopyonread_lstm_lstm_cell_bias:	,
"read_7_disablecopyonread_iteration:	 0
&read_8_disablecopyonread_learning_rate: H
5read_9_disablecopyonread_adam_m_lstm_lstm_cell_kernel:	I
6read_10_disablecopyonread_adam_v_lstm_lstm_cell_kernel:	T
@read_11_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel:
T
@read_12_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel:
C
4read_13_disablecopyonread_adam_m_lstm_lstm_cell_bias:	C
4read_14_disablecopyonread_adam_v_lstm_lstm_cell_bias:	@
-read_15_disablecopyonread_adam_m_dense_kernel:	@@
-read_16_disablecopyonread_adam_v_dense_kernel:	@9
+read_17_disablecopyonread_adam_m_dense_bias:@9
+read_18_disablecopyonread_adam_v_dense_bias:@A
/read_19_disablecopyonread_adam_m_dense_1_kernel:@A
/read_20_disablecopyonread_adam_v_dense_1_kernel:@;
-read_21_disablecopyonread_adam_m_dense_1_bias:;
-read_22_disablecopyonread_adam_v_dense_1_bias:+
!read_23_disablecopyonread_total_1: +
!read_24_disablecopyonread_count_1: )
read_25_disablecopyonread_total: )
read_26_disablecopyonread_count: 
savev2_const
identity_55¢MergeV2Checkpoints¢Read/DisableCopyOnRead¢Read/ReadVariableOp¢Read_1/DisableCopyOnRead¢Read_1/ReadVariableOp¢Read_10/DisableCopyOnRead¢Read_10/ReadVariableOp¢Read_11/DisableCopyOnRead¢Read_11/ReadVariableOp¢Read_12/DisableCopyOnRead¢Read_12/ReadVariableOp¢Read_13/DisableCopyOnRead¢Read_13/ReadVariableOp¢Read_14/DisableCopyOnRead¢Read_14/ReadVariableOp¢Read_15/DisableCopyOnRead¢Read_15/ReadVariableOp¢Read_16/DisableCopyOnRead¢Read_16/ReadVariableOp¢Read_17/DisableCopyOnRead¢Read_17/ReadVariableOp¢Read_18/DisableCopyOnRead¢Read_18/ReadVariableOp¢Read_19/DisableCopyOnRead¢Read_19/ReadVariableOp¢Read_2/DisableCopyOnRead¢Read_2/ReadVariableOp¢Read_20/DisableCopyOnRead¢Read_20/ReadVariableOp¢Read_21/DisableCopyOnRead¢Read_21/ReadVariableOp¢Read_22/DisableCopyOnRead¢Read_22/ReadVariableOp¢Read_23/DisableCopyOnRead¢Read_23/ReadVariableOp¢Read_24/DisableCopyOnRead¢Read_24/ReadVariableOp¢Read_25/DisableCopyOnRead¢Read_25/ReadVariableOp¢Read_26/DisableCopyOnRead¢Read_26/ReadVariableOp¢Read_3/DisableCopyOnRead¢Read_3/ReadVariableOp¢Read_4/DisableCopyOnRead¢Read_4/ReadVariableOp¢Read_5/DisableCopyOnRead¢Read_5/ReadVariableOp¢Read_6/DisableCopyOnRead¢Read_6/ReadVariableOp¢Read_7/DisableCopyOnRead¢Read_7/ReadVariableOp¢Read_8/DisableCopyOnRead¢Read_8/ReadVariableOp¢Read_9/DisableCopyOnRead¢Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
  
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	@w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 §
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 ”
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_4/DisableCopyOnReadDisableCopyOnRead.read_4_disablecopyonread_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 Æ
Read_4/ReadVariableOpReadVariableOp.read_4_disablecopyonread_lstm_lstm_cell_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ŗ
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_lstm_lstm_cell_recurrent_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_6/DisableCopyOnReadDisableCopyOnRead,read_6_disablecopyonread_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ©
Read_6/ReadVariableOpReadVariableOp,read_6_disablecopyonread_lstm_lstm_cell_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:v
Read_7/DisableCopyOnReadDisableCopyOnRead"read_7_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOp"read_7_disablecopyonread_iteration^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_9/DisableCopyOnReadDisableCopyOnRead5read_9_disablecopyonread_adam_m_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ¶
Read_9/ReadVariableOpReadVariableOp5read_9_disablecopyonread_adam_m_lstm_lstm_cell_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_10/DisableCopyOnReadDisableCopyOnRead6read_10_disablecopyonread_adam_v_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ¹
Read_10/ReadVariableOpReadVariableOp6read_10_disablecopyonread_adam_v_lstm_lstm_cell_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_11/DisableCopyOnReadDisableCopyOnRead@read_11_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Ä
Read_11/ReadVariableOpReadVariableOp@read_11_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel^Read_11/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_12/DisableCopyOnReadDisableCopyOnRead@read_12_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Ä
Read_12/ReadVariableOpReadVariableOp@read_12_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_13/DisableCopyOnReadDisableCopyOnRead4read_13_disablecopyonread_adam_m_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ³
Read_13/ReadVariableOpReadVariableOp4read_13_disablecopyonread_adam_m_lstm_lstm_cell_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_14/DisableCopyOnReadDisableCopyOnRead4read_14_disablecopyonread_adam_v_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ³
Read_14/ReadVariableOpReadVariableOp4read_14_disablecopyonread_adam_v_lstm_lstm_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 °
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_m_dense_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 °
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_v_dense_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_17/DisableCopyOnReadDisableCopyOnRead+read_17_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 ©
Read_17/ReadVariableOpReadVariableOp+read_17_disablecopyonread_adam_m_dense_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead+read_18_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 ©
Read_18/ReadVariableOpReadVariableOp+read_18_disablecopyonread_adam_v_dense_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 ±
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_m_dense_1_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 ±
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_v_dense_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 «
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_m_dense_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 «
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_adam_v_dense_1_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_23/DisableCopyOnReadDisableCopyOnRead!read_23_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_23/ReadVariableOpReadVariableOp!read_23_disablecopyonread_total_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_count_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_25/DisableCopyOnReadDisableCopyOnReadread_25_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_25/ReadVariableOpReadVariableOpread_25_disablecopyonread_total^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_count^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: ģ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH„
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ą
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_54Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_55IdentityIdentity_54:output:0^NoOp*
T0*
_output_shapes
: Ś
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ņ
±

lstm_while_body_269418&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   æ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
$lstm/while/lstm_cell/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻi
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
"lstm/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Æ
 lstm/while/lstm_cell/dropout/MulMul'lstm/while/lstm_cell/ones_like:output:0+lstm/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
"lstm/while/lstm_cell/dropout/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¶
9lstm/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform+lstm/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0p
+lstm/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>å
)lstm/while/lstm_cell/dropout/GreaterEqualGreaterEqualBlstm/while/lstm_cell/dropout/random_uniform/RandomUniform:output:04lstm/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ē
%lstm/while/lstm_cell/dropout/SelectV2SelectV2-lstm/while/lstm_cell/dropout/GreaterEqual:z:0$lstm/while/lstm_cell/dropout/Mul:z:0-lstm/while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
"lstm/while/lstm_cell/dropout_1/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_1/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻŗ
;lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ė
+lstm/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ļ
'lstm/while/lstm_cell/dropout_1/SelectV2SelectV2/lstm/while/lstm_cell/dropout_1/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_1/Mul:z:0/lstm/while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
"lstm/while/lstm_cell/dropout_2/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_2/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻŗ
;lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ė
+lstm/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ļ
'lstm/while/lstm_cell/dropout_2/SelectV2SelectV2/lstm/while/lstm_cell/dropout_2/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_2/Mul:z:0/lstm/while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
"lstm/while/lstm_cell/dropout_3/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_3/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻŗ
;lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ė
+lstm/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ļ
'lstm/while/lstm_cell/dropout_3/SelectV2SelectV2/lstm/while/lstm_cell/dropout_3/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_3/Mul:z:0/lstm/while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’|
&lstm/while/lstm_cell/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
::ķĻk
&lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
 lstm/while/lstm_cell/ones_like_1Fill/lstm/while/lstm_cell/ones_like_1/Shape:output:0/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
"lstm/while/lstm_cell/dropout_4/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_4/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ»
;lstm/while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ģ
+lstm/while/lstm_cell/dropout_4/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    š
'lstm/while/lstm_cell/dropout_4/SelectV2SelectV2/lstm/while/lstm_cell/dropout_4/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_4/Mul:z:0/lstm/while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
"lstm/while/lstm_cell/dropout_5/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_5/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ»
;lstm/while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ģ
+lstm/while/lstm_cell/dropout_5/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    š
'lstm/while/lstm_cell/dropout_5/SelectV2SelectV2/lstm/while/lstm_cell/dropout_5/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_5/Mul:z:0/lstm/while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
"lstm/while/lstm_cell/dropout_6/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_6/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ»
;lstm/while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ģ
+lstm/while/lstm_cell/dropout_6/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    š
'lstm/while/lstm_cell/dropout_6/SelectV2SelectV2/lstm/while/lstm_cell/dropout_6/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_6/Mul:z:0/lstm/while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
$lstm/while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
"lstm/while/lstm_cell/dropout_7/MulMul)lstm/while/lstm_cell/ones_like_1:output:0-lstm/while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
$lstm/while/lstm_cell/dropout_7/ShapeShape)lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ»
;lstm/while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ģ
+lstm/while/lstm_cell/dropout_7/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    š
'lstm/while/lstm_cell/dropout_7/SelectV2SelectV2/lstm/while/lstm_cell/dropout_7/GreaterEqual:z:0&lstm/while/lstm_cell/dropout_7/Mul:z:0/lstm/while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ø
lstm/while/lstm_cell/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm/while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¼
lstm/while/lstm_cell/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¼
lstm/while/lstm_cell/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¼
lstm/while/lstm_cell/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:00lstm/while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0į
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/while/lstm_cell/MatMulMatMullstm/while/lstm_cell/mul:z:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_1MatMullstm/while/lstm_cell/mul_1:z:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_2MatMullstm/while/lstm_cell/mul_2:z:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_3MatMullstm/while/lstm_cell/mul_3:z:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitØ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while/lstm_cell/mul_4Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while/lstm_cell/mul_5Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while/lstm_cell/mul_6Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while/lstm_cell/mul_7Mullstm_while_placeholder_20lstm/while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask§
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul_4:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’x
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_5:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_8Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_6:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_9Mul lstm/while/lstm_cell/Sigmoid:y:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_8:z:0lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_7:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’v
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_10Mul"lstm/while/lstm_cell/Sigmoid_2:y:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’w
5lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ’
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1>lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_10:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’É
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"3
lstm_while_identitylstm/while/Identity:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namelstm/while/maximum_iterations:O K

_output_shapes
: 
1
_user_specified_namelstm/while/loop_counter


Æ
&__inference_model_layer_call_fn_268975
input_1
input_2
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	@
	unknown_3:@
	unknown_4:@
	unknown_5:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_268958o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ö¹
Ń
A__inference_model_layer_call_and_return_conditional_losses_270544
inputs_0
inputs_1?
,lstm_lstm_cell_split_readvariableop_resource:	=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢lstm/lstm_cell/ReadVariableOp_4¢lstm/lstm_cell/ReadVariableOp_5¢lstm/lstm_cell/ReadVariableOp_6¢lstm/lstm_cell/ReadVariableOp_7¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢%lstm/lstm_cell/split_2/ReadVariableOp¢%lstm/lstm_cell/split_3/ReadVariableOp¢
lstm/while¢lstm/while_1U
reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
::ķĻg
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:~
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’S
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻe
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ł
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Æ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:z
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’b

lstm/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
::ķĻb
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm/transpose	Transposereshape_1/Reshape:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ć
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ļ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masky
lstm/lstm_cell/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
::ķĻc
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
 lstm/lstm_cell/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
lstm/lstm_cell/ones_like_1Fill)lstm/lstm_cell/ones_like_1/Shape:output:0)lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mulMullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_1Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_2Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_3Mullstm/strided_slice_2:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ļ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/lstm_cell/MatMulMatMullstm/lstm_cell/mul:z:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_1MatMullstm/lstm_cell/mul_1:z:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_2MatMullstm/lstm_cell/mul_2:z:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_3MatMullstm/lstm_cell/mul_3:z:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_4Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_5Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_6Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_7Mullstm/zeros:output:0#lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ø
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul_4:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’l
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_5:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_8Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_6:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’h
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_9Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_8:z:0lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_7:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_10Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   c
!lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ō
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0*lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_270152*"
condR
lstm_while_cond_270151*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ę
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsm
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
lstm/Shape_2Shapereshape/Reshape:output:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_4StridedSlicelstm/Shape_2:output:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_2/packedPacklstm/strided_slice_4:output:0lstm/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_2Filllstm/zeros_2/packed:output:0lstm/zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
lstm/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_3/packedPacklstm/strided_slice_4:output:0lstm/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_3Filllstm/zeros_3/packed:output:0lstm/zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm/transpose_2	Transposereshape/Reshape:output:0lstm/transpose_2/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’^
lstm/Shape_3Shapelstm/transpose_2:y:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_5StridedSlicelstm/Shape_3:output:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
lstm/TensorArrayV2_3TensorListReserve+lstm/TensorArrayV2_3/element_shape:output:0lstm/strided_slice_5:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
<lstm/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   õ
.lstm/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm/transpose_2:y:0Elstm/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅd
lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_6StridedSlicelstm/transpose_2:y:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask{
 lstm/lstm_cell/ones_like_2/ShapeShapelstm/strided_slice_6:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ŗ
lstm/lstm_cell/ones_like_2Fill)lstm/lstm_cell/ones_like_2/Shape:output:0)lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
 lstm/lstm_cell/ones_like_3/ShapeShapelstm/zeros_2:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
lstm/lstm_cell/ones_like_3Fill)lstm/lstm_cell/ones_like_3/Shape:output:0)lstm/lstm_cell/ones_like_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_11Mullstm/strided_slice_6:output:0#lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_12Mullstm/strided_slice_6:output:0#lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_13Mullstm/strided_slice_6:output:0#lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_14Mullstm/strided_slice_6:output:0#lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%lstm/lstm_cell/split_2/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Õ
lstm/lstm_cell/split_2Split)lstm/lstm_cell/split_2/split_dim:output:0-lstm/lstm_cell/split_2/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/lstm_cell/MatMul_8MatMullstm/lstm_cell/mul_11:z:0lstm/lstm_cell/split_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_9MatMullstm/lstm_cell/mul_12:z:0lstm/lstm_cell/split_2:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_10MatMullstm/lstm_cell/mul_13:z:0lstm/lstm_cell/split_2:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_11MatMullstm/lstm_cell/mul_14:z:0lstm/lstm_cell/split_2:output:3*
T0*(
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_3/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_3/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_3Split)lstm/lstm_cell/split_3/split_dim:output:0-lstm/lstm_cell/split_3/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAdd_4BiasAdd!lstm/lstm_cell/MatMul_8:product:0lstm/lstm_cell/split_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_5BiasAdd!lstm/lstm_cell/MatMul_9:product:0lstm/lstm_cell/split_3:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_6BiasAdd"lstm/lstm_cell/MatMul_10:product:0lstm/lstm_cell/split_3:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_7BiasAdd"lstm/lstm_cell/MatMul_11:product:0lstm/lstm_cell/split_3:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_15Mullstm/zeros_2:output:0#lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_16Mullstm/zeros_2:output:0#lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_17Mullstm/zeros_2:output:0#lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_18Mullstm/zeros_2:output:0#lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_4ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_4StridedSlice'lstm/lstm_cell/ReadVariableOp_4:value:0-lstm/lstm_cell/strided_slice_4/stack:output:0/lstm/lstm_cell/strided_slice_4/stack_1:output:0/lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_12MatMullstm/lstm_cell/mul_15:z:0'lstm/lstm_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_5AddV2!lstm/lstm_cell/BiasAdd_4:output:0"lstm/lstm_cell/MatMul_12:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_3Sigmoidlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_5ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_5StridedSlice'lstm/lstm_cell/ReadVariableOp_5:value:0-lstm/lstm_cell/strided_slice_5/stack:output:0/lstm/lstm_cell/strided_slice_5/stack_1:output:0/lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_13MatMullstm/lstm_cell/mul_16:z:0'lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_5:output:0"lstm/lstm_cell/MatMul_13:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_4Sigmoidlstm/lstm_cell/add_6:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_19Mullstm/lstm_cell/Sigmoid_4:y:0lstm/zeros_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_6ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_6StridedSlice'lstm/lstm_cell/ReadVariableOp_6:value:0-lstm/lstm_cell/strided_slice_6/stack:output:0/lstm/lstm_cell/strided_slice_6/stack_1:output:0/lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_14MatMullstm/lstm_cell/mul_17:z:0'lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_7AddV2!lstm/lstm_cell/BiasAdd_6:output:0"lstm/lstm_cell/MatMul_14:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/add_7:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_20Mullstm/lstm_cell/Sigmoid_3:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_8AddV2lstm/lstm_cell/mul_19:z:0lstm/lstm_cell/mul_20:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_7ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_7StridedSlice'lstm/lstm_cell/ReadVariableOp_7:value:0-lstm/lstm_cell/strided_slice_7/stack:output:0/lstm/lstm_cell/strided_slice_7/stack_1:output:0/lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_15MatMullstm/lstm_cell/mul_18:z:0'lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_9AddV2!lstm/lstm_cell/BiasAdd_7:output:0"lstm/lstm_cell/MatMul_15:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_5Sigmoidlstm/lstm_cell/add_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/add_8:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_21Mullstm/lstm_cell/Sigmoid_5:y:0lstm/lstm_cell/Tanh_3:y:0*
T0*(
_output_shapes
:’’’’’’’’’s
"lstm/TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   c
!lstm/TensorArrayV2_4/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ō
lstm/TensorArrayV2_4TensorListReserve+lstm/TensorArrayV2_4/element_shape:output:0*lstm/TensorArrayV2_4/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅM
lstm/time_1Const*
_output_shapes
: *
dtype0*
value	B : j
lstm/while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’[
lstm/while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : É
lstm/while_1While"lstm/while_1/loop_counter:output:0(lstm/while_1/maximum_iterations:output:0lstm/time_1:output:0lstm/TensorArrayV2_4:handle:0lstm/zeros_2:output:0lstm/zeros_3:output:0lstm/strided_slice_5:output:0>lstm/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_while_1_body_270390*$
condR
lstm_while_1_cond_270389*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
7lstm/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ģ
)lstm/TensorArrayV2Stack_1/TensorListStackTensorListStacklstm/while_1:output:3@lstm/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsm
lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’f
lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
lstm/strided_slice_7StridedSlice2lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskj
lstm/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
lstm/transpose_3	Transpose2lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0lstm/transpose_3/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’b
lstm/runtime_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout/IdentityIdentitylstm/strided_slice_7:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout_1/IdentityIdentitylstm/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :“
concatenate/concatConcatV2dropout/Identity:output:0dropout_1/Identity:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’@j
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3 ^lstm/lstm_cell/ReadVariableOp_4 ^lstm/lstm_cell/ReadVariableOp_5 ^lstm/lstm_cell/ReadVariableOp_6 ^lstm/lstm_cell/ReadVariableOp_7$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp&^lstm/lstm_cell/split_2/ReadVariableOp&^lstm/lstm_cell/split_3/ReadVariableOp^lstm/while^lstm/while_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32B
lstm/lstm_cell/ReadVariableOp_4lstm/lstm_cell/ReadVariableOp_42B
lstm/lstm_cell/ReadVariableOp_5lstm/lstm_cell/ReadVariableOp_52B
lstm/lstm_cell/ReadVariableOp_6lstm/lstm_cell/ReadVariableOp_62B
lstm/lstm_cell/ReadVariableOp_7lstm/lstm_cell/ReadVariableOp_72>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2N
%lstm/lstm_cell/split_2/ReadVariableOp%lstm/lstm_cell/split_2/ReadVariableOp2N
%lstm/lstm_cell/split_3/ReadVariableOp%lstm/lstm_cell/split_3/ReadVariableOp2
lstm/while_1lstm/while_12

lstm/while
lstm/while:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0
Ć+
“
A__inference_model_layer_call_and_return_conditional_losses_268958

inputs
inputs_1
lstm_268932:	
lstm_268934:	
lstm_268936:

dense_268946:	@
dense_268948:@ 
dense_1_268952:@
dense_1_268954:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢lstm/StatefulPartitionedCall_1Ą
reshape_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143ŗ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_268158
lstm/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_268932lstm_268934lstm_268936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268532
lstm/StatefulPartitionedCall_1StatefulPartitionedCall reshape/PartitionedCall:output:0lstm_268932lstm_268934lstm_268936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268532č
dropout/StatefulPartitionedCallStatefulPartitionedCall'lstm/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268556
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268570
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_268579
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_268946dense_268948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_268592
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268610
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_268952dense_1_268954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_268623w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’²
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
lstm/StatefulPartitionedCall_1lstm/StatefulPartitionedCall_12<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾
q
G__inference_concatenate_layer_call_and_return_conditional_losses_268579

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’:’’’’’’’’’:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


model_lstm_while_body_2672152
.model_lstm_while_model_lstm_while_loop_counter8
4model_lstm_while_model_lstm_while_maximum_iterations 
model_lstm_while_placeholder"
model_lstm_while_placeholder_1"
model_lstm_while_placeholder_2"
model_lstm_while_placeholder_31
-model_lstm_while_model_lstm_strided_slice_1_0m
imodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0M
:model_lstm_while_lstm_cell_split_readvariableop_resource_0:	K
<model_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	H
4model_lstm_while_lstm_cell_readvariableop_resource_0:

model_lstm_while_identity
model_lstm_while_identity_1
model_lstm_while_identity_2
model_lstm_while_identity_3
model_lstm_while_identity_4
model_lstm_while_identity_5/
+model_lstm_while_model_lstm_strided_slice_1k
gmodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorK
8model_lstm_while_lstm_cell_split_readvariableop_resource:	I
:model_lstm_while_lstm_cell_split_1_readvariableop_resource:	F
2model_lstm_while_lstm_cell_readvariableop_resource:
¢)model/lstm/while/lstm_cell/ReadVariableOp¢+model/lstm/while/lstm_cell/ReadVariableOp_1¢+model/lstm/while/lstm_cell/ReadVariableOp_2¢+model/lstm/while/lstm_cell/ReadVariableOp_3¢/model/lstm/while/lstm_cell/split/ReadVariableOp¢1model/lstm/while/lstm_cell/split_1/ReadVariableOp
Bmodel/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   Ż
4model/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemimodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0model_lstm_while_placeholderKmodel/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0£
*model/lstm/while/lstm_cell/ones_like/ShapeShape;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻo
*model/lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Č
$model/lstm/while/lstm_cell/ones_likeFill3model/lstm/while/lstm_cell/ones_like/Shape:output:03model/lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
,model/lstm/while/lstm_cell/ones_like_1/ShapeShapemodel_lstm_while_placeholder_2*
T0*
_output_shapes
::ķĻq
,model/lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ļ
&model/lstm/while/lstm_cell/ones_like_1Fill5model/lstm/while/lstm_cell/ones_like_1/Shape:output:05model/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ć
model/lstm/while/lstm_cell/mulMul;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-model/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Å
 model/lstm/while/lstm_cell/mul_1Mul;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-model/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Å
 model/lstm/while/lstm_cell/mul_2Mul;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-model/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Å
 model/lstm/while/lstm_cell/mul_3Mul;model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-model/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’l
*model/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
/model/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp:model_lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ó
 model/lstm/while/lstm_cell/splitSplit3model/lstm/while/lstm_cell/split/split_dim:output:07model/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split­
!model/lstm/while/lstm_cell/MatMulMatMul"model/lstm/while/lstm_cell/mul:z:0)model/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’±
#model/lstm/while/lstm_cell/MatMul_1MatMul$model/lstm/while/lstm_cell/mul_1:z:0)model/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’±
#model/lstm/while/lstm_cell/MatMul_2MatMul$model/lstm/while/lstm_cell/mul_2:z:0)model/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’±
#model/lstm/while/lstm_cell/MatMul_3MatMul$model/lstm/while/lstm_cell/mul_3:z:0)model/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’n
,model/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : «
1model/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp<model_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0é
"model/lstm/while/lstm_cell/split_1Split5model/lstm/while/lstm_cell/split_1/split_dim:output:09model/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitŗ
"model/lstm/while/lstm_cell/BiasAddBiasAdd+model/lstm/while/lstm_cell/MatMul:product:0+model/lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¾
$model/lstm/while/lstm_cell/BiasAdd_1BiasAdd-model/lstm/while/lstm_cell/MatMul_1:product:0+model/lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’¾
$model/lstm/while/lstm_cell/BiasAdd_2BiasAdd-model/lstm/while/lstm_cell/MatMul_2:product:0+model/lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’¾
$model/lstm/while/lstm_cell/BiasAdd_3BiasAdd-model/lstm/while/lstm_cell/MatMul_3:product:0+model/lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’«
 model/lstm/while/lstm_cell/mul_4Mulmodel_lstm_while_placeholder_2/model/lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’«
 model/lstm/while/lstm_cell/mul_5Mulmodel_lstm_while_placeholder_2/model/lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’«
 model/lstm/while/lstm_cell/mul_6Mulmodel_lstm_while_placeholder_2/model/lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’«
 model/lstm/while/lstm_cell/mul_7Mulmodel_lstm_while_placeholder_2/model/lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ 
)model/lstm/while/lstm_cell/ReadVariableOpReadVariableOp4model_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
.model/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0model/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0model/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ō
(model/lstm/while/lstm_cell/strided_sliceStridedSlice1model/lstm/while/lstm_cell/ReadVariableOp:value:07model/lstm/while/lstm_cell/strided_slice/stack:output:09model/lstm/while/lstm_cell/strided_slice/stack_1:output:09model/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask¹
#model/lstm/while/lstm_cell/MatMul_4MatMul$model/lstm/while/lstm_cell/mul_4:z:01model/lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’¶
model/lstm/while/lstm_cell/addAddV2+model/lstm/while/lstm_cell/BiasAdd:output:0-model/lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’
"model/lstm/while/lstm_cell/SigmoidSigmoid"model/lstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’¢
+model/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp4model_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0model/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2model/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2model/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ž
*model/lstm/while/lstm_cell/strided_slice_1StridedSlice3model/lstm/while/lstm_cell/ReadVariableOp_1:value:09model/lstm/while/lstm_cell/strided_slice_1/stack:output:0;model/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0;model/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask»
#model/lstm/while/lstm_cell/MatMul_5MatMul$model/lstm/while/lstm_cell/mul_5:z:03model/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’ŗ
 model/lstm/while/lstm_cell/add_1AddV2-model/lstm/while/lstm_cell/BiasAdd_1:output:0-model/lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’
$model/lstm/while/lstm_cell/Sigmoid_1Sigmoid$model/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’¤
 model/lstm/while/lstm_cell/mul_8Mul(model/lstm/while/lstm_cell/Sigmoid_1:y:0model_lstm_while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’¢
+model/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp4model_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0model/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2model/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
2model/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ž
*model/lstm/while/lstm_cell/strided_slice_2StridedSlice3model/lstm/while/lstm_cell/ReadVariableOp_2:value:09model/lstm/while/lstm_cell/strided_slice_2/stack:output:0;model/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0;model/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask»
#model/lstm/while/lstm_cell/MatMul_6MatMul$model/lstm/while/lstm_cell/mul_6:z:03model/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’ŗ
 model/lstm/while/lstm_cell/add_2AddV2-model/lstm/while/lstm_cell/BiasAdd_2:output:0-model/lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/while/lstm_cell/TanhTanh$model/lstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’§
 model/lstm/while/lstm_cell/mul_9Mul&model/lstm/while/lstm_cell/Sigmoid:y:0#model/lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
 model/lstm/while/lstm_cell/add_3AddV2$model/lstm/while/lstm_cell/mul_8:z:0$model/lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’¢
+model/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp4model_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0model/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
2model/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2model/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ž
*model/lstm/while/lstm_cell/strided_slice_3StridedSlice3model/lstm/while/lstm_cell/ReadVariableOp_3:value:09model/lstm/while/lstm_cell/strided_slice_3/stack:output:0;model/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0;model/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask»
#model/lstm/while/lstm_cell/MatMul_7MatMul$model/lstm/while/lstm_cell/mul_7:z:03model/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’ŗ
 model/lstm/while/lstm_cell/add_4AddV2-model/lstm/while/lstm_cell/BiasAdd_3:output:0-model/lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’
$model/lstm/while/lstm_cell/Sigmoid_2Sigmoid$model/lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’
!model/lstm/while/lstm_cell/Tanh_1Tanh$model/lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’¬
!model/lstm/while/lstm_cell/mul_10Mul(model/lstm/while/lstm_cell/Sigmoid_2:y:0%model/lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’}
;model/lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
5model/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_lstm_while_placeholder_1Dmodel/lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0%model/lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅX
model/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :}
model/lstm/while/addAddV2model_lstm_while_placeholdermodel/lstm/while/add/y:output:0*
T0*
_output_shapes
: Z
model/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm/while/add_1AddV2.model_lstm_while_model_lstm_while_loop_counter!model/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: z
model/lstm/while/IdentityIdentitymodel/lstm/while/add_1:z:0^model/lstm/while/NoOp*
T0*
_output_shapes
: 
model/lstm/while/Identity_1Identity4model_lstm_while_model_lstm_while_maximum_iterations^model/lstm/while/NoOp*
T0*
_output_shapes
: z
model/lstm/while/Identity_2Identitymodel/lstm/while/add:z:0^model/lstm/while/NoOp*
T0*
_output_shapes
: §
model/lstm/while/Identity_3IdentityEmodel/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lstm/while/NoOp*
T0*
_output_shapes
: 
model/lstm/while/Identity_4Identity%model/lstm/while/lstm_cell/mul_10:z:0^model/lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/while/Identity_5Identity$model/lstm/while/lstm_cell/add_3:z:0^model/lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’ó
model/lstm/while/NoOpNoOp*^model/lstm/while/lstm_cell/ReadVariableOp,^model/lstm/while/lstm_cell/ReadVariableOp_1,^model/lstm/while/lstm_cell/ReadVariableOp_2,^model/lstm/while/lstm_cell/ReadVariableOp_30^model/lstm/while/lstm_cell/split/ReadVariableOp2^model/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
model_lstm_while_identity_1$model/lstm/while/Identity_1:output:0"C
model_lstm_while_identity_2$model/lstm/while/Identity_2:output:0"C
model_lstm_while_identity_3$model/lstm/while/Identity_3:output:0"C
model_lstm_while_identity_4$model/lstm/while/Identity_4:output:0"C
model_lstm_while_identity_5$model/lstm/while/Identity_5:output:0"?
model_lstm_while_identity"model/lstm/while/Identity:output:0"j
2model_lstm_while_lstm_cell_readvariableop_resource4model_lstm_while_lstm_cell_readvariableop_resource_0"z
:model_lstm_while_lstm_cell_split_1_readvariableop_resource<model_lstm_while_lstm_cell_split_1_readvariableop_resource_0"v
8model_lstm_while_lstm_cell_split_readvariableop_resource:model_lstm_while_lstm_cell_split_readvariableop_resource_0"\
+model_lstm_while_model_lstm_strided_slice_1-model_lstm_while_model_lstm_strided_slice_1_0"Ō
gmodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensorimodel_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2Z
+model/lstm/while/lstm_cell/ReadVariableOp_1+model/lstm/while/lstm_cell/ReadVariableOp_12Z
+model/lstm/while/lstm_cell/ReadVariableOp_2+model/lstm/while/lstm_cell/ReadVariableOp_22Z
+model/lstm/while/lstm_cell/ReadVariableOp_3+model/lstm/while/lstm_cell/ReadVariableOp_32V
)model/lstm/while/lstm_cell/ReadVariableOp)model/lstm/while/lstm_cell/ReadVariableOp2b
/model/lstm/while/lstm_cell/split/ReadVariableOp/model/lstm/while/lstm_cell/split/ReadVariableOp2f
1model/lstm/while/lstm_cell/split_1/ReadVariableOp1model/lstm/while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :[W

_output_shapes
: 
=
_user_specified_name%#model/lstm/while/maximum_iterations:U Q

_output_shapes
: 
7
_user_specified_namemodel/lstm/while/loop_counter

µ
%__inference_lstm_layer_call_fn_270602
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268067p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
Ų
Å	
!__inference__wrapped_model_267607
input_1
input_2E
2model_lstm_lstm_cell_split_readvariableop_resource:	C
4model_lstm_lstm_cell_split_1_readvariableop_resource:	@
,model_lstm_lstm_cell_readvariableop_resource:
=
*model_dense_matmul_readvariableop_resource:	@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@;
-model_dense_1_biasadd_readvariableop_resource:
identity¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢#model/lstm/lstm_cell/ReadVariableOp¢%model/lstm/lstm_cell/ReadVariableOp_1¢%model/lstm/lstm_cell/ReadVariableOp_2¢%model/lstm/lstm_cell/ReadVariableOp_3¢%model/lstm/lstm_cell/ReadVariableOp_4¢%model/lstm/lstm_cell/ReadVariableOp_5¢%model/lstm/lstm_cell/ReadVariableOp_6¢%model/lstm/lstm_cell/ReadVariableOp_7¢)model/lstm/lstm_cell/split/ReadVariableOp¢+model/lstm/lstm_cell/split_1/ReadVariableOp¢+model/lstm/lstm_cell/split_2/ReadVariableOp¢+model/lstm/lstm_cell/split_3/ReadVariableOp¢model/lstm/while¢model/lstm/while_1Z
model/reshape_1/ShapeShapeinput_2*
T0*
_output_shapes
::ķĻm
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ļ
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
model/reshape_1/ReshapeReshapeinput_2&model/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’X
model/reshape/ShapeShapeinput_1*
T0*
_output_shapes
::ķĻk
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :_
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ē
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
model/reshape/ReshapeReshapeinput_1$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’n
model/lstm/ShapeShape model/reshape_1/Reshape:output:0*
T0*
_output_shapes
::ķĻh
model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_sliceStridedSlicemodel/lstm/Shape:output:0'model/lstm/strided_slice/stack:output:0)model/lstm/strided_slice/stack_1:output:0)model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
model/lstm/zeros/packedPack!model/lstm/strided_slice:output:0"model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:[
model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zerosFill model/lstm/zeros/packed:output:0model/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
model/lstm/zeros_1/packedPack!model/lstm/strided_slice:output:0$model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zeros_1Fill"model/lstm/zeros_1/packed:output:0!model/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’n
model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model/lstm/transpose	Transpose model/reshape_1/Reshape:output:0"model/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’h
model/lstm/Shape_1Shapemodel/lstm/transpose:y:0*
T0*
_output_shapes
::ķĻj
 model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_slice_1StridedSlicemodel/lstm/Shape_1:output:0)model/lstm/strided_slice_1/stack:output:0+model/lstm/strided_slice_1/stack_1:output:0+model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
&model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Õ
model/lstm/TensorArrayV2TensorListReserve/model/lstm/TensorArrayV2/element_shape:output:0#model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
@model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
2model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm/transpose:y:0Imodel/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅj
 model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
model/lstm/strided_slice_2StridedSlicemodel/lstm/transpose:y:0)model/lstm/strided_slice_2/stack:output:0+model/lstm/strided_slice_2/stack_1:output:0+model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
$model/lstm/lstm_cell/ones_like/ShapeShape#model/lstm/strided_slice_2:output:0*
T0*
_output_shapes
::ķĻi
$model/lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
model/lstm/lstm_cell/ones_likeFill-model/lstm/lstm_cell/ones_like/Shape:output:0-model/lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
&model/lstm/lstm_cell/ones_like_1/ShapeShapemodel/lstm/zeros:output:0*
T0*
_output_shapes
::ķĻk
&model/lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
 model/lstm/lstm_cell/ones_like_1Fill/model/lstm/lstm_cell/ones_like_1/Shape:output:0/model/lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mulMul#model/lstm/strided_slice_2:output:0'model/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’”
model/lstm/lstm_cell/mul_1Mul#model/lstm/strided_slice_2:output:0'model/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’”
model/lstm/lstm_cell/mul_2Mul#model/lstm/strided_slice_2:output:0'model/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’”
model/lstm/lstm_cell/mul_3Mul#model/lstm/strided_slice_2:output:0'model/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
$model/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)model/lstm/lstm_cell/split/ReadVariableOpReadVariableOp2model_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0į
model/lstm/lstm_cell/splitSplit-model/lstm/lstm_cell/split/split_dim:output:01model/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
model/lstm/lstm_cell/MatMulMatMulmodel/lstm/lstm_cell/mul:z:0#model/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/MatMul_1MatMulmodel/lstm/lstm_cell/mul_1:z:0#model/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/MatMul_2MatMulmodel/lstm/lstm_cell/mul_2:z:0#model/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/MatMul_3MatMulmodel/lstm/lstm_cell/mul_3:z:0#model/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’h
&model/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+model/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp4model_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0×
model/lstm/lstm_cell/split_1Split/model/lstm/lstm_cell/split_1/split_dim:output:03model/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitØ
model/lstm/lstm_cell/BiasAddBiasAdd%model/lstm/lstm_cell/MatMul:product:0%model/lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¬
model/lstm/lstm_cell/BiasAdd_1BiasAdd'model/lstm/lstm_cell/MatMul_1:product:0%model/lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’¬
model/lstm/lstm_cell/BiasAdd_2BiasAdd'model/lstm/lstm_cell/MatMul_2:product:0%model/lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’¬
model/lstm/lstm_cell/BiasAdd_3BiasAdd'model/lstm/lstm_cell/MatMul_3:product:0%model/lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_4Mulmodel/lstm/zeros:output:0)model/lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_5Mulmodel/lstm/zeros:output:0)model/lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_6Mulmodel/lstm/zeros:output:0)model/lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_7Mulmodel/lstm/zeros:output:0)model/lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
#model/lstm/lstm_cell/ReadVariableOpReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(model/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*model/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*model/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"model/lstm/lstm_cell/strided_sliceStridedSlice+model/lstm/lstm_cell/ReadVariableOp:value:01model/lstm/lstm_cell/strided_slice/stack:output:03model/lstm/lstm_cell/strided_slice/stack_1:output:03model/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask§
model/lstm/lstm_cell/MatMul_4MatMulmodel/lstm/lstm_cell/mul_4:z:0+model/lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’¤
model/lstm/lstm_cell/addAddV2%model/lstm/lstm_cell/BiasAdd:output:0'model/lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’x
model/lstm/lstm_cell/SigmoidSigmoidmodel/lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_1StridedSlice-model/lstm/lstm_cell/ReadVariableOp_1:value:03model/lstm/lstm_cell/strided_slice_1/stack:output:05model/lstm/lstm_cell/strided_slice_1/stack_1:output:05model/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
model/lstm/lstm_cell/MatMul_5MatMulmodel/lstm/lstm_cell/mul_5:z:0-model/lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
model/lstm/lstm_cell/add_1AddV2'model/lstm/lstm_cell/BiasAdd_1:output:0'model/lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/lstm/lstm_cell/Sigmoid_1Sigmoidmodel/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_8Mul"model/lstm/lstm_cell/Sigmoid_1:y:0model/lstm/zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,model/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_2StridedSlice-model/lstm/lstm_cell/ReadVariableOp_2:value:03model/lstm/lstm_cell/strided_slice_2/stack:output:05model/lstm/lstm_cell/strided_slice_2/stack_1:output:05model/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
model/lstm/lstm_cell/MatMul_6MatMulmodel/lstm/lstm_cell/mul_6:z:0-model/lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
model/lstm/lstm_cell/add_2AddV2'model/lstm/lstm_cell/BiasAdd_2:output:0'model/lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’t
model/lstm/lstm_cell/TanhTanhmodel/lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_9Mul model/lstm/lstm_cell/Sigmoid:y:0model/lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/add_3AddV2model/lstm/lstm_cell/mul_8:z:0model/lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,model/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,model/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_3StridedSlice-model/lstm/lstm_cell/ReadVariableOp_3:value:03model/lstm/lstm_cell/strided_slice_3/stack:output:05model/lstm/lstm_cell/strided_slice_3/stack_1:output:05model/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
model/lstm/lstm_cell/MatMul_7MatMulmodel/lstm/lstm_cell/mul_7:z:0-model/lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
model/lstm/lstm_cell/add_4AddV2'model/lstm/lstm_cell/BiasAdd_3:output:0'model/lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/lstm/lstm_cell/Sigmoid_2Sigmoidmodel/lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’v
model/lstm/lstm_cell/Tanh_1Tanhmodel/lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_10Mul"model/lstm/lstm_cell/Sigmoid_2:y:0model/lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’y
(model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   i
'model/lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ę
model/lstm/TensorArrayV2_1TensorListReserve1model/lstm/TensorArrayV2_1/element_shape:output:00model/lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅQ
model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : n
#model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’_
model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model/lstm/whileWhile&model/lstm/while/loop_counter:output:0,model/lstm/while/maximum_iterations:output:0model/lstm/time:output:0#model/lstm/TensorArrayV2_1:handle:0model/lstm/zeros:output:0model/lstm/zeros_1:output:0#model/lstm/strided_slice_1:output:0Bmodel/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02model_lstm_lstm_cell_split_readvariableop_resource4model_lstm_lstm_cell_split_1_readvariableop_resource,model_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
model_lstm_while_body_267215*(
cond R
model_lstm_while_cond_267214*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
;model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ų
-model/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm/while:output:3Dmodel/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementss
 model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’l
"model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
model/lstm/strided_slice_3StridedSlice6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/lstm/strided_slice_3/stack:output:0+model/lstm/strided_slice_3/stack_1:output:0+model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskp
model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ø
model/lstm/transpose_1	Transpose6model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0$model/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’f
model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
model/lstm/Shape_2Shapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
::ķĻj
 model/lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_slice_4StridedSlicemodel/lstm/Shape_2:output:0)model/lstm/strided_slice_4/stack:output:0+model/lstm/strided_slice_4/stack_1:output:0+model/lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
model/lstm/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
model/lstm/zeros_2/packedPack#model/lstm/strided_slice_4:output:0$model/lstm/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zeros_2Fill"model/lstm/zeros_2/packed:output:0!model/lstm/zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
model/lstm/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
model/lstm/zeros_3/packedPack#model/lstm/strided_slice_4:output:0$model/lstm/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:]
model/lstm/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/lstm/zeros_3Fill"model/lstm/zeros_3/packed:output:0!model/lstm/zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
model/lstm/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model/lstm/transpose_2	Transposemodel/reshape/Reshape:output:0$model/lstm/transpose_2/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
model/lstm/Shape_3Shapemodel/lstm/transpose_2:y:0*
T0*
_output_shapes
::ķĻj
 model/lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/lstm/strided_slice_5StridedSlicemodel/lstm/Shape_3:output:0)model/lstm/strided_slice_5/stack:output:0+model/lstm/strided_slice_5/stack_1:output:0+model/lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(model/lstm/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ł
model/lstm/TensorArrayV2_3TensorListReserve1model/lstm/TensorArrayV2_3/element_shape:output:0#model/lstm/strided_slice_5:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
Bmodel/lstm/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
4model/lstm/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormodel/lstm/transpose_2:y:0Kmodel/lstm/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅj
 model/lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"model/lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
model/lstm/strided_slice_6StridedSlicemodel/lstm/transpose_2:y:0)model/lstm/strided_slice_6/stack:output:0+model/lstm/strided_slice_6/stack_1:output:0+model/lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask
&model/lstm/lstm_cell/ones_like_2/ShapeShape#model/lstm/strided_slice_6:output:0*
T0*
_output_shapes
::ķĻk
&model/lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
 model/lstm/lstm_cell/ones_like_2Fill/model/lstm/lstm_cell/ones_like_2/Shape:output:0/model/lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
&model/lstm/lstm_cell/ones_like_3/ShapeShapemodel/lstm/zeros_2:output:0*
T0*
_output_shapes
::ķĻk
&model/lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
 model/lstm/lstm_cell/ones_like_3Fill/model/lstm/lstm_cell/ones_like_3/Shape:output:0/model/lstm/lstm_cell/ones_like_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’¤
model/lstm/lstm_cell/mul_11Mul#model/lstm/strided_slice_6:output:0)model/lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
model/lstm/lstm_cell/mul_12Mul#model/lstm/strided_slice_6:output:0)model/lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
model/lstm/lstm_cell/mul_13Mul#model/lstm/strided_slice_6:output:0)model/lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
model/lstm/lstm_cell/mul_14Mul#model/lstm/strided_slice_6:output:0)model/lstm/lstm_cell/ones_like_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’h
&model/lstm/lstm_cell/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
+model/lstm/lstm_cell/split_2/ReadVariableOpReadVariableOp2model_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0ē
model/lstm/lstm_cell/split_2Split/model/lstm/lstm_cell/split_2/split_dim:output:03model/lstm/lstm_cell/split_2/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split¢
model/lstm/lstm_cell/MatMul_8MatMulmodel/lstm/lstm_cell/mul_11:z:0%model/lstm/lstm_cell/split_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’¢
model/lstm/lstm_cell/MatMul_9MatMulmodel/lstm/lstm_cell/mul_12:z:0%model/lstm/lstm_cell/split_2:output:1*
T0*(
_output_shapes
:’’’’’’’’’£
model/lstm/lstm_cell/MatMul_10MatMulmodel/lstm/lstm_cell/mul_13:z:0%model/lstm/lstm_cell/split_2:output:2*
T0*(
_output_shapes
:’’’’’’’’’£
model/lstm/lstm_cell/MatMul_11MatMulmodel/lstm/lstm_cell/mul_14:z:0%model/lstm/lstm_cell/split_2:output:3*
T0*(
_output_shapes
:’’’’’’’’’h
&model/lstm/lstm_cell/split_3/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+model/lstm/lstm_cell/split_3/ReadVariableOpReadVariableOp4model_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0×
model/lstm/lstm_cell/split_3Split/model/lstm/lstm_cell/split_3/split_dim:output:03model/lstm/lstm_cell/split_3/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split¬
model/lstm/lstm_cell/BiasAdd_4BiasAdd'model/lstm/lstm_cell/MatMul_8:product:0%model/lstm/lstm_cell/split_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’¬
model/lstm/lstm_cell/BiasAdd_5BiasAdd'model/lstm/lstm_cell/MatMul_9:product:0%model/lstm/lstm_cell/split_3:output:1*
T0*(
_output_shapes
:’’’’’’’’’­
model/lstm/lstm_cell/BiasAdd_6BiasAdd(model/lstm/lstm_cell/MatMul_10:product:0%model/lstm/lstm_cell/split_3:output:2*
T0*(
_output_shapes
:’’’’’’’’’­
model/lstm/lstm_cell/BiasAdd_7BiasAdd(model/lstm/lstm_cell/MatMul_11:product:0%model/lstm/lstm_cell/split_3:output:3*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_15Mulmodel/lstm/zeros_2:output:0)model/lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_16Mulmodel/lstm/zeros_2:output:0)model/lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_17Mulmodel/lstm/zeros_2:output:0)model/lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_18Mulmodel/lstm/zeros_2:output:0)model/lstm/lstm_cell/ones_like_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_4ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,model/lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_4StridedSlice-model/lstm/lstm_cell/ReadVariableOp_4:value:03model/lstm/lstm_cell/strided_slice_4/stack:output:05model/lstm/lstm_cell/strided_slice_4/stack_1:output:05model/lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask«
model/lstm/lstm_cell/MatMul_12MatMulmodel/lstm/lstm_cell/mul_15:z:0-model/lstm/lstm_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
model/lstm/lstm_cell/add_5AddV2'model/lstm/lstm_cell/BiasAdd_4:output:0(model/lstm/lstm_cell/MatMul_12:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/lstm/lstm_cell/Sigmoid_3Sigmoidmodel/lstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_5ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_5StridedSlice-model/lstm/lstm_cell/ReadVariableOp_5:value:03model/lstm/lstm_cell/strided_slice_5/stack:output:05model/lstm/lstm_cell/strided_slice_5/stack_1:output:05model/lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask«
model/lstm/lstm_cell/MatMul_13MatMulmodel/lstm/lstm_cell/mul_16:z:0-model/lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
model/lstm/lstm_cell/add_6AddV2'model/lstm/lstm_cell/BiasAdd_5:output:0(model/lstm/lstm_cell/MatMul_13:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/lstm/lstm_cell/Sigmoid_4Sigmoidmodel/lstm/lstm_cell/add_6:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_19Mul"model/lstm/lstm_cell/Sigmoid_4:y:0model/lstm/zeros_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_6ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model/lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,model/lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_6StridedSlice-model/lstm/lstm_cell/ReadVariableOp_6:value:03model/lstm/lstm_cell/strided_slice_6/stack:output:05model/lstm/lstm_cell/strided_slice_6/stack_1:output:05model/lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask«
model/lstm/lstm_cell/MatMul_14MatMulmodel/lstm/lstm_cell/mul_17:z:0-model/lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
model/lstm/lstm_cell/add_7AddV2'model/lstm/lstm_cell/BiasAdd_6:output:0(model/lstm/lstm_cell/MatMul_14:product:0*
T0*(
_output_shapes
:’’’’’’’’’v
model/lstm/lstm_cell/Tanh_2Tanhmodel/lstm/lstm_cell/add_7:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_20Mul"model/lstm/lstm_cell/Sigmoid_3:y:0model/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/add_8AddV2model/lstm/lstm_cell/mul_19:z:0model/lstm/lstm_cell/mul_20:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%model/lstm/lstm_cell/ReadVariableOp_7ReadVariableOp,model_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0{
*model/lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,model/lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,model/lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$model/lstm/lstm_cell/strided_slice_7StridedSlice-model/lstm/lstm_cell/ReadVariableOp_7:value:03model/lstm/lstm_cell/strided_slice_7/stack:output:05model/lstm/lstm_cell/strided_slice_7/stack_1:output:05model/lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask«
model/lstm/lstm_cell/MatMul_15MatMulmodel/lstm/lstm_cell/mul_18:z:0-model/lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
model/lstm/lstm_cell/add_9AddV2'model/lstm/lstm_cell/BiasAdd_7:output:0(model/lstm/lstm_cell/MatMul_15:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/lstm/lstm_cell/Sigmoid_5Sigmoidmodel/lstm/lstm_cell/add_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’v
model/lstm/lstm_cell/Tanh_3Tanhmodel/lstm/lstm_cell/add_8:z:0*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/lstm_cell/mul_21Mul"model/lstm/lstm_cell/Sigmoid_5:y:0model/lstm/lstm_cell/Tanh_3:y:0*
T0*(
_output_shapes
:’’’’’’’’’y
(model/lstm/TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   i
'model/lstm/TensorArrayV2_4/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ę
model/lstm/TensorArrayV2_4TensorListReserve1model/lstm/TensorArrayV2_4/element_shape:output:00model/lstm/TensorArrayV2_4/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅS
model/lstm/time_1Const*
_output_shapes
: *
dtype0*
value	B : p
%model/lstm/while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’a
model/lstm/while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model/lstm/while_1While(model/lstm/while_1/loop_counter:output:0.model/lstm/while_1/maximum_iterations:output:0model/lstm/time_1:output:0#model/lstm/TensorArrayV2_4:handle:0model/lstm/zeros_2:output:0model/lstm/zeros_3:output:0#model/lstm/strided_slice_5:output:0Dmodel/lstm/TensorArrayUnstack_1/TensorListFromTensor:output_handle:02model_lstm_lstm_cell_split_readvariableop_resource4model_lstm_lstm_cell_split_1_readvariableop_resource,model_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
model_lstm_while_1_body_267453**
cond"R 
model_lstm_while_1_cond_267452*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
=model/lstm/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ž
/model/lstm/TensorArrayV2Stack_1/TensorListStackTensorListStackmodel/lstm/while_1:output:3Fmodel/lstm/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementss
 model/lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’l
"model/lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: l
"model/lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Į
model/lstm/strided_slice_7StridedSlice8model/lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0)model/lstm/strided_slice_7/stack:output:0+model/lstm/strided_slice_7/stack_1:output:0+model/lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskp
model/lstm/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          ŗ
model/lstm/transpose_3	Transpose8model/lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0$model/lstm/transpose_3/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’h
model/lstm/runtime_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    z
model/dropout/IdentityIdentity#model/lstm/strided_slice_7:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
model/dropout_1/IdentityIdentity#model/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ģ
model/concatenate/concatConcatV2model/dropout/Identity:output:0!model/dropout_1/Identity:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’@v
model/dropout_2/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’@
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0 
model/dense_1/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
model/dense_1/SigmoidSigmoidmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’h
IdentityIdentitymodel/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’ų
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp$^model/lstm/lstm_cell/ReadVariableOp&^model/lstm/lstm_cell/ReadVariableOp_1&^model/lstm/lstm_cell/ReadVariableOp_2&^model/lstm/lstm_cell/ReadVariableOp_3&^model/lstm/lstm_cell/ReadVariableOp_4&^model/lstm/lstm_cell/ReadVariableOp_5&^model/lstm/lstm_cell/ReadVariableOp_6&^model/lstm/lstm_cell/ReadVariableOp_7*^model/lstm/lstm_cell/split/ReadVariableOp,^model/lstm/lstm_cell/split_1/ReadVariableOp,^model/lstm/lstm_cell/split_2/ReadVariableOp,^model/lstm/lstm_cell/split_3/ReadVariableOp^model/lstm/while^model/lstm/while_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2N
%model/lstm/lstm_cell/ReadVariableOp_1%model/lstm/lstm_cell/ReadVariableOp_12N
%model/lstm/lstm_cell/ReadVariableOp_2%model/lstm/lstm_cell/ReadVariableOp_22N
%model/lstm/lstm_cell/ReadVariableOp_3%model/lstm/lstm_cell/ReadVariableOp_32N
%model/lstm/lstm_cell/ReadVariableOp_4%model/lstm/lstm_cell/ReadVariableOp_42N
%model/lstm/lstm_cell/ReadVariableOp_5%model/lstm/lstm_cell/ReadVariableOp_52N
%model/lstm/lstm_cell/ReadVariableOp_6%model/lstm/lstm_cell/ReadVariableOp_62N
%model/lstm/lstm_cell/ReadVariableOp_7%model/lstm/lstm_cell/ReadVariableOp_72J
#model/lstm/lstm_cell/ReadVariableOp#model/lstm/lstm_cell/ReadVariableOp2V
)model/lstm/lstm_cell/split/ReadVariableOp)model/lstm/lstm_cell/split/ReadVariableOp2Z
+model/lstm/lstm_cell/split_1/ReadVariableOp+model/lstm/lstm_cell/split_1/ReadVariableOp2Z
+model/lstm/lstm_cell/split_2/ReadVariableOp+model/lstm/lstm_cell/split_2/ReadVariableOp2Z
+model/lstm/lstm_cell/split_3/ReadVariableOp+model/lstm/lstm_cell/split_3/ReadVariableOp2(
model/lstm/while_1model/lstm/while_12$
model/lstm/whilemodel/lstm/while:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ės
ė
"__inference__traced_restore_272533
file_prefix0
assignvariableop_dense_kernel:	@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@-
assignvariableop_3_dense_1_bias:;
(assignvariableop_4_lstm_lstm_cell_kernel:	F
2assignvariableop_5_lstm_lstm_cell_recurrent_kernel:
5
&assignvariableop_6_lstm_lstm_cell_bias:	&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: B
/assignvariableop_9_adam_m_lstm_lstm_cell_kernel:	C
0assignvariableop_10_adam_v_lstm_lstm_cell_kernel:	N
:assignvariableop_11_adam_m_lstm_lstm_cell_recurrent_kernel:
N
:assignvariableop_12_adam_v_lstm_lstm_cell_recurrent_kernel:
=
.assignvariableop_13_adam_m_lstm_lstm_cell_bias:	=
.assignvariableop_14_adam_v_lstm_lstm_cell_bias:	:
'assignvariableop_15_adam_m_dense_kernel:	@:
'assignvariableop_16_adam_v_dense_kernel:	@3
%assignvariableop_17_adam_m_dense_bias:@3
%assignvariableop_18_adam_v_dense_bias:@;
)assignvariableop_19_adam_m_dense_1_kernel:@;
)assignvariableop_20_adam_v_dense_1_kernel:@5
'assignvariableop_21_adam_m_dense_1_bias:5
'assignvariableop_22_adam_v_dense_1_bias:%
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: 
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ļ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ø
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:æ
AssignVariableOp_4AssignVariableOp(assignvariableop_4_lstm_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_5AssignVariableOp2assignvariableop_5_lstm_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_6AssignVariableOp&assignvariableop_6_lstm_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:³
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ę
AssignVariableOp_9AssignVariableOp/assignvariableop_9_adam_m_lstm_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_10AssignVariableOp0assignvariableop_10_adam_v_lstm_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_11AssignVariableOp:assignvariableop_11_adam_m_lstm_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_12AssignVariableOp:assignvariableop_12_adam_v_lstm_lstm_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ē
AssignVariableOp_13AssignVariableOp.assignvariableop_13_adam_m_lstm_lstm_cell_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ē
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_v_lstm_lstm_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_m_dense_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_v_dense_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_m_dense_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_v_dense_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ā
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_m_dense_1_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ā
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_v_dense_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_m_dense_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_v_dense_1_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ”
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
å

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_270580

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ų
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_268914

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

µ
%__inference_lstm_layer_call_fn_270591
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_267870p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
	
Ć
while_cond_271106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_271106___redundant_placeholder04
0while_while_cond_271106___redundant_placeholder14
0while_while_cond_271106___redundant_placeholder24
0while_while_cond_271106___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ü
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_271914

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ
c
*__inference_dropout_1_layer_call_fn_271892

inputs
identity¢StatefulPartitionedCallĮ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268570p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ų
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_271974

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
®
X
,__inference_concatenate_layer_call_fn_271920
inputs_0
inputs_1
identityĄ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_268579a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:’’’’’’’’’:’’’’’’’’’:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0


Æ
&__inference_model_layer_call_fn_269027
input_1
input_2
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	@
	unknown_3:@
	unknown_4:@
	unknown_5:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_269010o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ń
a
(__inference_dropout_layer_call_fn_271865

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268556p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±

§
lstm_while_cond_269417&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_269417___redundant_placeholder0>
:lstm_while_lstm_while_cond_269417___redundant_placeholder1>
:lstm_while_lstm_while_cond_269417___redundant_placeholder2>
:lstm_while_lstm_while_cond_269417___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namelstm/while/maximum_iterations:O K

_output_shapes
: 
1
_user_specified_namelstm/while/loop_counter
 

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_268570

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ć
while_cond_267996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_267996___redundant_placeholder04
0while_while_cond_267996___redundant_placeholder14
0while_while_cond_267996___redundant_placeholder24
0while_while_cond_267996___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
'
Ź
A__inference_model_layer_call_and_return_conditional_losses_269010

inputs
inputs_1
lstm_268984:	
lstm_268986:	
lstm_268988:

dense_268998:	@
dense_269000:@ 
dense_1_269004:@
dense_1_269006:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢lstm/StatefulPartitionedCall_1Ą
reshape_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143ŗ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_268158
lstm/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_268984lstm_268986lstm_268988*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268880
lstm/StatefulPartitionedCall_1StatefulPartitionedCall reshape/PartitionedCall:output:0lstm_268984lstm_268986lstm_268988*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268880Ų
dropout/PartitionedCallPartitionedCall'lstm/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268896Ś
dropout_1/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268902ž
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_268579
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_268998dense_269000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_268592Ś
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268914
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_269004dense_1_269006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_268623w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Č
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm/StatefulPartitionedCall_1lstm/StatefulPartitionedCall_12<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
Å
model_lstm_while_1_body_2674536
2model_lstm_while_1_model_lstm_while_1_loop_counter<
8model_lstm_while_1_model_lstm_while_1_maximum_iterations"
model_lstm_while_1_placeholder$
 model_lstm_while_1_placeholder_1$
 model_lstm_while_1_placeholder_2$
 model_lstm_while_1_placeholder_33
/model_lstm_while_1_model_lstm_strided_slice_5_0q
mmodel_lstm_while_1_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_1_tensorlistfromtensor_0O
<model_lstm_while_1_lstm_cell_split_readvariableop_resource_0:	M
>model_lstm_while_1_lstm_cell_split_1_readvariableop_resource_0:	J
6model_lstm_while_1_lstm_cell_readvariableop_resource_0:

model_lstm_while_1_identity!
model_lstm_while_1_identity_1!
model_lstm_while_1_identity_2!
model_lstm_while_1_identity_3!
model_lstm_while_1_identity_4!
model_lstm_while_1_identity_51
-model_lstm_while_1_model_lstm_strided_slice_5o
kmodel_lstm_while_1_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_1_tensorlistfromtensorM
:model_lstm_while_1_lstm_cell_split_readvariableop_resource:	K
<model_lstm_while_1_lstm_cell_split_1_readvariableop_resource:	H
4model_lstm_while_1_lstm_cell_readvariableop_resource:
¢+model/lstm/while_1/lstm_cell/ReadVariableOp¢-model/lstm/while_1/lstm_cell/ReadVariableOp_1¢-model/lstm/while_1/lstm_cell/ReadVariableOp_2¢-model/lstm/while_1/lstm_cell/ReadVariableOp_3¢1model/lstm/while_1/lstm_cell/split/ReadVariableOp¢3model/lstm/while_1/lstm_cell/split_1/ReadVariableOp
Dmodel/lstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ē
6model/lstm/while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemmmodel_lstm_while_1_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_1_tensorlistfromtensor_0model_lstm_while_1_placeholderMmodel/lstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0§
,model/lstm/while_1/lstm_cell/ones_like/ShapeShape=model/lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻq
,model/lstm/while_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ī
&model/lstm/while_1/lstm_cell/ones_likeFill5model/lstm/while_1/lstm_cell/ones_like/Shape:output:05model/lstm/while_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
.model/lstm/while_1/lstm_cell/ones_like_1/ShapeShape model_lstm_while_1_placeholder_2*
T0*
_output_shapes
::ķĻs
.model/lstm/while_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
(model/lstm/while_1/lstm_cell/ones_like_1Fill7model/lstm/while_1/lstm_cell/ones_like_1/Shape:output:07model/lstm/while_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’É
 model/lstm/while_1/lstm_cell/mulMul=model/lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0/model/lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ė
"model/lstm/while_1/lstm_cell/mul_1Mul=model/lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0/model/lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ė
"model/lstm/while_1/lstm_cell/mul_2Mul=model/lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0/model/lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ė
"model/lstm/while_1/lstm_cell/mul_3Mul=model/lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0/model/lstm/while_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’n
,model/lstm/while_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Æ
1model/lstm/while_1/lstm_cell/split/ReadVariableOpReadVariableOp<model_lstm_while_1_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ł
"model/lstm/while_1/lstm_cell/splitSplit5model/lstm/while_1/lstm_cell/split/split_dim:output:09model/lstm/while_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split³
#model/lstm/while_1/lstm_cell/MatMulMatMul$model/lstm/while_1/lstm_cell/mul:z:0+model/lstm/while_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’·
%model/lstm/while_1/lstm_cell/MatMul_1MatMul&model/lstm/while_1/lstm_cell/mul_1:z:0+model/lstm/while_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’·
%model/lstm/while_1/lstm_cell/MatMul_2MatMul&model/lstm/while_1/lstm_cell/mul_2:z:0+model/lstm/while_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’·
%model/lstm/while_1/lstm_cell/MatMul_3MatMul&model/lstm/while_1/lstm_cell/mul_3:z:0+model/lstm/while_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’p
.model/lstm/while_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Æ
3model/lstm/while_1/lstm_cell/split_1/ReadVariableOpReadVariableOp>model_lstm_while_1_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ļ
$model/lstm/while_1/lstm_cell/split_1Split7model/lstm/while_1/lstm_cell/split_1/split_dim:output:0;model/lstm/while_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitĄ
$model/lstm/while_1/lstm_cell/BiasAddBiasAdd-model/lstm/while_1/lstm_cell/MatMul:product:0-model/lstm/while_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ä
&model/lstm/while_1/lstm_cell/BiasAdd_1BiasAdd/model/lstm/while_1/lstm_cell/MatMul_1:product:0-model/lstm/while_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’Ä
&model/lstm/while_1/lstm_cell/BiasAdd_2BiasAdd/model/lstm/while_1/lstm_cell/MatMul_2:product:0-model/lstm/while_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’Ä
&model/lstm/while_1/lstm_cell/BiasAdd_3BiasAdd/model/lstm/while_1/lstm_cell/MatMul_3:product:0-model/lstm/while_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’±
"model/lstm/while_1/lstm_cell/mul_4Mul model_lstm_while_1_placeholder_21model/lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’±
"model/lstm/while_1/lstm_cell/mul_5Mul model_lstm_while_1_placeholder_21model/lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’±
"model/lstm/while_1/lstm_cell/mul_6Mul model_lstm_while_1_placeholder_21model/lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’±
"model/lstm/while_1/lstm_cell/mul_7Mul model_lstm_while_1_placeholder_21model/lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¤
+model/lstm/while_1/lstm_cell/ReadVariableOpReadVariableOp6model_lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
0model/lstm/while_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
2model/lstm/while_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2model/lstm/while_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ž
*model/lstm/while_1/lstm_cell/strided_sliceStridedSlice3model/lstm/while_1/lstm_cell/ReadVariableOp:value:09model/lstm/while_1/lstm_cell/strided_slice/stack:output:0;model/lstm/while_1/lstm_cell/strided_slice/stack_1:output:0;model/lstm/while_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskæ
%model/lstm/while_1/lstm_cell/MatMul_4MatMul&model/lstm/while_1/lstm_cell/mul_4:z:03model/lstm/while_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’¼
 model/lstm/while_1/lstm_cell/addAddV2-model/lstm/while_1/lstm_cell/BiasAdd:output:0/model/lstm/while_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’
$model/lstm/while_1/lstm_cell/SigmoidSigmoid$model/lstm/while_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’¦
-model/lstm/while_1/lstm_cell/ReadVariableOp_1ReadVariableOp6model_lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
2model/lstm/while_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
4model/lstm/while_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
4model/lstm/while_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,model/lstm/while_1/lstm_cell/strided_slice_1StridedSlice5model/lstm/while_1/lstm_cell/ReadVariableOp_1:value:0;model/lstm/while_1/lstm_cell/strided_slice_1/stack:output:0=model/lstm/while_1/lstm_cell/strided_slice_1/stack_1:output:0=model/lstm/while_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĮ
%model/lstm/while_1/lstm_cell/MatMul_5MatMul&model/lstm/while_1/lstm_cell/mul_5:z:05model/lstm/while_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą
"model/lstm/while_1/lstm_cell/add_1AddV2/model/lstm/while_1/lstm_cell/BiasAdd_1:output:0/model/lstm/while_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’
&model/lstm/while_1/lstm_cell/Sigmoid_1Sigmoid&model/lstm/while_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’Ŗ
"model/lstm/while_1/lstm_cell/mul_8Mul*model/lstm/while_1/lstm_cell/Sigmoid_1:y:0 model_lstm_while_1_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’¦
-model/lstm/while_1/lstm_cell/ReadVariableOp_2ReadVariableOp6model_lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
2model/lstm/while_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
4model/lstm/while_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
4model/lstm/while_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,model/lstm/while_1/lstm_cell/strided_slice_2StridedSlice5model/lstm/while_1/lstm_cell/ReadVariableOp_2:value:0;model/lstm/while_1/lstm_cell/strided_slice_2/stack:output:0=model/lstm/while_1/lstm_cell/strided_slice_2/stack_1:output:0=model/lstm/while_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĮ
%model/lstm/while_1/lstm_cell/MatMul_6MatMul&model/lstm/while_1/lstm_cell/mul_6:z:05model/lstm/while_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą
"model/lstm/while_1/lstm_cell/add_2AddV2/model/lstm/while_1/lstm_cell/BiasAdd_2:output:0/model/lstm/while_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’
!model/lstm/while_1/lstm_cell/TanhTanh&model/lstm/while_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’­
"model/lstm/while_1/lstm_cell/mul_9Mul(model/lstm/while_1/lstm_cell/Sigmoid:y:0%model/lstm/while_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’®
"model/lstm/while_1/lstm_cell/add_3AddV2&model/lstm/while_1/lstm_cell/mul_8:z:0&model/lstm/while_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’¦
-model/lstm/while_1/lstm_cell/ReadVariableOp_3ReadVariableOp6model_lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
2model/lstm/while_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
4model/lstm/while_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
4model/lstm/while_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,model/lstm/while_1/lstm_cell/strided_slice_3StridedSlice5model/lstm/while_1/lstm_cell/ReadVariableOp_3:value:0;model/lstm/while_1/lstm_cell/strided_slice_3/stack:output:0=model/lstm/while_1/lstm_cell/strided_slice_3/stack_1:output:0=model/lstm/while_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĮ
%model/lstm/while_1/lstm_cell/MatMul_7MatMul&model/lstm/while_1/lstm_cell/mul_7:z:05model/lstm/while_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ą
"model/lstm/while_1/lstm_cell/add_4AddV2/model/lstm/while_1/lstm_cell/BiasAdd_3:output:0/model/lstm/while_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’
&model/lstm/while_1/lstm_cell/Sigmoid_2Sigmoid&model/lstm/while_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’
#model/lstm/while_1/lstm_cell/Tanh_1Tanh&model/lstm/while_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’²
#model/lstm/while_1/lstm_cell/mul_10Mul*model/lstm/while_1/lstm_cell/Sigmoid_2:y:0'model/lstm/while_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’
=model/lstm/while_1/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
7model/lstm/while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItem model_lstm_while_1_placeholder_1Fmodel/lstm/while_1/TensorArrayV2Write/TensorListSetItem/index:output:0'model/lstm/while_1/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅZ
model/lstm/while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm/while_1/addAddV2model_lstm_while_1_placeholder!model/lstm/while_1/add/y:output:0*
T0*
_output_shapes
: \
model/lstm/while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lstm/while_1/add_1AddV22model_lstm_while_1_model_lstm_while_1_loop_counter#model/lstm/while_1/add_1/y:output:0*
T0*
_output_shapes
: 
model/lstm/while_1/IdentityIdentitymodel/lstm/while_1/add_1:z:0^model/lstm/while_1/NoOp*
T0*
_output_shapes
: 
model/lstm/while_1/Identity_1Identity8model_lstm_while_1_model_lstm_while_1_maximum_iterations^model/lstm/while_1/NoOp*
T0*
_output_shapes
: 
model/lstm/while_1/Identity_2Identitymodel/lstm/while_1/add:z:0^model/lstm/while_1/NoOp*
T0*
_output_shapes
: ­
model/lstm/while_1/Identity_3IdentityGmodel/lstm/while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lstm/while_1/NoOp*
T0*
_output_shapes
: 
model/lstm/while_1/Identity_4Identity'model/lstm/while_1/lstm_cell/mul_10:z:0^model/lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/while_1/Identity_5Identity&model/lstm/while_1/lstm_cell/add_3:z:0^model/lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
model/lstm/while_1/NoOpNoOp,^model/lstm/while_1/lstm_cell/ReadVariableOp.^model/lstm/while_1/lstm_cell/ReadVariableOp_1.^model/lstm/while_1/lstm_cell/ReadVariableOp_2.^model/lstm/while_1/lstm_cell/ReadVariableOp_32^model/lstm/while_1/lstm_cell/split/ReadVariableOp4^model/lstm/while_1/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "G
model_lstm_while_1_identity_1&model/lstm/while_1/Identity_1:output:0"G
model_lstm_while_1_identity_2&model/lstm/while_1/Identity_2:output:0"G
model_lstm_while_1_identity_3&model/lstm/while_1/Identity_3:output:0"G
model_lstm_while_1_identity_4&model/lstm/while_1/Identity_4:output:0"G
model_lstm_while_1_identity_5&model/lstm/while_1/Identity_5:output:0"C
model_lstm_while_1_identity$model/lstm/while_1/Identity:output:0"n
4model_lstm_while_1_lstm_cell_readvariableop_resource6model_lstm_while_1_lstm_cell_readvariableop_resource_0"~
<model_lstm_while_1_lstm_cell_split_1_readvariableop_resource>model_lstm_while_1_lstm_cell_split_1_readvariableop_resource_0"z
:model_lstm_while_1_lstm_cell_split_readvariableop_resource<model_lstm_while_1_lstm_cell_split_readvariableop_resource_0"`
-model_lstm_while_1_model_lstm_strided_slice_5/model_lstm_while_1_model_lstm_strided_slice_5_0"Ü
kmodel_lstm_while_1_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_1_tensorlistfromtensormmodel_lstm_while_1_tensorarrayv2read_tensorlistgetitem_model_lstm_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2^
-model/lstm/while_1/lstm_cell/ReadVariableOp_1-model/lstm/while_1/lstm_cell/ReadVariableOp_12^
-model/lstm/while_1/lstm_cell/ReadVariableOp_2-model/lstm/while_1/lstm_cell/ReadVariableOp_22^
-model/lstm/while_1/lstm_cell/ReadVariableOp_3-model/lstm/while_1/lstm_cell/ReadVariableOp_32Z
+model/lstm/while_1/lstm_cell/ReadVariableOp+model/lstm/while_1/lstm_cell/ReadVariableOp2f
1model/lstm/while_1/lstm_cell/split/ReadVariableOp1model/lstm/while_1/lstm_cell/split/ReadVariableOp2j
3model/lstm/while_1/lstm_cell/split_1/ReadVariableOp3model/lstm/while_1/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :]Y

_output_shapes
: 
?
_user_specified_name'%model/lstm/while_1/maximum_iterations:W S

_output_shapes
: 
9
_user_specified_name!model/lstm/while_1/loop_counter
ņ	
­
$__inference_signature_wrapper_269185
input_1
input_2
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	@
	unknown_3:@
	unknown_4:@
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_267607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1


ō
C__inference_dense_1_layer_call_and_return_conditional_losses_271994

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ō
³
%__inference_lstm_layer_call_fn_270624

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268880p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©
Ź
@__inference_lstm_layer_call_and_return_conditional_losses_271860

inputs:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_271725*
condR
while_cond_271724*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š
ō
*__inference_lstm_cell_layer_call_fn_272011

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267785p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_1:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


b
C__inference_dropout_layer_call_and_return_conditional_losses_271882

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
łĮ
	
while_body_270798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¬
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ö
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ó
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter


d
E__inference_dropout_2_layer_call_and_return_conditional_losses_271969

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

F
*__inference_dropout_2_layer_call_fn_271957

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268914`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’@:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ł8
÷
@__inference_lstm_layer_call_and_return_conditional_losses_267870

inputs#
lstm_cell_267786:	
lstm_cell_267788:	$
lstm_cell_267790:

identity¢!lstm_cell/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_maské
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_267786lstm_cell_267788lstm_cell_267790*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267785n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_267786lstm_cell_267788lstm_cell_267790*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_267800*
condR
while_cond_267799*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
'
Ź
A__inference_model_layer_call_and_return_conditional_losses_268922
input_1
input_2
lstm_268881:	
lstm_268883:	
lstm_268885:

dense_268905:	@
dense_268907:@ 
dense_1_268916:@
dense_1_268918:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall¢lstm/StatefulPartitionedCall_1æ
reshape_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143»
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_268158
lstm/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_268881lstm_268883lstm_268885*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268880
lstm/StatefulPartitionedCall_1StatefulPartitionedCall reshape/PartitionedCall:output:0lstm_268881lstm_268883lstm_268885*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268880Ų
dropout/PartitionedCallPartitionedCall'lstm/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_268896Ś
dropout_1/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_268902ž
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_268579
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_268905dense_268907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_268592Ś
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_268914
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_268916dense_1_268918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_268623w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’Č
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm/StatefulPartitionedCall_1lstm/StatefulPartitionedCall_12<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
	
Ć
while_cond_268744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_268744___redundant_placeholder04
0while_while_cond_268744___redundant_placeholder14
0while_while_cond_268744___redundant_placeholder24
0while_while_cond_268744___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter


ó
A__inference_dense_layer_call_and_return_conditional_losses_268592

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ęĘ
Ź
@__inference_lstm_layer_call_and_return_conditional_losses_271615

inputs:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ 
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ä
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    »
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_271416*
condR
while_cond_271415*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
£
D
(__inference_reshape_layer_call_fn_270549

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_268158d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


ō
C__inference_dense_1_layer_call_and_return_conditional_losses_268623

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ś
a
C__inference_dropout_layer_call_and_return_conditional_losses_271887

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ęĘ
Ź
@__inference_lstm_layer_call_and_return_conditional_losses_268532

inputs:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ 
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ä
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    »
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_268333*
condR
while_cond_268332*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
§
F
*__inference_reshape_1_layer_call_fn_270567

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ć

_
C__inference_reshape_layer_call_and_return_conditional_losses_268158

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Å
model_lstm_while_1_cond_2674526
2model_lstm_while_1_model_lstm_while_1_loop_counter<
8model_lstm_while_1_model_lstm_while_1_maximum_iterations"
model_lstm_while_1_placeholder$
 model_lstm_while_1_placeholder_1$
 model_lstm_while_1_placeholder_2$
 model_lstm_while_1_placeholder_36
2model_lstm_while_1_less_model_lstm_strided_slice_5N
Jmodel_lstm_while_1_model_lstm_while_1_cond_267452___redundant_placeholder0N
Jmodel_lstm_while_1_model_lstm_while_1_cond_267452___redundant_placeholder1N
Jmodel_lstm_while_1_model_lstm_while_1_cond_267452___redundant_placeholder2N
Jmodel_lstm_while_1_model_lstm_while_1_cond_267452___redundant_placeholder3
model_lstm_while_1_identity

model/lstm/while_1/LessLessmodel_lstm_while_1_placeholder2model_lstm_while_1_less_model_lstm_strided_slice_5*
T0*
_output_shapes
: e
model/lstm/while_1/IdentityIdentitymodel/lstm/while_1/Less:z:0*
T0
*
_output_shapes
: "C
model_lstm_while_1_identity$model/lstm/while_1/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :]Y

_output_shapes
: 
?
_user_specified_name'%model/lstm/while_1/maximum_iterations:W S

_output_shapes
: 
9
_user_specified_name!model/lstm/while_1/loop_counter
	
Ć
while_cond_271724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_271724___redundant_placeholder04
0while_while_cond_271724___redundant_placeholder14
0while_while_cond_271724___redundant_placeholder24
0while_while_cond_271724___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
óD
©
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272256

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpS
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::ķĻT
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’W
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
::ķĻV
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:’’’’’’’’’S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’_
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ķ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_1:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
t
	
while_body_268745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’¢
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ł8
÷
@__inference_lstm_layer_call_and_return_conditional_losses_268067

inputs#
lstm_cell_267983:	
lstm_cell_267985:	$
lstm_cell_267987:

identity¢!lstm_cell/StatefulPartitionedCall¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_maské
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_267983lstm_cell_267985lstm_cell_267987*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267982n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_267983lstm_cell_267985lstm_cell_267987*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_267997*
condR
while_cond_267996*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
æ

&__inference_dense_layer_call_fn_271936

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_268592o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
§$
Š
while_body_267800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_267824_0:	'
while_lstm_cell_267826_0:	,
while_lstm_cell_267828_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_267824:	%
while_lstm_cell_267826:	*
while_lstm_cell_267828:
¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0§
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_267824_0while_lstm_cell_267826_0while_lstm_cell_267828_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267785r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_267824while_lstm_cell_267824_0"2
while_lstm_cell_267826while_lstm_cell_267826_0"2
while_lstm_cell_267828while_lstm_cell_267828_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
t
	
while_body_271107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’¢
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
å

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_268143

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
é

Ķ
lstm_while_1_cond_269783*
&lstm_while_1_lstm_while_1_loop_counter0
,lstm_while_1_lstm_while_1_maximum_iterations
lstm_while_1_placeholder
lstm_while_1_placeholder_1
lstm_while_1_placeholder_2
lstm_while_1_placeholder_3*
&lstm_while_1_less_lstm_strided_slice_5B
>lstm_while_1_lstm_while_1_cond_269783___redundant_placeholder0B
>lstm_while_1_lstm_while_1_cond_269783___redundant_placeholder1B
>lstm_while_1_lstm_while_1_cond_269783___redundant_placeholder2B
>lstm_while_1_lstm_while_1_cond_269783___redundant_placeholder3
lstm_while_1_identity
|
lstm/while_1/LessLesslstm_while_1_placeholder&lstm_while_1_less_lstm_strided_slice_5*
T0*
_output_shapes
: Y
lstm/while_1/IdentityIdentitylstm/while_1/Less:z:0*
T0
*
_output_shapes
: "7
lstm_while_1_identitylstm/while_1/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :WS

_output_shapes
: 
9
_user_specified_name!lstm/while_1/maximum_iterations:Q M

_output_shapes
: 
3
_user_specified_namelstm/while_1/loop_counter
 

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_271909

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
łĮ
	
while_body_268333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¬
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ö
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ó
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
t
	
while_body_271725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’¢
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’¤
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
łĮ
	
while_body_271416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
/while_lstm_cell_split_readvariableop_resource_0:	@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
-while_lstm_cell_split_readvariableop_resource:	>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻd
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¬
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ö
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ó
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ°
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’r
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
::ķĻf
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ±
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’©
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’­
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ņ
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ē
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ė
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter


±
&__inference_model_layer_call_fn_269205
inputs_0
inputs_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	@
	unknown_3:@
	unknown_4:@
	unknown_5:
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_268958o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0
	
Ć
while_cond_267799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_267799___redundant_placeholder04
0while_while_cond_267799___redundant_placeholder14
0while_while_cond_267799___redundant_placeholder24
0while_while_cond_267799___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
±

§
lstm_while_cond_270151&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_270151___redundant_placeholder0>
:lstm_while_lstm_while_cond_270151___redundant_placeholder1>
:lstm_while_lstm_while_cond_270151___redundant_placeholder2>
:lstm_while_lstm_while_cond_270151___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namelstm/while/maximum_iterations:O K

_output_shapes
: 
1
_user_specified_namelstm/while/loop_counter
	
Ć
while_cond_271415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_271415___redundant_placeholder04
0while_while_cond_271415___redundant_placeholder14
0while_while_cond_271415___redundant_placeholder24
0while_while_cond_271415___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
	
Ć
while_cond_270797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_270797___redundant_placeholder04
0while_while_cond_270797___redundant_placeholder14
0while_while_cond_270797___redundant_placeholder24
0while_while_cond_270797___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ę

model_lstm_while_cond_2672142
.model_lstm_while_model_lstm_while_loop_counter8
4model_lstm_while_model_lstm_while_maximum_iterations 
model_lstm_while_placeholder"
model_lstm_while_placeholder_1"
model_lstm_while_placeholder_2"
model_lstm_while_placeholder_34
0model_lstm_while_less_model_lstm_strided_slice_1J
Fmodel_lstm_while_model_lstm_while_cond_267214___redundant_placeholder0J
Fmodel_lstm_while_model_lstm_while_cond_267214___redundant_placeholder1J
Fmodel_lstm_while_model_lstm_while_cond_267214___redundant_placeholder2J
Fmodel_lstm_while_model_lstm_while_cond_267214___redundant_placeholder3
model_lstm_while_identity

model/lstm/while/LessLessmodel_lstm_while_placeholder0model_lstm_while_less_model_lstm_strided_slice_1*
T0*
_output_shapes
: a
model/lstm/while/IdentityIdentitymodel/lstm/while/Less:z:0*
T0
*
_output_shapes
: "?
model_lstm_while_identity"model/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :[W

_output_shapes
: 
=
_user_specified_name%#model/lstm/while/maximum_iterations:U Q

_output_shapes
: 
7
_user_specified_namemodel/lstm/while/loop_counter
Ēė
Ń
A__inference_model_layer_call_and_return_conditional_losses_270023
inputs_0
inputs_1?
,lstm_lstm_cell_split_readvariableop_resource:	=
.lstm_lstm_cell_split_1_readvariableop_resource:	:
&lstm_lstm_cell_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp¢lstm/lstm_cell/ReadVariableOp_1¢lstm/lstm_cell/ReadVariableOp_2¢lstm/lstm_cell/ReadVariableOp_3¢lstm/lstm_cell/ReadVariableOp_4¢lstm/lstm_cell/ReadVariableOp_5¢lstm/lstm_cell/ReadVariableOp_6¢lstm/lstm_cell/ReadVariableOp_7¢#lstm/lstm_cell/split/ReadVariableOp¢%lstm/lstm_cell/split_1/ReadVariableOp¢%lstm/lstm_cell/split_2/ReadVariableOp¢%lstm/lstm_cell/split_3/ReadVariableOp¢
lstm/while¢lstm/while_1U
reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
::ķĻg
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:~
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’S
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻe
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ł
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Æ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:z
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’b

lstm/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
::ķĻb
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm/transpose	Transposereshape_1/Reshape:output:0lstm/transpose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ć
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ļ
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅd
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masky
lstm/lstm_cell/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
::ķĻc
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’a
lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm/lstm_cell/dropout/MulMul!lstm/lstm_cell/ones_like:output:0%lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’{
lstm/lstm_cell/dropout/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻŖ
3lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform%lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0j
%lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ó
#lstm/lstm_cell/dropout/GreaterEqualGreaterEqual<lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:0.lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ļ
lstm/lstm_cell/dropout/SelectV2SelectV2'lstm/lstm_cell/dropout/GreaterEqual:z:0lstm/lstm_cell/dropout/Mul:z:0'lstm/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?”
lstm/lstm_cell/dropout_1/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
lstm/lstm_cell/dropout_1/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ®
5lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ł
%lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ×
!lstm/lstm_cell/dropout_1/SelectV2SelectV2)lstm/lstm_cell/dropout_1/GreaterEqual:z:0 lstm/lstm_cell/dropout_1/Mul:z:0)lstm/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?”
lstm/lstm_cell/dropout_2/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
lstm/lstm_cell/dropout_2/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ®
5lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ł
%lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ×
!lstm/lstm_cell/dropout_2/SelectV2SelectV2)lstm/lstm_cell/dropout_2/GreaterEqual:z:0 lstm/lstm_cell/dropout_2/Mul:z:0)lstm/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?”
lstm/lstm_cell/dropout_3/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’}
lstm/lstm_cell/dropout_3/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ®
5lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ł
%lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ×
!lstm/lstm_cell/dropout_3/SelectV2SelectV2)lstm/lstm_cell/dropout_3/GreaterEqual:z:0 lstm/lstm_cell/dropout_3/Mul:z:0)lstm/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
 lstm/lstm_cell/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
lstm/lstm_cell/ones_like_1Fill)lstm/lstm_cell/ones_like_1/Shape:output:0)lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
lstm/lstm_cell/dropout_4/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_4/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻÆ
5lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ś
%lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ų
!lstm/lstm_cell/dropout_4/SelectV2SelectV2)lstm/lstm_cell/dropout_4/GreaterEqual:z:0 lstm/lstm_cell/dropout_4/Mul:z:0)lstm/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
lstm/lstm_cell/dropout_5/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_5/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻÆ
5lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ś
%lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ų
!lstm/lstm_cell/dropout_5/SelectV2SelectV2)lstm/lstm_cell/dropout_5/GreaterEqual:z:0 lstm/lstm_cell/dropout_5/Mul:z:0)lstm/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
lstm/lstm_cell/dropout_6/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_6/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻÆ
5lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ś
%lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ų
!lstm/lstm_cell/dropout_6/SelectV2SelectV2)lstm/lstm_cell/dropout_6/GreaterEqual:z:0 lstm/lstm_cell/dropout_6/Mul:z:0)lstm/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
lstm/lstm_cell/dropout_7/MulMul#lstm/lstm_cell/ones_like_1:output:0'lstm/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_7/ShapeShape#lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻÆ
5lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ś
%lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ų
!lstm/lstm_cell/dropout_7/SelectV2SelectV2)lstm/lstm_cell/dropout_7/GreaterEqual:z:0 lstm/lstm_cell/dropout_7/Mul:z:0)lstm/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mulMullstm/strided_slice_2:output:0(lstm/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_1Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_2Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_3Mullstm/strided_slice_2:output:0*lstm/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ļ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/lstm_cell/MatMulMatMullstm/lstm_cell/mul:z:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_1MatMullstm/lstm_cell/mul_1:z:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_2MatMullstm/lstm_cell/mul_2:z:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_3MatMullstm/lstm_cell/mul_3:z:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_4Mullstm/zeros:output:0*lstm/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_5Mullstm/zeros:output:0*lstm/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_6Mullstm/zeros:output:0*lstm/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_7Mullstm/zeros:output:0*lstm/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ø
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul_4:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’l
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_5:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_8Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_6:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’h
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_9Mullstm/lstm_cell/Sigmoid:y:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_8:z:0lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_7:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_10Mullstm/lstm_cell/Sigmoid_2:y:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   c
!lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ō
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0*lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅK
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_269418*"
condR
lstm_while_cond_269417*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ę
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsm
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
lstm/Shape_2Shapereshape/Reshape:output:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_4StridedSlicelstm/Shape_2:output:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_2/packedPacklstm/strided_slice_4:output:0lstm/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_2Filllstm/zeros_2/packed:output:0lstm/zeros_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’X
lstm/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm/zeros_3/packedPacklstm/strided_slice_4:output:0lstm/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm/zeros_3Filllstm/zeros_3/packed:output:0lstm/zeros_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm/transpose_2	Transposereshape/Reshape:output:0lstm/transpose_2/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’^
lstm/Shape_3Shapelstm/transpose_2:y:0*
T0*
_output_shapes
::ķĻd
lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ō
lstm/strided_slice_5StridedSlicelstm/Shape_3:output:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’Ē
lstm/TensorArrayV2_3TensorListReserve+lstm/TensorArrayV2_3/element_shape:output:0lstm/strided_slice_5:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
<lstm/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   õ
.lstm/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm/transpose_2:y:0Elstm/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅd
lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm/strided_slice_6StridedSlicelstm/transpose_2:y:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask{
 lstm/lstm_cell/ones_like_2/ShapeShapelstm/strided_slice_6:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ŗ
lstm/lstm_cell/ones_like_2Fill)lstm/lstm_cell/ones_like_2/Shape:output:0)lstm/lstm_cell/ones_like_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
lstm/lstm_cell/dropout_8/MulMul#lstm/lstm_cell/ones_like_2:output:0'lstm/lstm_cell/dropout_8/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_8/ShapeShape#lstm/lstm_cell/ones_like_2:output:0*
T0*
_output_shapes
::ķĻ®
5lstm/lstm_cell/dropout_8/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_8/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_8/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ł
%lstm/lstm_cell/dropout_8/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_8/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_8/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ×
!lstm/lstm_cell/dropout_8/SelectV2SelectV2)lstm/lstm_cell/dropout_8/GreaterEqual:z:0 lstm/lstm_cell/dropout_8/Mul:z:0)lstm/lstm_cell/dropout_8/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
lstm/lstm_cell/dropout_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
lstm/lstm_cell/dropout_9/MulMul#lstm/lstm_cell/ones_like_2:output:0'lstm/lstm_cell/dropout_9/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_9/ShapeShape#lstm/lstm_cell/ones_like_2:output:0*
T0*
_output_shapes
::ķĻ®
5lstm/lstm_cell/dropout_9/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_9/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0l
'lstm/lstm_cell/dropout_9/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ł
%lstm/lstm_cell/dropout_9/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_9/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_9/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’e
 lstm/lstm_cell/dropout_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ×
!lstm/lstm_cell/dropout_9/SelectV2SelectV2)lstm/lstm_cell/dropout_9/GreaterEqual:z:0 lstm/lstm_cell/dropout_9/Mul:z:0)lstm/lstm_cell/dropout_9/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?„
lstm/lstm_cell/dropout_10/MulMul#lstm/lstm_cell/ones_like_2:output:0(lstm/lstm_cell/dropout_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_10/ShapeShape#lstm/lstm_cell/ones_like_2:output:0*
T0*
_output_shapes
::ķĻ°
6lstm/lstm_cell/dropout_10/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_10/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_10/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&lstm/lstm_cell/dropout_10/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_10/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_10/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"lstm/lstm_cell/dropout_10/SelectV2SelectV2*lstm/lstm_cell/dropout_10/GreaterEqual:z:0!lstm/lstm_cell/dropout_10/Mul:z:0*lstm/lstm_cell/dropout_10/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?„
lstm/lstm_cell/dropout_11/MulMul#lstm/lstm_cell/ones_like_2:output:0(lstm/lstm_cell/dropout_11/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_11/ShapeShape#lstm/lstm_cell/ones_like_2:output:0*
T0*
_output_shapes
::ķĻ°
6lstm/lstm_cell/dropout_11/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_11/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_11/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ü
&lstm/lstm_cell/dropout_11/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_11/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_11/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_11/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ū
"lstm/lstm_cell/dropout_11/SelectV2SelectV2*lstm/lstm_cell/dropout_11/GreaterEqual:z:0!lstm/lstm_cell/dropout_11/Mul:z:0*lstm/lstm_cell/dropout_11/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
 lstm/lstm_cell/ones_like_3/ShapeShapelstm/zeros_2:output:0*
T0*
_output_shapes
::ķĻe
 lstm/lstm_cell/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
lstm/lstm_cell/ones_like_3Fill)lstm/lstm_cell/ones_like_3/Shape:output:0)lstm/lstm_cell/ones_like_3/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_12/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
lstm/lstm_cell/dropout_12/MulMul#lstm/lstm_cell/ones_like_3:output:0(lstm/lstm_cell/dropout_12/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_12/ShapeShape#lstm/lstm_cell/ones_like_3:output:0*
T0*
_output_shapes
::ķĻ±
6lstm/lstm_cell/dropout_12/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_12/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_12/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&lstm/lstm_cell/dropout_12/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_12/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_12/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_12/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"lstm/lstm_cell/dropout_12/SelectV2SelectV2*lstm/lstm_cell/dropout_12/GreaterEqual:z:0!lstm/lstm_cell/dropout_12/Mul:z:0*lstm/lstm_cell/dropout_12/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_13/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
lstm/lstm_cell/dropout_13/MulMul#lstm/lstm_cell/ones_like_3:output:0(lstm/lstm_cell/dropout_13/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_13/ShapeShape#lstm/lstm_cell/ones_like_3:output:0*
T0*
_output_shapes
::ķĻ±
6lstm/lstm_cell/dropout_13/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_13/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_13/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&lstm/lstm_cell/dropout_13/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_13/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_13/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_13/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"lstm/lstm_cell/dropout_13/SelectV2SelectV2*lstm/lstm_cell/dropout_13/GreaterEqual:z:0!lstm/lstm_cell/dropout_13/Mul:z:0*lstm/lstm_cell/dropout_13/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
lstm/lstm_cell/dropout_14/MulMul#lstm/lstm_cell/ones_like_3:output:0(lstm/lstm_cell/dropout_14/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_14/ShapeShape#lstm/lstm_cell/ones_like_3:output:0*
T0*
_output_shapes
::ķĻ±
6lstm/lstm_cell/dropout_14/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_14/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_14/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&lstm/lstm_cell/dropout_14/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_14/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_14/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_14/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"lstm/lstm_cell/dropout_14/SelectV2SelectV2*lstm/lstm_cell/dropout_14/GreaterEqual:z:0!lstm/lstm_cell/dropout_14/Mul:z:0*lstm/lstm_cell/dropout_14/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
lstm/lstm_cell/dropout_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
lstm/lstm_cell/dropout_15/MulMul#lstm/lstm_cell/ones_like_3:output:0(lstm/lstm_cell/dropout_15/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/dropout_15/ShapeShape#lstm/lstm_cell/ones_like_3:output:0*
T0*
_output_shapes
::ķĻ±
6lstm/lstm_cell/dropout_15/random_uniform/RandomUniformRandomUniform(lstm/lstm_cell/dropout_15/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0m
(lstm/lstm_cell/dropout_15/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ż
&lstm/lstm_cell/dropout_15/GreaterEqualGreaterEqual?lstm/lstm_cell/dropout_15/random_uniform/RandomUniform:output:01lstm/lstm_cell/dropout_15/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
!lstm/lstm_cell/dropout_15/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"lstm/lstm_cell/dropout_15/SelectV2SelectV2*lstm/lstm_cell/dropout_15/GreaterEqual:z:0!lstm/lstm_cell/dropout_15/Mul:z:0*lstm/lstm_cell/dropout_15/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_11Mullstm/strided_slice_6:output:0*lstm/lstm_cell/dropout_8/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_12Mullstm/strided_slice_6:output:0*lstm/lstm_cell/dropout_9/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_13Mullstm/strided_slice_6:output:0+lstm/lstm_cell/dropout_10/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_14Mullstm/strided_slice_6:output:0+lstm/lstm_cell/dropout_11/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%lstm/lstm_cell/split_2/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Õ
lstm/lstm_cell/split_2Split)lstm/lstm_cell/split_2/split_dim:output:0-lstm/lstm_cell/split_2/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/lstm_cell/MatMul_8MatMullstm/lstm_cell/mul_11:z:0lstm/lstm_cell/split_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_9MatMullstm/lstm_cell/mul_12:z:0lstm/lstm_cell/split_2:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_10MatMullstm/lstm_cell/mul_13:z:0lstm/lstm_cell/split_2:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/MatMul_11MatMullstm/lstm_cell/mul_14:z:0lstm/lstm_cell/split_2:output:3*
T0*(
_output_shapes
:’’’’’’’’’b
 lstm/lstm_cell/split_3/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
%lstm/lstm_cell/split_3/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Å
lstm/lstm_cell/split_3Split)lstm/lstm_cell/split_3/split_dim:output:0-lstm/lstm_cell/split_3/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm/lstm_cell/BiasAdd_4BiasAdd!lstm/lstm_cell/MatMul_8:product:0lstm/lstm_cell/split_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_5BiasAdd!lstm/lstm_cell/MatMul_9:product:0lstm/lstm_cell/split_3:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_6BiasAdd"lstm/lstm_cell/MatMul_10:product:0lstm/lstm_cell/split_3:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/BiasAdd_7BiasAdd"lstm/lstm_cell/MatMul_11:product:0lstm/lstm_cell/split_3:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_15Mullstm/zeros_2:output:0+lstm/lstm_cell/dropout_12/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_16Mullstm/zeros_2:output:0+lstm/lstm_cell/dropout_13/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_17Mullstm/zeros_2:output:0+lstm/lstm_cell/dropout_14/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_18Mullstm/zeros_2:output:0+lstm/lstm_cell/dropout_15/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_4ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_4StridedSlice'lstm/lstm_cell/ReadVariableOp_4:value:0-lstm/lstm_cell/strided_slice_4/stack:output:0/lstm/lstm_cell/strided_slice_4/stack_1:output:0/lstm/lstm_cell/strided_slice_4/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_12MatMullstm/lstm_cell/mul_15:z:0'lstm/lstm_cell/strided_slice_4:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_5AddV2!lstm/lstm_cell/BiasAdd_4:output:0"lstm/lstm_cell/MatMul_12:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_3Sigmoidlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_5ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_5StridedSlice'lstm/lstm_cell/ReadVariableOp_5:value:0-lstm/lstm_cell/strided_slice_5/stack:output:0/lstm/lstm_cell/strided_slice_5/stack_1:output:0/lstm/lstm_cell/strided_slice_5/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_13MatMullstm/lstm_cell/mul_16:z:0'lstm/lstm_cell/strided_slice_5:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_5:output:0"lstm/lstm_cell/MatMul_13:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_4Sigmoidlstm/lstm_cell/add_6:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_19Mullstm/lstm_cell/Sigmoid_4:y:0lstm/zeros_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_6ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&lstm/lstm_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_6StridedSlice'lstm/lstm_cell/ReadVariableOp_6:value:0-lstm/lstm_cell/strided_slice_6/stack:output:0/lstm/lstm_cell/strided_slice_6/stack_1:output:0/lstm/lstm_cell/strided_slice_6/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_14MatMullstm/lstm_cell/mul_17:z:0'lstm/lstm_cell/strided_slice_6:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_7AddV2!lstm/lstm_cell/BiasAdd_6:output:0"lstm/lstm_cell/MatMul_14:product:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/add_7:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_20Mullstm/lstm_cell/Sigmoid_3:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_8AddV2lstm/lstm_cell/mul_19:z:0lstm/lstm_cell/mul_20:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/ReadVariableOp_7ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0u
$lstm/lstm_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"      w
&lstm/lstm_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ā
lstm/lstm_cell/strided_slice_7StridedSlice'lstm/lstm_cell/ReadVariableOp_7:value:0-lstm/lstm_cell/strided_slice_7/stack:output:0/lstm/lstm_cell/strided_slice_7/stack_1:output:0/lstm/lstm_cell/strided_slice_7/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm/lstm_cell/MatMul_15MatMullstm/lstm_cell/mul_18:z:0'lstm/lstm_cell/strided_slice_7:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/add_9AddV2!lstm/lstm_cell/BiasAdd_7:output:0"lstm/lstm_cell/MatMul_15:product:0*
T0*(
_output_shapes
:’’’’’’’’’p
lstm/lstm_cell/Sigmoid_5Sigmoidlstm/lstm_cell/add_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/add_8:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/lstm_cell/mul_21Mullstm/lstm_cell/Sigmoid_5:y:0lstm/lstm_cell/Tanh_3:y:0*
T0*(
_output_shapes
:’’’’’’’’’s
"lstm/TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   c
!lstm/TensorArrayV2_4/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ō
lstm/TensorArrayV2_4TensorListReserve+lstm/TensorArrayV2_4/element_shape:output:0*lstm/TensorArrayV2_4/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅM
lstm/time_1Const*
_output_shapes
: *
dtype0*
value	B : j
lstm/while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’[
lstm/while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : É
lstm/while_1While"lstm/while_1/loop_counter:output:0(lstm/while_1/maximum_iterations:output:0lstm/time_1:output:0lstm/TensorArrayV2_4:handle:0lstm/zeros_2:output:0lstm/zeros_3:output:0lstm/strided_slice_5:output:0>lstm/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_while_1_body_269784*$
condR
lstm_while_1_cond_269783*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
7lstm/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ģ
)lstm/TensorArrayV2Stack_1/TensorListStackTensorListStacklstm/while_1:output:3@lstm/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsm
lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’f
lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
lstm/strided_slice_7StridedSlice2lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maskj
lstm/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
lstm/transpose_3	Transpose2lstm/TensorArrayV2Stack_1/TensorListStack:tensor:0lstm/transpose_3/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’b
lstm/runtime_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMullstm/strided_slice_7:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’p
dropout/dropout/ShapeShapelstm/strided_slice_7:output:0*
T0*
_output_shapes
::ķĻ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>æ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    “
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_1/dropout/MulMullstm/strided_slice_3:output:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’r
dropout_1/dropout/ShapeShapelstm/strided_slice_3:output:0*
T0*
_output_shapes
::ķĻ”
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Å
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ¼
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ä
concatenate/concatConcatV2!dropout/dropout/SelectV2:output:0#dropout_1/dropout/SelectV2:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:’’’’’’’’’
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’@\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_2/dropout/MulMuldense/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’@m
dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::ķĻ 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’@*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’@^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    »
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_1/MatMulMatMul#dropout_2/dropout/SelectV2:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3 ^lstm/lstm_cell/ReadVariableOp_4 ^lstm/lstm_cell/ReadVariableOp_5 ^lstm/lstm_cell/ReadVariableOp_6 ^lstm/lstm_cell/ReadVariableOp_7$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp&^lstm/lstm_cell/split_2/ReadVariableOp&^lstm/lstm_cell/split_3/ReadVariableOp^lstm/while^lstm/while_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:’’’’’’’’’:’’’’’’’’’: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32B
lstm/lstm_cell/ReadVariableOp_4lstm/lstm_cell/ReadVariableOp_42B
lstm/lstm_cell/ReadVariableOp_5lstm/lstm_cell/ReadVariableOp_52B
lstm/lstm_cell/ReadVariableOp_6lstm/lstm_cell/ReadVariableOp_62B
lstm/lstm_cell/ReadVariableOp_7lstm/lstm_cell/ReadVariableOp_72>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2N
%lstm/lstm_cell/split_2/ReadVariableOp%lstm/lstm_cell/split_2/ReadVariableOp2N
%lstm/lstm_cell/split_3/ReadVariableOp%lstm/lstm_cell/split_3/ReadVariableOp2
lstm/while_1lstm/while_12

lstm/while
lstm/while:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0
Ć
§
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267785

inputs

states
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpS
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::ķĻT
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’]
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’U
ones_like_1/ShapeShapestates*
T0*
_output_shapes
::ķĻV
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
mulMulinputsdropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:’’’’’’’’’S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’d
mul_4Mulstatesdropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
mul_5Mulstatesdropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
mul_6Mulstatesdropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’d
mul_7Mulstatesdropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ķ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:PL
(
_output_shapes
:’’’’’’’’’
 
_user_specified_namestates:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
§$
Š
while_body_267997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_268021_0:	'
while_lstm_cell_268023_0:	,
while_lstm_cell_268025_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_268021:	%
while_lstm_cell_268023:	*
while_lstm_cell_268025:
¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0§
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_268021_0while_lstm_cell_268023_0while_lstm_cell_268025_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267982r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:00while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŅM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"2
while_lstm_cell_268021while_lstm_cell_268021_0"2
while_lstm_cell_268023while_lstm_cell_268023_0"2
while_lstm_cell_268025while_lstm_cell_268025_0"0
while_strided_slice_1while_strided_slice_1_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
é

Ķ
lstm_while_1_cond_270389*
&lstm_while_1_lstm_while_1_loop_counter0
,lstm_while_1_lstm_while_1_maximum_iterations
lstm_while_1_placeholder
lstm_while_1_placeholder_1
lstm_while_1_placeholder_2
lstm_while_1_placeholder_3*
&lstm_while_1_less_lstm_strided_slice_5B
>lstm_while_1_lstm_while_1_cond_270389___redundant_placeholder0B
>lstm_while_1_lstm_while_1_cond_270389___redundant_placeholder1B
>lstm_while_1_lstm_while_1_cond_270389___redundant_placeholder2B
>lstm_while_1_lstm_while_1_cond_270389___redundant_placeholder3
lstm_while_1_identity
|
lstm/while_1/LessLesslstm_while_1_placeholder&lstm_while_1_less_lstm_strided_slice_5*
T0*
_output_shapes
: Y
lstm/while_1/IdentityIdentitylstm/while_1/Less:z:0*
T0
*
_output_shapes
: "7
lstm_while_1_identitylstm/while_1/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :’’’’’’’’’:’’’’’’’’’: :::::

_output_shapes
::

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :WS

_output_shapes
: 
9
_user_specified_name!lstm/while_1/maximum_iterations:Q M

_output_shapes
: 
3
_user_specified_namelstm/while_1/loop_counter
Ü
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_268902

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ÕŲ
ó

lstm_while_1_body_269784*
&lstm_while_1_lstm_while_1_loop_counter0
,lstm_while_1_lstm_while_1_maximum_iterations
lstm_while_1_placeholder
lstm_while_1_placeholder_1
lstm_while_1_placeholder_2
lstm_while_1_placeholder_3'
#lstm_while_1_lstm_strided_slice_5_0e
alstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0I
6lstm_while_1_lstm_cell_split_readvariableop_resource_0:	G
8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0:	D
0lstm_while_1_lstm_cell_readvariableop_resource_0:

lstm_while_1_identity
lstm_while_1_identity_1
lstm_while_1_identity_2
lstm_while_1_identity_3
lstm_while_1_identity_4
lstm_while_1_identity_5%
!lstm_while_1_lstm_strided_slice_5c
_lstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensorG
4lstm_while_1_lstm_cell_split_readvariableop_resource:	E
6lstm_while_1_lstm_cell_split_1_readvariableop_resource:	B
.lstm_while_1_lstm_cell_readvariableop_resource:
¢%lstm/while_1/lstm_cell/ReadVariableOp¢'lstm/while_1/lstm_cell/ReadVariableOp_1¢'lstm/while_1/lstm_cell/ReadVariableOp_2¢'lstm/while_1/lstm_cell/ReadVariableOp_3¢+lstm/while_1/lstm_cell/split/ReadVariableOp¢-lstm/while_1/lstm_cell/split_1/ReadVariableOp
>lstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   É
0lstm/while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0lstm_while_1_placeholderGlstm/while_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
&lstm/while_1/lstm_cell/ones_like/ShapeShape7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻk
&lstm/while_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
 lstm/while_1/lstm_cell/ones_likeFill/lstm/while_1/lstm_cell/ones_like/Shape:output:0/lstm/while_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’i
$lstm/while_1/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?µ
"lstm/while_1/lstm_cell/dropout/MulMul)lstm/while_1/lstm_cell/ones_like:output:0-lstm/while_1/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
$lstm/while_1/lstm_cell/dropout/ShapeShape)lstm/while_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻŗ
;lstm/while_1/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform-lstm/while_1/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0r
-lstm/while_1/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ė
+lstm/while_1/lstm_cell/dropout/GreaterEqualGreaterEqualDlstm/while_1/lstm_cell/dropout/random_uniform/RandomUniform:output:06lstm/while_1/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ļ
'lstm/while_1/lstm_cell/dropout/SelectV2SelectV2/lstm/while_1/lstm_cell/dropout/GreaterEqual:z:0&lstm/while_1/lstm_cell/dropout/Mul:z:0/lstm/while_1/lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
$lstm/while_1/lstm_cell/dropout_1/MulMul)lstm/while_1/lstm_cell/ones_like:output:0/lstm/while_1/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_1/ShapeShape)lstm/while_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¾
=lstm/while_1/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ń
-lstm/while_1/lstm_cell/dropout_1/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_1/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ÷
)lstm/while_1/lstm_cell/dropout_1/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_1/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_1/Mul:z:01lstm/while_1/lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
$lstm/while_1/lstm_cell/dropout_2/MulMul)lstm/while_1/lstm_cell/ones_like:output:0/lstm/while_1/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_2/ShapeShape)lstm/while_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¾
=lstm/while_1/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ń
-lstm/while_1/lstm_cell/dropout_2/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_2/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ÷
)lstm/while_1/lstm_cell/dropout_2/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_2/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_2/Mul:z:01lstm/while_1/lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
$lstm/while_1/lstm_cell/dropout_3/MulMul)lstm/while_1/lstm_cell/ones_like:output:0/lstm/while_1/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_3/ShapeShape)lstm/while_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¾
=lstm/while_1/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ń
-lstm/while_1/lstm_cell/dropout_3/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_3/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ÷
)lstm/while_1/lstm_cell/dropout_3/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_3/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_3/Mul:z:01lstm/while_1/lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’
(lstm/while_1/lstm_cell/ones_like_1/ShapeShapelstm_while_1_placeholder_2*
T0*
_output_shapes
::ķĻm
(lstm/while_1/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ć
"lstm/while_1/lstm_cell/ones_like_1Fill1lstm/while_1/lstm_cell/ones_like_1/Shape:output:01lstm/while_1/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¼
$lstm/while_1/lstm_cell/dropout_4/MulMul+lstm/while_1/lstm_cell/ones_like_1:output:0/lstm/while_1/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_4/ShapeShape+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻæ
=lstm/while_1/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ņ
-lstm/while_1/lstm_cell/dropout_4/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_4/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ų
)lstm/while_1/lstm_cell/dropout_4/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_4/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_4/Mul:z:01lstm/while_1/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¼
$lstm/while_1/lstm_cell/dropout_5/MulMul+lstm/while_1/lstm_cell/ones_like_1:output:0/lstm/while_1/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_5/ShapeShape+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻæ
=lstm/while_1/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ņ
-lstm/while_1/lstm_cell/dropout_5/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_5/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ų
)lstm/while_1/lstm_cell/dropout_5/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_5/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_5/Mul:z:01lstm/while_1/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¼
$lstm/while_1/lstm_cell/dropout_6/MulMul+lstm/while_1/lstm_cell/ones_like_1:output:0/lstm/while_1/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_6/ShapeShape+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻæ
=lstm/while_1/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ņ
-lstm/while_1/lstm_cell/dropout_6/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_6/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ų
)lstm/while_1/lstm_cell/dropout_6/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_6/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_6/Mul:z:01lstm/while_1/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’k
&lstm/while_1/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¼
$lstm/while_1/lstm_cell/dropout_7/MulMul+lstm/while_1/lstm_cell/ones_like_1:output:0/lstm/while_1/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’
&lstm/while_1/lstm_cell/dropout_7/ShapeShape+lstm/while_1/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻæ
=lstm/while_1/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform/lstm/while_1/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0t
/lstm/while_1/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>ņ
-lstm/while_1/lstm_cell/dropout_7/GreaterEqualGreaterEqualFlstm/while_1/lstm_cell/dropout_7/random_uniform/RandomUniform:output:08lstm/while_1/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
(lstm/while_1/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ų
)lstm/while_1/lstm_cell/dropout_7/SelectV2SelectV21lstm/while_1/lstm_cell/dropout_7/GreaterEqual:z:0(lstm/while_1/lstm_cell/dropout_7/Mul:z:01lstm/while_1/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¾
lstm/while_1/lstm_cell/mulMul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:00lstm/while_1/lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ā
lstm/while_1/lstm_cell/mul_1Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:02lstm/while_1/lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ā
lstm/while_1/lstm_cell/mul_2Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:02lstm/while_1/lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’Ā
lstm/while_1/lstm_cell/mul_3Mul7lstm/while_1/TensorArrayV2Read/TensorListGetItem:item:02lstm/while_1/lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’h
&lstm/while_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :£
+lstm/while_1/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_while_1_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0ē
lstm/while_1/lstm_cell/splitSplit/lstm/while_1/lstm_cell/split/split_dim:output:03lstm/while_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split”
lstm/while_1/lstm_cell/MatMulMatMullstm/while_1/lstm_cell/mul:z:0%lstm/while_1/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_1MatMul lstm/while_1/lstm_cell/mul_1:z:0%lstm/while_1/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_2MatMul lstm/while_1/lstm_cell/mul_2:z:0%lstm/while_1/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’„
lstm/while_1/lstm_cell/MatMul_3MatMul lstm/while_1/lstm_cell/mul_3:z:0%lstm/while_1/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’j
(lstm/while_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : £
-lstm/while_1/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ż
lstm/while_1/lstm_cell/split_1Split1lstm/while_1/lstm_cell/split_1/split_dim:output:05lstm/while_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split®
lstm/while_1/lstm_cell/BiasAddBiasAdd'lstm/while_1/lstm_cell/MatMul:product:0'lstm/while_1/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_1BiasAdd)lstm/while_1/lstm_cell/MatMul_1:product:0'lstm/while_1/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_2BiasAdd)lstm/while_1/lstm_cell/MatMul_2:product:0'lstm/while_1/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’²
 lstm/while_1/lstm_cell/BiasAdd_3BiasAdd)lstm/while_1/lstm_cell/MatMul_3:product:0'lstm/while_1/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’¦
lstm/while_1/lstm_cell/mul_4Mullstm_while_1_placeholder_22lstm/while_1/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’¦
lstm/while_1/lstm_cell/mul_5Mullstm_while_1_placeholder_22lstm/while_1/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’¦
lstm/while_1/lstm_cell/mul_6Mullstm_while_1_placeholder_22lstm/while_1/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’¦
lstm/while_1/lstm_cell/mul_7Mullstm_while_1_placeholder_22lstm/while_1/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while_1/lstm_cell/ReadVariableOpReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while_1/lstm_cell/strided_sliceStridedSlice-lstm/while_1/lstm_cell/ReadVariableOp:value:03lstm/while_1/lstm_cell/strided_slice/stack:output:05lstm/while_1/lstm_cell/strided_slice/stack_1:output:05lstm/while_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask­
lstm/while_1/lstm_cell/MatMul_4MatMul lstm/while_1/lstm_cell/mul_4:z:0-lstm/while_1/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ŗ
lstm/while_1/lstm_cell/addAddV2'lstm/while_1/lstm_cell/BiasAdd:output:0)lstm/while_1/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while_1/lstm_cell/SigmoidSigmoidlstm/while_1/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_1StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_1:value:05lstm/while_1/lstm_cell/strided_slice_1/stack:output:07lstm/while_1/lstm_cell/strided_slice_1/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_5MatMul lstm/while_1/lstm_cell/mul_5:z:0/lstm/while_1/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_1AddV2)lstm/while_1/lstm_cell/BiasAdd_1:output:0)lstm/while_1/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’
 lstm/while_1/lstm_cell/Sigmoid_1Sigmoid lstm/while_1/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_8Mul$lstm/while_1/lstm_cell/Sigmoid_1:y:0lstm_while_1_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.lstm/while_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
.lstm/while_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_2StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_2:value:05lstm/while_1/lstm_cell/strided_slice_2/stack:output:07lstm/while_1/lstm_cell/strided_slice_2/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_6MatMul lstm/while_1/lstm_cell/mul_6:z:0/lstm/while_1/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_2AddV2)lstm/while_1/lstm_cell/BiasAdd_2:output:0)lstm/while_1/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’x
lstm/while_1/lstm_cell/TanhTanh lstm/while_1/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/mul_9Mul"lstm/while_1/lstm_cell/Sigmoid:y:0lstm/while_1/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/lstm_cell/add_3AddV2 lstm/while_1/lstm_cell/mul_8:z:0 lstm/while_1/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
'lstm/while_1/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_while_1_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,lstm/while_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
.lstm/while_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm/while_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ź
&lstm/while_1/lstm_cell/strided_slice_3StridedSlice/lstm/while_1/lstm_cell/ReadVariableOp_3:value:05lstm/while_1/lstm_cell/strided_slice_3/stack:output:07lstm/while_1/lstm_cell/strided_slice_3/stack_1:output:07lstm/while_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÆ
lstm/while_1/lstm_cell/MatMul_7MatMul lstm/while_1/lstm_cell/mul_7:z:0/lstm/while_1/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’®
lstm/while_1/lstm_cell/add_4AddV2)lstm/while_1/lstm_cell/BiasAdd_3:output:0)lstm/while_1/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’
 lstm/while_1/lstm_cell/Sigmoid_2Sigmoid lstm/while_1/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’z
lstm/while_1/lstm_cell/Tanh_1Tanh lstm/while_1/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’ 
lstm/while_1/lstm_cell/mul_10Mul$lstm/while_1/lstm_cell/Sigmoid_2:y:0!lstm/while_1/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’y
7lstm/while_1/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm/while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_1_placeholder_1@lstm/while_1/TensorArrayV2Write/TensorListSetItem/index:output:0!lstm/while_1/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅT
lstm/while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm/while_1/addAddV2lstm_while_1_placeholderlstm/while_1/add/y:output:0*
T0*
_output_shapes
: V
lstm/while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm/while_1/add_1AddV2&lstm_while_1_lstm_while_1_loop_counterlstm/while_1/add_1/y:output:0*
T0*
_output_shapes
: n
lstm/while_1/IdentityIdentitylstm/while_1/add_1:z:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_1Identity,lstm_while_1_lstm_while_1_maximum_iterations^lstm/while_1/NoOp*
T0*
_output_shapes
: n
lstm/while_1/Identity_2Identitylstm/while_1/add:z:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_3IdentityAlstm/while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while_1/NoOp*
T0*
_output_shapes
: 
lstm/while_1/Identity_4Identity!lstm/while_1/lstm_cell/mul_10:z:0^lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while_1/Identity_5Identity lstm/while_1/lstm_cell/add_3:z:0^lstm/while_1/NoOp*
T0*(
_output_shapes
:’’’’’’’’’×
lstm/while_1/NoOpNoOp&^lstm/while_1/lstm_cell/ReadVariableOp(^lstm/while_1/lstm_cell/ReadVariableOp_1(^lstm/while_1/lstm_cell/ReadVariableOp_2(^lstm/while_1/lstm_cell/ReadVariableOp_3,^lstm/while_1/lstm_cell/split/ReadVariableOp.^lstm/while_1/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_while_1_identity_1 lstm/while_1/Identity_1:output:0";
lstm_while_1_identity_2 lstm/while_1/Identity_2:output:0";
lstm_while_1_identity_3 lstm/while_1/Identity_3:output:0";
lstm_while_1_identity_4 lstm/while_1/Identity_4:output:0";
lstm_while_1_identity_5 lstm/while_1/Identity_5:output:0"7
lstm_while_1_identitylstm/while_1/Identity:output:0"b
.lstm_while_1_lstm_cell_readvariableop_resource0lstm_while_1_lstm_cell_readvariableop_resource_0"r
6lstm_while_1_lstm_cell_split_1_readvariableop_resource8lstm_while_1_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_while_1_lstm_cell_split_readvariableop_resource6lstm_while_1_lstm_cell_split_readvariableop_resource_0"H
!lstm_while_1_lstm_strided_slice_5#lstm_while_1_lstm_strided_slice_5_0"Ä
_lstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensoralstm_while_1_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2R
'lstm/while_1/lstm_cell/ReadVariableOp_1'lstm/while_1/lstm_cell/ReadVariableOp_12R
'lstm/while_1/lstm_cell/ReadVariableOp_2'lstm/while_1/lstm_cell/ReadVariableOp_22R
'lstm/while_1/lstm_cell/ReadVariableOp_3'lstm/while_1/lstm_cell/ReadVariableOp_32N
%lstm/while_1/lstm_cell/ReadVariableOp%lstm/while_1/lstm_cell/ReadVariableOp2Z
+lstm/while_1/lstm_cell/split/ReadVariableOp+lstm/while_1/lstm_cell/split/ReadVariableOp2^
-lstm/while_1/lstm_cell/split_1/ReadVariableOp-lstm/while_1/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :WS

_output_shapes
: 
9
_user_specified_name!lstm/while_1/maximum_iterations:Q M

_output_shapes
: 
3
_user_specified_namelstm/while_1/loop_counter


ó
A__inference_dense_layer_call_and_return_conditional_losses_271947

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ō
³
%__inference_lstm_layer_call_fn_270613

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_268532p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó
©
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272174

inputs
states_0
states_10
split_readvariableop_resource:	.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpS
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::ķĻT
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’]
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’_
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
::ķĻ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’W
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
::ķĻV
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’a
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
::ķĻ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’_
mulMulinputsdropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’c
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	*
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:’’’’’’’’’`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:’’’’’’’’’S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’f
mul_4Mulstates_0dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
mul_5Mulstates_0dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
mul_6Mulstates_0dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’f
mul_7Mulstates_0dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ķ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32 
ReadVariableOpReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_1:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©
Ź
@__inference_lstm_layer_call_and_return_conditional_losses_268880

inputs:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_268745*
condR
while_cond_268744*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē
Ģ
@__inference_lstm_layer_call_and_return_conditional_losses_270997
inputs_0:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’q
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ 
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ä
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    »
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’s
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
::ķĻ¤
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ź
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ć
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
::ķĻ„
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>Ė
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_270798*
condR
while_cond_270797*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
ć

_
C__inference_reshape_layer_call_and_return_conditional_losses_270562

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ģ
Ģ
@__inference_lstm_layer_call_and_return_conditional_losses_271242
inputs_0:
'lstm_cell_split_readvariableop_resource:	8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ķĻ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ķĻ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ū
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’“
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ą
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_masko
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
::ķĻ^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’g
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
::ķĻ`
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes
:	*
dtype0Ą
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¶
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŅF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_271107*
condR
while_cond_271106*M
output_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’’’’’’’’’’: : : 28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_324
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp2@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs_0
Ą

(__inference_dense_1_layer_call_fn_271983

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallŲ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_268623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
š
ō
*__inference_lstm_cell_layer_call_fn_272028

inputs
states_0
states_1
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1

identity_2¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_267982p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:’’’’’’’’’r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_1:RN
(
_output_shapes
:’’’’’’’’’
"
_user_specified_name
states_0:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

±

lstm_while_body_270152&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0G
4lstm_while_lstm_cell_split_readvariableop_resource_0:	E
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	B
.lstm_while_lstm_cell_readvariableop_resource_0:

lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorE
2lstm_while_lstm_cell_split_readvariableop_resource:	C
4lstm_while_lstm_cell_split_1_readvariableop_resource:	@
,lstm_while_lstm_cell_readvariableop_resource:
¢#lstm/while/lstm_cell/ReadVariableOp¢%lstm/while/lstm_cell/ReadVariableOp_1¢%lstm/while/lstm_cell/ReadVariableOp_2¢%lstm/while/lstm_cell/ReadVariableOp_3¢)lstm/while/lstm_cell/split/ReadVariableOp¢+lstm/while/lstm_cell/split_1/ReadVariableOp
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   æ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype0
$lstm/while/lstm_cell/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
::ķĻi
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’|
&lstm/while/lstm_cell/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
::ķĻk
&lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?½
 lstm/while/lstm_cell/ones_like_1Fill/lstm/while/lstm_cell/ones_like_1/Shape:output:0/lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’±
lstm/while/lstm_cell/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’³
lstm/while/lstm_cell/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’³
lstm/while/lstm_cell/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’³
lstm/while/lstm_cell/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:’’’’’’’’’f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0*
_output_shapes
:	*
dtype0į
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split
lstm/while/lstm_cell/MatMulMatMullstm/while/lstm_cell/mul:z:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_1MatMullstm/while/lstm_cell/mul_1:z:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_2MatMullstm/while/lstm_cell/mul_2:z:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/MatMul_3MatMullstm/while/lstm_cell/mul_3:z:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:’’’’’’’’’h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0×
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitØ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’¬
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_4Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_5Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_6Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_7Mullstm_while_placeholder_2)lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask§
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul_4:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:’’’’’’’’’¤
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:’’’’’’’’’x
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_5:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_8Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_6:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:’’’’’’’’’t
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_9Mul lstm/while/lstm_cell/Sigmoid:y:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_8:z:0lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:’’’’’’’’’
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ą
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask©
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_7:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:’’’’’’’’’Ø
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:’’’’’’’’’|
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:’’’’’’’’’v
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/lstm_cell/mul_10Mul"lstm/while/lstm_cell/Sigmoid_2:y:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:’’’’’’’’’w
5lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ’
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1>lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŅR
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: 
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_10:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:’’’’’’’’’É
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"3
lstm_while_identitylstm/while/Identity:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :’’’’’’’’’:’’’’’’’’’: : : : : 2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:’’’’’’’’’:.*
(
_output_shapes
:’’’’’’’’’:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namelstm/while/maximum_iterations:O K

_output_shapes
: 
1
_user_specified_namelstm/while/loop_counter
Ś
a
C__inference_dropout_layer_call_and_return_conditional_losses_268896

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"ó
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ē
serving_defaultÓ
;
input_10
serving_default_input_1:0’’’’’’’’’
;
input_20
serving_default_input_2:0’’’’’’’’’;
dense_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Ķ«
Į
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer_with_weights-2
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
„
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
„
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
Ś
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec"
_tf_keras_rnn_layer
¼
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator"
_tf_keras_layer
¼
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator"
_tf_keras_layer
„
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
»
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias"
_tf_keras_layer
¼
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator"
_tf_keras_layer
»
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias"
_tf_keras_layer
Q
U0
V1
W2
D3
E4
S5
T6"
trackable_list_wrapper
Q
U0
V1
W2
D3
E4
S5
T6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ć
]trace_0
^trace_1
_trace_2
`trace_32Ų
&__inference_model_layer_call_fn_268975
&__inference_model_layer_call_fn_269027
&__inference_model_layer_call_fn_269205
&__inference_model_layer_call_fn_269225µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z]trace_0z^trace_1z_trace_2z`trace_3
Æ
atrace_0
btrace_1
ctrace_2
dtrace_32Ä
A__inference_model_layer_call_and_return_conditional_losses_268630
A__inference_model_layer_call_and_return_conditional_losses_268922
A__inference_model_layer_call_and_return_conditional_losses_270023
A__inference_model_layer_call_and_return_conditional_losses_270544µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zatrace_0zbtrace_1zctrace_2zdtrace_3
ÕBŅ
!__inference__wrapped_model_267607input_1input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 

e
_variables
f_iterations
g_learning_rate
h_index_dict
i
_momentums
j_velocities
k_update_step_xla"
experimentalOptimizer
,
lserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ā
rtrace_02Å
(__inference_reshape_layer_call_fn_270549
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zrtrace_0
ż
strace_02ą
C__inference_reshape_layer_call_and_return_conditional_losses_270562
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zstrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ä
ytrace_02Ē
*__inference_reshape_1_layer_call_fn_270567
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zytrace_0
’
ztrace_02ā
E__inference_reshape_1_layer_call_and_return_conditional_losses_270580
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zztrace_0
5
U0
V1
W2"
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
 "
trackable_list_wrapper
ŗ

{states
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ü
trace_0
trace_1
trace_2
trace_32é
%__inference_lstm_layer_call_fn_270591
%__inference_lstm_layer_call_fn_270602
%__inference_lstm_layer_call_fn_270613
%__inference_lstm_layer_call_fn_270624Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Č
trace_0
trace_1
trace_2
trace_32Õ
@__inference_lstm_layer_call_and_return_conditional_losses_270997
@__inference_lstm_layer_call_and_return_conditional_losses_271242
@__inference_lstm_layer_call_and_return_conditional_losses_271615
@__inference_lstm_layer_call_and_return_conditional_losses_271860Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

Ukernel
Vrecurrent_kernel
Wbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
»
trace_0
trace_12
(__inference_dropout_layer_call_fn_271865
(__inference_dropout_layer_call_fn_271870©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1
ń
trace_0
trace_12¶
C__inference_dropout_layer_call_and_return_conditional_losses_271882
C__inference_dropout_layer_call_and_return_conditional_losses_271887©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
æ
trace_0
 trace_12
*__inference_dropout_1_layer_call_fn_271892
*__inference_dropout_1_layer_call_fn_271897©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ztrace_0z trace_1
õ
”trace_0
¢trace_12ŗ
E__inference_dropout_1_layer_call_and_return_conditional_losses_271909
E__inference_dropout_1_layer_call_and_return_conditional_losses_271914©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z”trace_0z¢trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
„metrics
 ¦layer_regularization_losses
§layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
č
Øtrace_02É
,__inference_concatenate_layer_call_fn_271920
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zØtrace_0

©trace_02ä
G__inference_concatenate_layer_call_and_return_conditional_losses_271927
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z©trace_0
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ŗnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ā
Ætrace_02Ć
&__inference_dense_layer_call_fn_271936
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÆtrace_0
ż
°trace_02Ž
A__inference_dense_layer_call_and_return_conditional_losses_271947
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z°trace_0
:	@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 “layer_regularization_losses
µlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
æ
¶trace_0
·trace_12
*__inference_dropout_2_layer_call_fn_271952
*__inference_dropout_2_layer_call_fn_271957©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z¶trace_0z·trace_1
õ
øtrace_0
¹trace_12ŗ
E__inference_dropout_2_layer_call_and_return_conditional_losses_271969
E__inference_dropout_2_layer_call_and_return_conditional_losses_271974©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zøtrace_0z¹trace_1
"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ä
ætrace_02Å
(__inference_dense_1_layer_call_fn_271983
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zætrace_0
’
Ątrace_02ą
C__inference_dense_1_layer_call_and_return_conditional_losses_271994
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zĄtrace_0
 :@2dense_1/kernel
:2dense_1/bias
(:&	2lstm/lstm_cell/kernel
3:1
2lstm/lstm_cell/recurrent_kernel
": 2lstm/lstm_cell/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
Į0
Ā1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bō
&__inference_model_layer_call_fn_268975input_1input_2"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
÷Bō
&__inference_model_layer_call_fn_269027input_1input_2"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
&__inference_model_layer_call_fn_269205inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
łBö
&__inference_model_layer_call_fn_269225inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_268630input_1input_2"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_268922input_1input_2"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_270023inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
A__inference_model_layer_call_and_return_conditional_losses_270544inputs_0inputs_1"µ
®²Ŗ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 

f0
Ć1
Ä2
Å3
Ę4
Ē5
Č6
É7
Ź8
Ė9
Ģ10
Ķ11
Ī12
Ļ13
Š14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
Ć0
Å1
Ē2
É3
Ė4
Ķ5
Ļ6"
trackable_list_wrapper
X
Ä0
Ę1
Č2
Ź3
Ģ4
Ī5
Š6"
trackable_list_wrapper
µ2²Æ
¦²¢
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 0
ŅBĻ
$__inference_signature_wrapper_269185input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŅBĻ
(__inference_reshape_layer_call_fn_270549inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķBź
C__inference_reshape_layer_call_and_return_conditional_losses_270562inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŌBŃ
*__inference_reshape_1_layer_call_fn_270567inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļBģ
E__inference_reshape_1_layer_call_and_return_conditional_losses_270580inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_lstm_layer_call_fn_270591inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
%__inference_lstm_layer_call_fn_270602inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bž
%__inference_lstm_layer_call_fn_270613inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bž
%__inference_lstm_layer_call_fn_270624inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
@__inference_lstm_layer_call_and_return_conditional_losses_270997inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
@__inference_lstm_layer_call_and_return_conditional_losses_271242inputs_0"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
@__inference_lstm_layer_call_and_return_conditional_losses_271615inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
@__inference_lstm_layer_call_and_return_conditional_losses_271860inputs"Ź
Ć²æ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults¢

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
5
U0
V1
W2"
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ńnon_trainable_variables
Ņlayers
Ómetrics
 Ōlayer_regularization_losses
Õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
É
Ötrace_0
×trace_12
*__inference_lstm_cell_layer_call_fn_272011
*__inference_lstm_cell_layer_call_fn_272028³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zÖtrace_0z×trace_1
’
Ųtrace_0
Łtrace_12Ä
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272174
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272256³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zŲtrace_0zŁtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ćBą
(__inference_dropout_layer_call_fn_271865inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ćBą
(__inference_dropout_layer_call_fn_271870inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
C__inference_dropout_layer_call_and_return_conditional_losses_271882inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
žBū
C__inference_dropout_layer_call_and_return_conditional_losses_271887inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
åBā
*__inference_dropout_1_layer_call_fn_271892inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
åBā
*__inference_dropout_1_layer_call_fn_271897inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
E__inference_dropout_1_layer_call_and_return_conditional_losses_271909inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
E__inference_dropout_1_layer_call_and_return_conditional_losses_271914inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
āBß
,__inference_concatenate_layer_call_fn_271920inputs_0inputs_1"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
żBś
G__inference_concatenate_layer_call_and_return_conditional_losses_271927inputs_0inputs_1"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŠBĶ
&__inference_dense_layer_call_fn_271936inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ėBč
A__inference_dense_layer_call_and_return_conditional_losses_271947inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
åBā
*__inference_dropout_2_layer_call_fn_271952inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
åBā
*__inference_dropout_2_layer_call_fn_271957inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
E__inference_dropout_2_layer_call_and_return_conditional_losses_271969inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Bż
E__inference_dropout_2_layer_call_and_return_conditional_losses_271974inputs"©
¢²
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŅBĻ
(__inference_dense_1_layer_call_fn_271983inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķBź
C__inference_dense_1_layer_call_and_return_conditional_losses_271994inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
R
Ś	variables
Ū	keras_api

Ütotal

Żcount"
_tf_keras_metric
c
Ž	variables
ß	keras_api

ątotal

įcount
ā
_fn_kwargs"
_tf_keras_metric
-:+	2Adam/m/lstm/lstm_cell/kernel
-:+	2Adam/v/lstm/lstm_cell/kernel
8:6
2&Adam/m/lstm/lstm_cell/recurrent_kernel
8:6
2&Adam/v/lstm/lstm_cell/recurrent_kernel
':%2Adam/m/lstm/lstm_cell/bias
':%2Adam/v/lstm/lstm_cell/bias
$:"	@2Adam/m/dense/kernel
$:"	@2Adam/v/dense/kernel
:@2Adam/m/dense/bias
:@2Adam/v/dense/bias
%:#@2Adam/m/dense_1/kernel
%:#@2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_lstm_cell_layer_call_fn_272011inputsstates_0states_1"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
*__inference_lstm_cell_layer_call_fn_272028inputsstates_0states_1"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272174inputsstates_0states_1"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272256inputsstates_0states_1"³
¬²Ø
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
0
Ü0
Ż1"
trackable_list_wrapper
.
Ś	variables"
_generic_user_object
:  (2total
:  (2count
0
ą0
į1"
trackable_list_wrapper
.
Ž	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper¼
!__inference__wrapped_model_267607UWVDESTX¢U
N¢K
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’
Ŗ "1Ŗ.
,
dense_1!
dense_1’’’’’’’’’Ł
G__inference_concatenate_layer_call_and_return_conditional_losses_271927\¢Y
R¢O
MJ
# 
inputs_0’’’’’’’’’
# 
inputs_1’’’’’’’’’
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 ³
,__inference_concatenate_layer_call_fn_271920\¢Y
R¢O
MJ
# 
inputs_0’’’’’’’’’
# 
inputs_1’’’’’’’’’
Ŗ ""
unknown’’’’’’’’’Ŗ
C__inference_dense_1_layer_call_and_return_conditional_losses_271994cST/¢,
%¢"
 
inputs’’’’’’’’’@
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 
(__inference_dense_1_layer_call_fn_271983XST/¢,
%¢"
 
inputs’’’’’’’’’@
Ŗ "!
unknown’’’’’’’’’©
A__inference_dense_layer_call_and_return_conditional_losses_271947dDE0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ ",¢)
"
tensor_0’’’’’’’’’@
 
&__inference_dense_layer_call_fn_271936YDE0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "!
unknown’’’’’’’’’@®
E__inference_dropout_1_layer_call_and_return_conditional_losses_271909e4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 ®
E__inference_dropout_1_layer_call_and_return_conditional_losses_271914e4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 
*__inference_dropout_1_layer_call_fn_271892Z4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ ""
unknown’’’’’’’’’
*__inference_dropout_1_layer_call_fn_271897Z4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ ""
unknown’’’’’’’’’¬
E__inference_dropout_2_layer_call_and_return_conditional_losses_271969c3¢0
)¢&
 
inputs’’’’’’’’’@
p
Ŗ ",¢)
"
tensor_0’’’’’’’’’@
 ¬
E__inference_dropout_2_layer_call_and_return_conditional_losses_271974c3¢0
)¢&
 
inputs’’’’’’’’’@
p 
Ŗ ",¢)
"
tensor_0’’’’’’’’’@
 
*__inference_dropout_2_layer_call_fn_271952X3¢0
)¢&
 
inputs’’’’’’’’’@
p
Ŗ "!
unknown’’’’’’’’’@
*__inference_dropout_2_layer_call_fn_271957X3¢0
)¢&
 
inputs’’’’’’’’’@
p 
Ŗ "!
unknown’’’’’’’’’@¬
C__inference_dropout_layer_call_and_return_conditional_losses_271882e4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 ¬
C__inference_dropout_layer_call_and_return_conditional_losses_271887e4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 
(__inference_dropout_layer_call_fn_271865Z4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ ""
unknown’’’’’’’’’
(__inference_dropout_layer_call_fn_271870Z4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ ""
unknown’’’’’’’’’ä
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272174UWV¢
x¢u
 
inputs’’’’’’’’’
M¢J
# 
states_0’’’’’’’’’
# 
states_1’’’’’’’’’
p
Ŗ "¢
¢~
%"

tensor_0_0’’’’’’’’’
UR
'$
tensor_0_1_0’’’’’’’’’
'$
tensor_0_1_1’’’’’’’’’
 ä
E__inference_lstm_cell_layer_call_and_return_conditional_losses_272256UWV¢
x¢u
 
inputs’’’’’’’’’
M¢J
# 
states_0’’’’’’’’’
# 
states_1’’’’’’’’’
p 
Ŗ "¢
¢~
%"

tensor_0_0’’’’’’’’’
UR
'$
tensor_0_1_0’’’’’’’’’
'$
tensor_0_1_1’’’’’’’’’
 ¶
*__inference_lstm_cell_layer_call_fn_272011UWV¢
x¢u
 
inputs’’’’’’’’’
M¢J
# 
states_0’’’’’’’’’
# 
states_1’’’’’’’’’
p
Ŗ "{¢x
# 
tensor_0’’’’’’’’’
QN
%"

tensor_1_0’’’’’’’’’
%"

tensor_1_1’’’’’’’’’¶
*__inference_lstm_cell_layer_call_fn_272028UWV¢
x¢u
 
inputs’’’’’’’’’
M¢J
# 
states_0’’’’’’’’’
# 
states_1’’’’’’’’’
p 
Ŗ "{¢x
# 
tensor_0’’’’’’’’’
QN
%"

tensor_1_0’’’’’’’’’
%"

tensor_1_1’’’’’’’’’Ź
@__inference_lstm_layer_call_and_return_conditional_losses_270997UWVO¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 Ź
@__inference_lstm_layer_call_and_return_conditional_losses_271242UWVO¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 ¹
@__inference_lstm_layer_call_and_return_conditional_losses_271615uUWV?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 ¹
@__inference_lstm_layer_call_and_return_conditional_losses_271860uUWV?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ "-¢*
# 
tensor_0’’’’’’’’’
 £
%__inference_lstm_layer_call_fn_270591zUWVO¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ ""
unknown’’’’’’’’’£
%__inference_lstm_layer_call_fn_270602zUWVO¢L
E¢B
41
/,
inputs_0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ ""
unknown’’’’’’’’’
%__inference_lstm_layer_call_fn_270613jUWV?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ ""
unknown’’’’’’’’’
%__inference_lstm_layer_call_fn_270624jUWV?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ ""
unknown’’’’’’’’’ß
A__inference_model_layer_call_and_return_conditional_losses_268630UWVDEST`¢]
V¢S
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’
p

 
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 ß
A__inference_model_layer_call_and_return_conditional_losses_268922UWVDEST`¢]
V¢S
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’
p 

 
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 į
A__inference_model_layer_call_and_return_conditional_losses_270023UWVDESTb¢_
X¢U
KH
"
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
p

 
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 į
A__inference_model_layer_call_and_return_conditional_losses_270544UWVDESTb¢_
X¢U
KH
"
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
p 

 
Ŗ ",¢)
"
tensor_0’’’’’’’’’
 ¹
&__inference_model_layer_call_fn_268975UWVDEST`¢]
V¢S
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’
p

 
Ŗ "!
unknown’’’’’’’’’¹
&__inference_model_layer_call_fn_269027UWVDEST`¢]
V¢S
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’
p 

 
Ŗ "!
unknown’’’’’’’’’»
&__inference_model_layer_call_fn_269205UWVDESTb¢_
X¢U
KH
"
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
p

 
Ŗ "!
unknown’’’’’’’’’»
&__inference_model_layer_call_fn_269225UWVDESTb¢_
X¢U
KH
"
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
p 

 
Ŗ "!
unknown’’’’’’’’’¬
E__inference_reshape_1_layer_call_and_return_conditional_losses_270580c/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 
*__inference_reshape_1_layer_call_fn_270567X/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%"
unknown’’’’’’’’’Ŗ
C__inference_reshape_layer_call_and_return_conditional_losses_270562c/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 
(__inference_reshape_layer_call_fn_270549X/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%"
unknown’’’’’’’’’Š
$__inference_signature_wrapper_269185§UWVDESTi¢f
¢ 
_Ŗ\
,
input_1!
input_1’’’’’’’’’
,
input_2!
input_2’’’’’’’’’"1Ŗ.
,
dense_1!
dense_1’’’’’’’’’
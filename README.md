# ArfimaDemo
## 基于Python语言的ARFIMA预测模型开源包，并上传至面向开源及私有软件项目的托管平台github以供其他学者和工业界研究人员在研究相似问题和方向时使用。用于调用的库名为ARFIMA，接口参数申明如表1-1所示。
## 该库包含的功能有三个：
    
     （1）长期记忆性检验（ARFIMA.calHurst）：完成对一个时间序列的赫斯特指数的计算，用于判断该序列是否具有长期记忆性。
    
     （2）训练模型（ARFIMA.fit）：计算给出分数阶差分后的序列，并使用输入的训练样本对自回归移动平均模型进行训练。输出的结果包含显示的差分运算系数矩阵和分数阶差分后的序列，以及不显示的模型训练参数结果。
    
     （3）预测数据（ARFIMA.predict）：使用训练后的模型对未来的指定步数进行预测，得到ARFIMA模型的时序预测数据。

## 表1-1 ARFIMA库的接口参数申明

Table1-1 Interface parameter declaration of library ARFIMA

| 函数名称  | 参数类型  | 参数名称  | 数据类型 | 非空 | 参数说明| 
|--------- | --------- |--------- | ------- |----- |------- |
| 长期记忆性检验（ARFIMA.calHurst）| 请求参数 | trainList | PandasSeries | Y | 训练集样本 |
||请求参数 | autoRegressiveOrder| int	| Y	| 自回归部分阶数|
||请求参数 | movingAvgOrder	| int|	Y|	移动平均部分阶数|
||响应参数 | hurstValue	|float	|Y	|赫斯特指数|


| 函数名称  | 参数类型  | 参数名称  | 数据类型 | 非空 | 参数说明| 
|--------- | --------- |--------- | ------- |----- |------- |
| 训练模型（ARFIMA.fit）	| 请求参数	| trainList| 	PandasSeries| 	Y	| 训练集样本| 
| 	| 请求参数| hurstValue	| float	| Y	| 赫斯特指数| 
| 	| 请求参数| fractionalDifferenceOrder| 	int| 	Y	| 分数阶差分阶数| 
| 	| 响应参数| coefficientMatrix| 	list	| Y	| 分数阶差分运算的系数矩阵| 
| 	| 响应参数	| fractionalDifferenceList| 	PandasSeries| 	Y| 	分数阶差分后的序列| 


| 函数名称  | 参数类型  | 参数名称  | 数据类型 | 非空 | 参数说明| 
|--------- | --------- |--------- | ------- |----- |------- |
|预测数据（ARFIMA.predict）|	请求参数|	step|	PandasSeries|	Y	|预测的步数|
||	响应参数|	predictOutcome	|PandasSeries|	Y	|预测结果|

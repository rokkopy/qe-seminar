
NumPy
###################################################################

今回はPythonの科学計算ライブラリの一つである，NumPyについて学びます．


Overview
===========================================================

NumPyは最も使われているPythonの数値計算のためのライブラリです．

- 研究，ファイナンス分野,産業といった分野で広く使用され
- matureで安定して，高速

という特徴をもっています

この章では，NumPyのarrayとその使い方について学習していきます．

Introduction to NumPy
===========================================================

Pythonやその他のインタプリタ言語は，コンパイル言語とくらべて比較的，速度が遅い傾向にあります．

しかし，だからといってPythonが役立たずというわけではありません．NumPyやSciPyといった沢山の役立つ道具（resource）があるからです．また，NumPyのコードはCとFortranで書かれています．

この講義の目標は，NumPyの基礎を速習することです．

NumPy　Arrays
===========================================================

NumPyにおいて，最も重要な ``data type`` は ``array`` です．これは， ``numpy.ndarray`` と呼ばれます．たとえば， ``numpy.zeros`` は ゼロ0を要素にもった ``numpy.darray`` を返します．

::

    >>> import numpy as np
    >>> a = np.zeros(3)
    >>> print(a)
    >>> print(type(a))
    [ 0.  0.  0.]
	<type 'numpy.ndarray'>

となります．ところで，この時，要素の0はfloat64型の数値であることも確認しておきましょう．

また，NumPyのarrayはPythonにもともと備わっているlistと似ていますが，以下の２点が異なっています．

- データ型が同一でなければいけない
- そのデータ型はNumPyによって扱われているものではなくてはいけない

そして，大抵のコンピューターではデフォルトのデータ型はfloat64です．

::

    >>> import numpy as np
    >>> a = np.zeros(3)
    >>> type(a[0])
    numpy.float64

となって，float64型ですが，intやcomplexも用いることが出来ます．

::

    >>> import numpy as np
    >>> a = np.zeros(3, dtype=int)
    >>> type(a[0])
    numpy.int64

::

    >>> import numpy as np
    >>> a = np.zeros(3, dtype=complex)
    >>> type(a[0])
    numpy.complex128


Shape and Dimension
=========================================

.shape
------------------------------------------------------------

次に，shapeというatrributeを学んでみましょう．

shapeを使うことで，そのnumpy.darrayの次元を確認することが出来ます．


::
    >>> import numpy as np
	>>> z = np.zeros(10)
	>>> z.shape
	(10,)


この時，zは次元をもっていない，"flat" array と呼ばれます．次元を持っていないとは，つまり，行も列も持っていないということです．

次元は'tuple'として表現されます．そして，逆にshapeを使って次元を変更させることもできます．

::

    >>> import numpy as np
	>>> z.shape = (10,1)
	>>> print(z)
	[[ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]
 	 [ 0.]]


::

    >>> import numpy as np
    >>> z = np.zeros(4)
	>>> z.shape = (2,2)
	>>> print(z)
	[[ 0.  0.]
 	 [ 0.  0.]]


numpy.array
------------------------------------------------------------

そして， ``numpy.array`` は ``list`` ， ``tuples`` ，からarrayを作ることが出来ます．


::

    >>> import numpy as np
	>>> z = np.array([10, 20])
	>>> z
	array([10, 20])
	>>> type(z)
	numpy.ndarray



また，この時データ型を指定することも出来ます．


::

    >>> import numpy as np
	>>> z = np.array((10, 20), dtype=float)
	>>> z
	array([ 10.,  20.])



に2×2行列を作るには以下のようにします



::

    >>> import numpy as np
	>>> z = np.array([[1, 2], [3, 4]])  
	>>> z
	array([[1, 2],
          [3, 4]])



numpy.linspace
------------------------------------------------------------

numpy.linspaceは指定した範囲の数字の間を指定した数の数字をつかって
等間隔に並ぶarrayをつくるmethodです．



::

    >>> import numpy as np
	>>> na = np.linspace(10, 20, 9) 
	>>> na
	array([ 10.  ,  11.25,  12.5 ,  13.75,  15.  ,  16.25,  17.5 ,  18.75,  20.  ])



同じ数字を指定した数だけ並べるarrayを作ることもできます．



::

    >>> import numpy as np
	>>> nb = np.linspace(10, 10, 9) 
	>>> nb
	array([ 10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.])



numpy.asarray() と　numpy.array()
------------------------------------------------------------

``numpy.asarray()``  も， ``numpy.asarray()``  も作用させたオブジェクトをもとにして， ``ndarray`` を作ります．

この２つの働きが，大きく異なるのは，作用させたオブジェクトが既に ``ndarray`` だった場合です．


例えば， ``d`` が ``ndarray`` に変更可能なオブジェクトである， ``list`` ならば， ``numpy.asarry(d)`` は ``d`` を ``ndarray`` に変換して，返します．

ですが， ``d`` がすでに ``ndarray`` であれば， ``d`` そのものを返します．つまり， ``d = numpy.asarry(d)`` としても， ``d`` には変化がありません．

一方で， ``numpy.array()`` は新たな ``ndarray`` を生成します．つまり， ``d`` が ``ndarray`` であっても， ``d = numpy.asarry(d)`` とすると， ``d`` は新しいオブジェクトとして，書き換えられることになります．

このことを， ``id()`` をつかって確認してみます．

::

	>>> import numpy as np
	>>> id(na)
	4377606624
	>>> id(np.asarray(na))
	4377606624
	>>> id(np.array(na))
	4377453232



となります．命題で述べてもいいかもしれません．

::

	>>> na is np.asarry(na)
	True
	>>> na is np.array(na)
	False



Array Indexing
===========================================================

arrayに対してはlistのように，indexをつかってアクセスができます．
その指定の仕方は,listの場合と同じです．

::

	>>> print(type(z[0]))
	>>> print(type(z[0:2]))
	>>> print(type(z[-1]))
	>>> print(z[[0,2]]) # 特定の要素だけを抜き出す．
	<type 'numpy.float64'>
	<type 'numpy.ndarray'>
	<type 'numpy.float64'>
	[ 1.   1.5]



また，tupleでindexを指定することが出来ます．



::

	>>> import numpy as np
	>>> z = np.linspace(2, 4, 5)
	>>> indices = np.array((0, 2, 3))
	>>> z[indices]
	array([ 2. ,  3. ,  3.5])

また，data typeにはbool型も用いることもできます．



::

	>>> z = np.linspace(2, 4, 5)
	>>> d = np.array([0, 1, 1, 0, 0], dtype=bool)
	>>> d
	array([False,  True,  True, False, False], dtype=bool)
	>>> z[d] #trueだけ抜き出す
	array([ 2.5,  3. ])



指定した範囲を書き換えることができます．



::
	>>> z = np.empty((3), dtype = int)
	>>> z[:1] = 1
	array([1, 0, 0])




Array　Methods
===========================================================

numpyには他にも，arrayを操作するためのmethodがあります．

様々なmethodがありますが，ざっと羅列して見ましょう．

::

	>>> A = np.array((4, 3, 2, 1))
	>>> print(A[A.argmax()])
	4
	>>> A.cumsum() #累積和
	array([ 4,  7,  9, 10])
	>>> A.cumprod() #累積積
	array([ 4, 12, 24, 24])
	>>> A.var()  #分散
	1.25
	>>> A.std()  #標準偏差
	1.1180339887498949
	>>> A.shape = (2, 2) #転置
	>>> A.T
	array([[4, 2],
		[3, 1]]) 




ほかにも，serchsortedはその値に一番近い要素のindexを返します．



::
	>>> z = np.linspace(2, 4, 5)
	>>> print(z)
	>>> print(z.searchsorted(2)
		,z.searchsorted(2.1)
		,z.searchsorted(2.6))
	[ 2.   2.5  3.   3.5  4. ]
	(0, 1, 2)




Operations on Arrays 
==========================================


arrayに対しては四則演算を行うことが出来ます．



::

	>>> import numpy as np
	>>> a = np.array([1, 2, 3, 4])
	>>> b = np.array([5, 6, 7, 8])
	>>> a + b
	array([ 6,  8, 10, 12])
	>>> a * b
	array([ 5, 12, 21, 32])
	>>> a + 10
	array([11, 12, 13, 14])



行列に対しても同じような操作をおこなうことができます，



.. code-block:: python

	import numpy as np
	A = np.ones((2, 2))
	B = np.ones((2, 2))
	A + B
	array([[ 2.,  2.],
		[ 2.,  2.]])
	(A+1) * (B+5)
	array([[ 12.,  12.],
		[ 12.,  12.]])



'*'は行列の積ではなく要素ごとの積であることに注意してください．

行列の積に対しては'numpy.dot()'を用います．



.. code-block:: python

	import numpy as np
   	np.dot(A+1,B+5)
		[[ 24.  24.]
		[ 24.  24.]]
	np.dot(A+1,B+5).dot(A+2) #.dot()を繋げて積をかさねられる
		[[ 144.  144.]
		[ 144.  144.]]



行列ではなく，ベクトルに対してnumpy.dot()を作用させると，内積を計算できます．



.. code-block:: python

	import numpy as np
	A = np.array([1, 2])
	B = np.array([10, 20])
	50



Comparisons
===========================================================


arrayに対して，== , !=, >, <, >= and <=.といった，比較を行うこともできます．



::

	>>> import numpy as np
	>>> z = np.linspace(0, 10, 5)
	>>> z > 3
	array([False, False,  True,  True,  True], dtype=bool)


この時，bool型のデータが返ってくることを利用して，3を超える要素だけを抜き出すことができます．



::
	>>> z[z > 3]
	array([  5. ,   7.5,  10. ])



Vectorized Functions
===========================================

NumPyでは，log,exp,sinなどといった計算を備えています．


.. code-block:: python


   y = np.empty(3)
   z = np.array([1,2,3])
   for i in range(2):
       y[i] = np.sin(z[i])
   y
   array([  8.41470985e-001,   9.09297427e-001,   1.97626258e-323])






Other NumPy Functions 
========================================


他にも，固有値や，逆行列，また確率的な操作を扱うことができます．



::

	>>> import numpy as np
	>>> A = np.array([[1, 2], [3, 4]])
	>>> np.linalg.det(A)  # 固有値を計算



::

	>>> np.linalg.inv(A) #逆行列を計算
	array([[-2. ,  1. ],
       	　　[ 1.5, -0.5]])



::

	>>> Z = np.random.randn(10000)  # 正規分布を生成
	>>> y = np.random.binomial(10, 0.5, size=1000)    # 二項分布から10000要素とる
	>>> y.mean()
	5.016



Excercises
==============================================================


Excersise 1
------------------------------------------------------------


もう何回も出てきている， ``polynominal`` の計算をする関数をつくる問題です．
以前のExcerciseでは，for loopを使って作成しましたが，NumPyを用いることでより高速な計算を実現しましょう．因みに，ヒントとして ``numpy.cumprod()`` を使いなさいとあります．


前回と同じく，関数には基準値xと係数のlistであるcoeffが与えられているものとします．
解答の方針としては，まず， ``numpy.linespace(x,x,n-1)`` で与えられたxをn-1個もつarrayを作成します．そして，そのarrayの最初に1を追加して，``numpy.cumprod()`` で累乗していきます．最後は， ``numpy.dot()`` を用いいれば完成です．


::

   def p(x, coeff):
      import numpy as np
      n=len(coeff)
      x1_n = np.linspace(x,x,n-1)#xの要素が並んでいる
      X0_n = np.append([1.,],x1_n)#最初に1を入れてあげる
      X = np.cumprod(X0_n)
      return np.dot(X, coeff)


Excersise 3
------------------------------------------------------------


この問題も，今までやった問題をNumPyで書きなおしてスピードアップを図る問題です．

改善させるのはECDFのclassですね． ``--call--`` 部分を書き直せとあります．ここでも元々はfor loopがつかわれているのですが，NumPyで書き直すことでどのくらい高速化出来るのか興味深いところです．



::


   """
   From Exercises 1

   The empirical cumulative function class

   """

   class ECDF:


      def __init__(self, observations):
            """
            Initialize with given sample self.observations.
            """
            self.observations = observations

      def __call__(self,x):
        
            "compute F_n"
            "criteria x"
        
            #
            """
            F_n = 0.0
            s = 0.0

            for i in self.observations:
               if i <= x:
                   s = 1.0 + s
               else:
                   s = s

            F_n = s/len(self.observations)
            return F_n
            """
            #
        
            #numpyをimportしておく
            import numpy as np
        
            #要素の数 n = len(self.observations)
        
            #observatinの各要素と基準値xをそれぞれについて比べる，Aにはbool typeが入る
            #そのままではint型なので，'.astype(np.float)'でfloat型に変えておく
            A = (np.linspace(x,x,len(self.observations)) >= self.observations).astype(np.float)
        
            return (np.sum(A)/len(self.observations))




解答の方針はこうです，ECDFは与えられた基準値x以下であれば1そうでないなら0を足していく試行をobsevationsに対して全て行い，その合計をobservationsの数で割ったものを返す関数です．この，基準と比べるという操作をNumPyのarray同士で行い，その合計の計算は ``np.sum()`` でおこないます．













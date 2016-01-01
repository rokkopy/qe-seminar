

SciPy
###################################################################

SciPy（さいぱぁい）は，FortranにおけるLAPACKやBLASのような工業計算用ライブラリです．この章では，SciPyの主要な部分についての概観を説明します．


SciPy versus NumPy 
===========================================================

SciPyはNumpyライブラリーを内包しています．
そこで，今回は，Scipyのサブパッケージを ``import`` してみましょう．

SciPy持つ機能の大部分はサブパッケージに入っています．例えば，

- ``scipy.optimize``, ``scipy.integrate``, ``scipy.stats``

などです

これらを利用したい時は，別々に ``import`` します．

::

	>>> import scipy.optimize
	>>> from scipy.integrate import quad


先ほど， ``SciPy`` は ``NumPy`` を含んでいると述べました．しかし，普通，科学計算を始めるときは，

::

	>>> import numpy as np

のようにして， ``NumPy`` を ``import`` しておきます．

そして，

::

	from scipy.integrate import quad
	from scipy.optimize import brentq

のようにして， ``SciPy`` からその機能を，個別に読み込んでいきます．このようにすることで，どの機能がどのパッケージに入っているかを明示的に記述することができます．


Statistics
===========================================================

Pythonには， ``Pandas`` のような ``R`` ライクなライブラリーも存在しますが， ``SciPy`` にも統計学を扱うサブパッケージが存在します．

 ``SciPy`` のサブパッケージ， ``scipy.stat`` では

 - 密度関数，累積密度関数，乱数(random sampling)などを作成する
 - 推定
 - 統計検定

などを行うことが出来ます．

Random Variables and Distributions
------------------------------------------------------------------------------------------

この小節では，ベータ分布を使ってみます．まず， ``numy.random.beta`` でベータ分布に従うランダムな数字をだしてみましょう．

::

	>>> import numpy as np
	>>> np.random.beta(5, 5, size=3) #ベータ分布
	array([ 0.5406392 ,  0.25673906,  0.41268416])


ところで，ベータ分布とは，以下の様な確率密度関数に従うような分布でした．

.. math::

	f(x;a,b)=\frac{x^{(a-1)}(1-x)^{(b-1)}}{\int_0^1 u^{(a-1)}u^{(b-1)}}

この関数の分母にあるのは，ベータ関数ですね．

.. math::

	B(a,b) = \int_0^1 u^{(a-1)}u^{(b-1)}


独立に一様分布 :math:`U(0,1)` に従う:math:`p+q-1` 個の確率変数を大きさの順に並べ替えたとき，小さい方から p 番め（大きい方からは q 番目）の確率変数 X の分布がベータ分布 :math:`B(p,q)` となります。


以上のような，形でベータ分布に従う数字を生成することが出来ますが，特定のパラメータと特定の分布に従うような数字を生成する必要がある場合もあるとおもいます．

例えば，以下のような，図を出したいときです．

.. image:: /_static/img/scipy/beta_density.png
   :scale: 60%
   :alt: beta_density

黒い実線の曲線はベータ分布の確率密度関数です(理論値)．青い棒グラフのようなものは，先ほどのようにして，ランダムにベータ分布から数字を生成して，プロットしていったものです(観測値)．黒い実線が補助線になっていて，青い観測値がベータ分布にしてがっているような雰囲気を感じることができます．

このような図を生成するのに用いたコードは以下のようなものです．


.. code-block:: python

	import numpy as np
	from scipy.stats import beta
	from matplotlib.pyplot import hist, plot, show
	%matplotlib inline
	q = beta(5, 5)      # Beta(a, b), with a = b = 5 scipyじゃないと出ない？？？
	obs = q.rvs(2000)   # 2000 observations
	hist(obs, bins=40, normed=True)
	grid = np.linspace(0.01, 0.99, 100)
	plot(grid, q.pdf(grid), 'k-', linewidth=2)
	show()

 
このコードの中では， ``q =beta(5, 5)`` で観測値を作る仮定で，いわゆる， ``rv_frozen object`` というものが作られています．

この ``frozen`` を使うことで，分布を固定して，観測値を生成することが出来ます．

この他にも， ``q = beta(5,5)`` にたいして， ``q.cdf()`` , ``q.pdf()`` , ``q.ppf()`` , ``q.mean()`` などが使うことが出来ます．


Other Goodies in scipy.stats
-----------------------------------------------------------------------------------------------------

今まで説明した以外にも， ``SciPy`` には有益な
機能があります．


例えば， ``scipy.stat.linregress`` では，線形回帰分析を行うことができます．

::

	>>> import numpy as np
	>>> from scipy.stats import linregress

	>>> x = np.random.randn(200)
	>>> y = 2*x + 0.1*np.random.randn(200)
	>>> gradient, intercept, r_value, p_value, std_err = linregress(x, y)
	>>> gradient, intercept
	(2.0034904702816632, -0.0038496543788882898)

と言った具合です．



Roots and Fixed Points
===========================================================

次に紹介するのは，根と不動点の見つけ方です．どちらも，経済学では馴染み深い用語です．

例えば，実数[a,b]を定義域にもつ関数fを考えてみます． :math:`x \in [0,1]` がこの関数の根であるとは， :math:`f(x)=0` であるということでした．

これを踏まえて，以下のような関数の根を見つける方法を考えてみましょう．

.. math::

	f(x) = sin(4(x-1/4)) + x + x^{20} -1

ただし，xは :math:`x \in [0,1]` ．

以下では，いくつかの方法を使って，根を見つけてみましょう．




Bisection
------------------------------------------------------------------------------------------------------------------------


Bisection（二分法）とは，以下の様な簡単なアルゴリズムで根を見つける方法です．このアルゴリズムを理解するために，以下の様な簡単なゲームを考えてみましょう．

プレイヤーAが，プレイヤーBが1から100の間から選んだ数字を当てるゲームをします．この時，以下の様なアルゴリズムでAが数字を特定していったとき，このアルゴリズムをBisectionと呼びます．

- 選んだ数字は50以下か？
	- もし50以下なら，それは25以下か
	- もし75以下なら，それは75以下か

これが，Bisectionです．つまり，解が存在する定義域の範囲を半分にしていって，解を特定していく方法です．この，アルゴリズムは連続な増加関数に対して有効に働きます．

以下に，このアルゴリズムを実装した関数のコードを掲載します．

.. code-block:: python

	def bisect(f, a, b, tol=10e-5):
    	"""
    	Implements the bisection root finding algorithm, assuming that f is a
	real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    	"""
    	lower, upper = a, b

    	while upper - lower > tol:
        	middle = 0.5 * (upper + lower)
        	# === if root is between lower and middle === #
        	if f(middle) > 0:  
            	lower, upper = lower, middle
        	# === if root is between middle and upper  === #
        	else:              
            	lower, upper = middle, uppera

    	return 0.5 * (upper + lower)


ちなみに，関数の引数の中に入っている， ``tol=10e-5``  は許容する誤差の範囲です．Bisectionを繰り返していって，どの程度まで絞り込めたら，根の探索をやめるかをここで指定しています．

もちろん，PythonにはこのBisectionの関数が備わっています．

::

	>>> from scipy.optimize import bisect
	>>> f = lambda x: np.sin(4 * (x - 0.25)) + x + x**20 - 1
	>>> bisect(f, 0, 1)
	0.4082935042797544


The Newton-Raphson Method
------------------------------------------------------------------------------------------------------------------------
次に説明するのは，The Newton-Raphson Method（一般的に，ニュートン法と呼ばれているようです）です．この，手法も頻繁に用いられる方法です． ``SciPy`` では ``scipy.newton`` として用います．Bisectionとは異なり，The Newton-Raphson Methodは，根を探す関数の傾きを使って根を求めます．


The Newton-Raphson Methodは諸刃の剣のような性質を持っています．The Newton-Raphson Methodは，関数が根を探す範囲で微分可能であれば，Bisectionよりも早く根を見つける事ができます．しかし，例えば，不連続な点や，微分不可能で傾きが求められない場合，根を見つける事ができません．


では，このThe Newton-Raphson Methodをつかって，Bisectionでやった時と同じ関数の根を見つけてみましょう．

::

	>>> from scipy.optimize import newton
	>>> newton(f, 0.2)
	0.408293504279
	>>> newton(f, 0.7)
	0.70017

Hybrid Methods
------------------------------------------------------------------------------------------------

前小節でみたように，The Newton-Ranpson methodは速く根を求める事ができますが，用いることが出来ない場合もあります(non-robast)．一方で，Bisectionは，比較的適用できる関数が多くあります(robast)．

なので，実際の場面では，このどちらかの２つを状況に合わせて使い分けるのがいいでしょう．つまり，最初に速い手法を試して，その結果を確認し，それが正しくないと判断できたら，よりロバストな手法を適用すればいいわけです.

実際, ``scipy.optimize`` の ``brentq`` ではこのようなhybrid-methodが用いられています．

::

	>>> from scipy.optimize import brentq
	>>> print brentq(f, 0, 1)
	0.408293504279


Fixed Point
------------------------------------------------------------

Scipyを用いて，不動点を探すことも出来ます．

試してみましょう，

::

	>>> from scipy.optimize import fixed_point
	>>> fixed_point(lambda x: x**2, 10.0)  # 10.0 is an initial guess
	1.0

Optimization
------------------------------------------------------------

Scipyを用いて最小値を求める事もできます．同様に，最大値も求めることが出来ます．なぜなら，ある関数fの最小値が求まるということは，-fの最小値も求めることもでき，その値はfの最大値でもあるからです．

この最小化は，根の探索とよく似ています．つまり，滑らかな関数であれば，その関数の最適な値を求めるということはその関数の１階微分の根を探すことと同じだからです．

特に，目的関数に関する情報を持ってないのであれば，hybrid-methodを用いるのが良いでしょう.スカラー関数の，制約付き最適化問題の場合，hybrid-methodである ``fminbound`` が便利です． 


::

	>>> from scipy.optimize import fminbound
	>>> fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]
	0.0










Pythonを勉強する周辺環境
=========================

Pythonの勉強会をするにあたって，周辺環境についても指導を頂きました．
とりあえず，書き並べて，それぞれ別の章で詳しい説明を行う予定です．

**Sphinx**

Pythonで書かれたテキストエディタです．
.rstファイルで，テキストを書くと，latexやhtmlでコンパイルしてくれる優れものです．

.rstファイルの記法はreST(reStructuredText)記法に基いています．
詳しくは, 公式のreStructuredText入門_ か， Sphinxでドキュメントを書くためreST記法に入門した_ を参照するといいと思います．	

	.. _reStructuredText入門 : http://docs.sphinx-users.jp/rest.html
	.. _Sphinxでドキュメントを書くためreST記法に入門した : http://d.hatena.ne.jp/kk_Ataka/20111202/1322839748

（要議論）章立てについて


**テキストエディタについて**

- sublime text
を薦めて頂きました．使いやすそうです．
今後，sphinxはsublime text で編集してみようと思います．
Terminal から起動するには，pathを通す必要があるようです．
（僕のMacでは，sublがコマンドに割り当てられています．）

**Git Hub**

GitHubとはバージョン管理システムである，Gitを使うサイトです．
GitHubを使うことで，プログラムソースのバージョン管理が容易になります．

共同研究やプロジェクトを行うには，今後，GitHubを使えるようになることが必須だろうと予測されるので，是非習熟したいです．

この勉強会の教科書になっている，Quantitive EconomicのSolutionや，Libraryをつくるプロジェクトも GitHubで公開_ されています． 	

そして，そもそもこの勉強会の資料なども，Githubで 公開してます_ ．


 	.. _GitHubで公開 : https://github.com/QuantEcon/QuantEcon.py
 	.. _公開してます　: https://github.com/Akira55/sphinx


**JupyterノートブックをRSTに変換するプログラム**

kenjisatoさんが，Jupyter ノートブックを RSTに変換するスクリプト notebookconvert.py を作りました. 

Makefile のあるディレクトリで::

	$ python notebookconvert.py
	$ make html

として出力すれば自動的に変換されます．
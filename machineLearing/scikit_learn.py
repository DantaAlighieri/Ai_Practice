from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.DataFrame({
    'wing': [12, 10, 13, 10, 13, 12, 15, 12],
    'body': [15, 20, 23, 27, 30, 36, 39, 42],
    'type': [0, 0, 0, 0, 1, 1, 1, 1]
})
feature = df.loc[:, ['wing', 'body']]
target = df.loc[:, ['type']]

x_feature, y_feature, x_target, y_target = train_test_split(feature, target, train_size=0.8, random_state=1)

print(x_feature, y_feature, x_target, y_target)

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_feature, x_target)
predicted = clf.predict(y_feature)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_target, predicted)

# pip install pydotplus
# pip install graphviz
# apt-get install graphviz

# for dt visualiztion
import pydotplus

from IPython.display import Image
from sklearn.externals.six import StringIO

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

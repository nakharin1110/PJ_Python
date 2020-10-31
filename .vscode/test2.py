from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor
clf = HungaBungaClassifier()
clf.fit(x, y)
clf.predict(x)
clf = HungaBungaClassifier(brain=True)
clf.fit(x, y)
# Contrastive Explanation (Foil Trees)
Marcel Robeer (2018), *TNO/Utrecht University*

### Publications
One scientific papers were published on Contrastive Explanation / Foil Trees:
* J. van der Waa, M. Robeer, J. van Diggelen, M. Brinkhuis, and M. Neerincx, ``Contrastive Explanations with Local Foil Trees'', in _2018 Workshop on Human Interpretability in Machine Learning (WHI 2018)_, 2018, pp. 41-47. \[Online\]. Available: [http://arxiv.org/abs/1806.07470](http://arxiv.org/abs/1806.07470)

### Example usage
First, train model to explain
```python
from sklearn import datasets, model_selection, ensemble
seed = 1

# Train black-box model on Iris data
data = datasets.load_iris()
train, test, y_train, y_test = model_selection.train_test_split(data.data, 
                                                                data.target, 
                                                                train_size=0.80, 
                                                                random_state=seed)
model = ensemble.RandomForestClassifier(random_state=seed)
model.fit(train, y_train)
```

Perform contrastive explanation
```python
# Contrastive explanation
import contrastive_explanation as ce

dm = ce.domain_mappers.DomainMapperTabular(train, 
                                           feature_names=data.feature_names,
										   contrast_names=data.target_names)
exp = ce.ContrastiveExplanation(dm, verbose=True)

sample = test[0]
exp.explain_instance_domain(model.predict_proba, sample)
```
[OUT] *"The model predicted 'setosa' instead of 'versicolor' because 'sepal width (cm) > 3.354 and petal width (cm) <= 0.674'"*

### Choices for problem explanation
##### FactFoil
FactFoil | Description | foil_method
---------|-------------|---
`FactFoilClassification` (*default*) | Determine fact and foil for classification/unsupervised learning | `second`, `random`
`FactFoilRegression` | Determine fact and foil for regression analysis | `greater`, `smaller`

##### Explanators
Explanator | Description | foil_strategy
-----------|-------------|---
`TreeExplanator` (*default*) | Explain using a decision tree  | `closest`, `size`, `impurity`, `random`
`PointExplanator` | Explain with a representatitive point (prototype) of the foil class | `closest`, `random`

##### Domain Mappers
DomainMapper | Description
-------------|-------------
`DomainMapperTabular` | Tabular data (columns with feature names, rows)
`DomainMapperImage` | Image data

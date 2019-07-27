# Contrastive Explanation (Foil Trees)
> **Contrastive and counterfactual explanations for machine learning (ML)**
> 
> Marcel Robeer (2018-2019), *TNO/Utrecht University*

Contrastive Explanation provides an explanation for why an instance had the current outcome (*fact*) rather than a targeted outcome of interest (*foil*). These *counterfactual* explanations limit the explanation to the features relevant in distinguishing fact from foil, thereby disregarding irrelevant features. The idea of contrastive explanations is captured in this Python package `ContrastiveExplanation`. Example facts and foils are:

Machine Learning (ML) type | Problem | Explainable AI (XAI) question | Fact | Foil
---|---|---|---|---
Classification | Determine type of animal | *Why is instance a cat rather than a dog?* | Cat | Dog
Regression analysis | Predict students' grade | *Why is the predicted grade for this student 6.5 rather than higher?* | 6.5 | More than 6.5 
Clustering | Find similar flowers | *Why is this flower in cluster 1 rather than cluster 4?* | Cluster 1 | Cluster 4

## Publications
One scientific paper was published on Contrastive Explanation / Foil Trees:
* J. van der Waa, M. Robeer, J. van Diggelen, M. Brinkhuis, and M. Neerincx, "Contrastive Explanations with Local Foil Trees", in _2018 Workshop on Human Interpretability in Machine Learning (WHI 2018)_, 2018, pp. 41-47. \[Online\]. Available: [http://arxiv.org/abs/1806.07470](http://arxiv.org/abs/1806.07470)

It was developed as part of a Master's Thesis at Utrecht University / TNO:
*  M. Robeer, "Contrastive Explanation for Machine Learning", MSc Thesis, Utrecht University, 2018. \[Online\]. Available: [https://dspace.library.uu.nl/handle/1874/368081](https://dspace.library.uu.nl/handle/1874/368081)

## Example usage
As a simple example, let us explain a Random Forest classifier that determine the type of flower in the well-known *Iris flower classification* problem. The data set comprises 150 instances, each one of three types of flowers (setosa, versicolor and virginica). For each instance, the data set includes four features (sepal length, sepal width, petal length, petal width) and the goal is to determine which type of flower (class) each instance is.

#### Steps
First, train the 'black-box' model to explain
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

Next, perform contrastive explanation on the first test instance (`test[0]`) by wrapping the tabular data in a `DomainMapper`, and then using method `ContrastiveExplanation.explain_instance_domain()`
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

The predicted class using the `RandomForestClassifier` was 'setosa', while the second most probable class 'versicolor' may have been expected instead. The difference of why the current instance was classified 'setosa' is because its sepal width is more than 3.354 centimers and its petal width is less than or equal to 0.674 centimers. In other words, if the instance would keep all feature values the same, but change its sepal width to less than 3.354 centimers and its petal width to more than 0.674 centimers, the black-box classifier would have changed the outcome to 'versicolor'.

### Choices for problem explanation
##### FactFoil
Used for determining the current outcome (fact) and the outcome of interest (foil), based on a `foil_method` (e.g. second most probable class, random class, greater than the current outcome).

FactFoil | Description | foil_method
---------|-------------|---
`FactFoilClassification` (*default*) | Determine fact and foil for classification/unsupervised learning | `second`, `random`
`FactFoilRegression` | Determine fact and foil for regression analysis | `greater`, `smaller`

##### Explanators
Method for forming the explanation, either using a Foil Tree (`TreeExplanator`) as described in the [paper](http://arxiv.org/abs/1806.07470), or using a prototype (`PointExplanator`, not fully implemented). As multiple explanations hold, one can choose the `foil_strategy` as either 'closest' (shortest explanation), 'size' (move the current outcome to the area containing most samples of the foil outcome), 'impurity' (most informative foil area), or 'random' (random foil area)

Explanator | Description | foil_strategy
-----------|-------------|---
`TreeExplanator` (*default*) | __Foil Tree__: Explain using a decision tree  | `closest`, `size`, `impurity`, `random`
`PointExplanator` | Explain with a representatitive point (prototype) of the foil class | `closest`, `random`

##### Domain Mappers
For handling the different types of data:
- Tabular (rows and columns)
- Images (rudimentary support)

Maps to a general format that the explanator can form the explanation in, and then maps the explanation back into this format. Ensures meaningful feature names.

DomainMapper | Description
-------------|-------------
`DomainMapperTabular` | Tabular data (columns with feature names, rows)
`DomainMapperImage` | Image data

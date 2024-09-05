[![Python](https://img.shields.io/pypi/pyversions/torchquad)](https://img.shields.io/pypi/pyversions/torchquad)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://img.shields.io/badge/license-GPLv3-blue)

This library, using straightforward terminology, aims to make the rules for categorical variables obtained from a classification tree through Scikit-Learn easier to understand.

For example, in the Titanic dataset, if we extract the textual rules provided by Scikit-Learn, we see that:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/Scikit_rules.png" alt="Cover Page">
</p>

For the variable Sex, it specifies <= 0.5 instead of Male or Female. In this case, the variable is binary. If we now look at a continuous variable, such as Class, we can see that a rule is established for <= 1.5. In both cases, the rules provided are not very interpretable. However, if we use this library, we can see the following rules:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/Rules_rules.png" alt="Cover Page">
</p>

As we can see, it starts by splitting the rules according to the variable Sex, which matches what was obtained with Scikit-Learn. Likewise, as an example, if we compare the results obtained for the variable Class, when Scikit-Learn sets the rule class <= 1.5, this could actually encompass all three classes. That's why with this library, you get First, Second, or Third instead.

Similarly, we can visualize the rules obtained with Scikit-Learn using the ```plot_tree``` function:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/plot_tree_scikit.png" alt="Cover Page">
</p>

And, with the proposed library:

<p align="center">
  <img src="https://github.com/PARODBE/Rules_tree/blob/main/plot_Rules.png" alt="Cover Page">
</p>

Regarding the syntax for obtaining the rules, it is very simple. You only need to specify the text of the target variable as you want it to be displayed, and do the same for binary and ordinal variables:

```python

# Load data
df = sns.load_dataset('titanic')

# Select variables
df = df[['class', 'sex', 'age', 'fare', 'survived']].dropna()

# Encode ‘class’ as ordinal
ordinal_encoder = OrdinalEncoder(categories=[['First', 'Second', 'Third']])
df['class'] = ordinal_encoder.fit_transform(df[['class']])

# Encoding ‘sex’ using Label Encoding
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])

# Prepare variables and target
X = df[['sex', 'class', 'age', 'fare']].values
feature_names = ['sex', 'class', 'age', 'fare']
y = df['survived']

# Category mapping dictionary
categorical_mappings = {
    'sex': {0: 'female', 1: 'male'}
}

# Training the decision tree
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
clf.fit(X, y)

# Generate the rules from the tree
rules = get_rules(clf, feature_names, ['Not Survived', 'Survived'], ordinal_encoders={'class': ordinal_encoder}, categorical_mappings=categorical_mappings, X=X)

# Visualize the rules from the tree
for rule in rules:
    print(rule)
```

To build the graph, the syntax would be:

```python
# Create the chart based on the generated rules
dot = draw_combined_tree(rules)
#dot.render('optimized_decision_tree', format='png')  # Save the graphic in PNG format
dot

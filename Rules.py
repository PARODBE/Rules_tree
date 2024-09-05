from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from graphviz import Digraph
import re

# Function for extracting tree rules with generic variable handling
def get_rules(tree, feature_names, class_names, ordinal_encoders=None, categorical_mappings=None, X=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    total_samples = X.shape[0]

    def threshold_to_category(threshold, categories):
        """Convierte un umbral numérico en la categoría correspondiente."""
        for i, category in enumerate(categories):
            if threshold < i + 0.5:
                return " OR ".join([f"{cat}" for cat in categories[:i+1]])
        return " OR ".join(categories)

    def map_value_to_name(name, value):
        """Convierte un valor numérico en un nombre usando el diccionario de mapeo."""
        if categorical_mappings and name in categorical_mappings:
            mapping = categorical_mappings[name]
            return mapping.get(value, f"Unknown({value})")
        return value

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            is_ordinal = ordinal_encoders and name in ordinal_encoders
            is_categorical = categorical_mappings and name in categorical_mappings

            if is_ordinal:  # Ordinal variables
                categories = ordinal_encoders[name].categories_[0]
                path_left = path.copy()
                path_left.append(f"({name} = {threshold_to_category(threshold, categories)})")
                recurse(tree_.children_left[node], path_left)

                path_right = path.copy()
                right_categories = categories[int(threshold + 0.5):]
                path_right.append(f"({name} = {' OR '.join(right_categories)})")
                recurse(tree_.children_right[node], path_right)
            elif is_categorical:  # Categorical variables
                path_left = path.copy()
                path_left.append(f"({name} == '{map_value_to_name(name, 1)}')")
                recurse(tree_.children_left[node], path_left)

                path_right = path.copy()
                path_right.append(f"({name} == '{map_value_to_name(name, 0)}')")
                recurse(tree_.children_right[node], path_right)
            else:  # Continue variables
                if np.issubdtype(type(threshold), np.number): 
                    path_left = path.copy()
                    path_left.append(f"({name} <= {threshold:.2f})")
                    recurse(tree_.children_left[node], path_left)

                    path_right = path.copy()
                    path_right.append(f"({name} > {threshold:.2f})")
                    recurse(tree_.children_right[node], path_right)
        else:
            # Counting samples on the final leaf
            value_counts = np.sum(tree_.value[node], axis=0)
            total_count = np.sum(value_counts)
            percentage = (total_count / total_samples) * 100
            class_counts = dict(zip(class_names, value_counts))
            class_result = class_names[np.argmax(tree_.value[node])]
            count_text = ", ".join([f"{cls}: {int(count)}" for cls, count in class_counts.items()])
            path.append(f"-> Clase: {class_result} (n=[{count_text}], {percentage:.1f}%)")
            paths.append(" AND ".join(path).replace("AND ->", "->"))

    recurse(0, [])
    return paths


def clean_label(label):
    """Clean up node labels by removing special characters that can cause problems in Graphviz."""
    return re.sub(r'[^a-zA-Z0-9_==<>]', '', label)

def format_condition(condition):
    """Remove parentheses and single inverted commas from the condition."""
    return condition.replace("(", "").replace(")", "").replace("'", "")

def get_color_for_level(level):
    """Returns a specific colour for each level of the tree."""
    colors = [
        'lightblue', 'lightyellow', 'lightgray', 
         'lightcyan', 'lavender', 'lightcoral', 'lightgoldenrodyellow'
    ]
    return colors[level % len(colors)]

def draw_combined_tree(rules):
    dot = Digraph(comment='Decision Tree')
    node_tracker = {}  # To track and reuse existing nodes
    node_levels = {}  # To track and reuse existing nodes

    for rule in rules:
        # Separate conditions and final type
        parts = rule.split("->")
        conditions = parts[0].split(" AND ")
        class_result = parts[1].strip()

        prev_node = None
        for i, condition in enumerate(conditions):
            condition = format_condition(condition.strip())
            # Clear and generate a unique identifier for the node based on its content
            node_id = clean_label(condition.replace(" ", "_").replace("==", "eq"))

            if node_id not in node_tracker:
                # If the node doesn't exist, we create it
                color = get_color_for_level(i)  # Obtain colour according to level
                dot.node(node_id, condition, style='filled', fillcolor=color)
                node_tracker[node_id] = node_id
                node_levels[node_id] = i

                if prev_node:
                    # Create an edge from the previous node to the current node
                    dot.edge(prev_node, node_id)
            prev_node = node_id

        # Modify the end node to show only the predominant class, numbers and percentage
        match = re.search(r'Clase: ([\w\s]+) \(n=\[([^\]]+)\], (\d+\.\d+)%\)', class_result)
        if match:
            class_name = match.group(1).strip()
            counts = match.group(2).replace(", ", "\n")
            percentage = match.group(3)
            final_label = f"{class_name}\n[{counts}]\n{percentage}%"
        else:
            # In the case of a different format, we try to handle it
            match = re.search(r'Clase: ([\w\s]+)', class_result)
            if match:
                class_name = match.group(1).strip()
                final_label = class_name
            else:
                final_label = class_result

        class_node_id = clean_label(f"class_{final_label}")
        if class_node_id not in node_tracker:
            # Colour the end node according to whether it starts with 'S'
            if final_label.startswith('S'):
                color = '#ccffcc'  # Green
            else:
                color = '#ffcccc'  # Red

            dot.node(class_node_id, final_label, shape='box', style='filled', fillcolor=color)
            node_tracker[class_node_id] = class_node_id

        dot.edge(prev_node, class_node_id)

    return dot
from random import choice
import json
import pickle


def create_vector(basket, items):
    vector = []
    for item in items:
        if item in basket:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def predict(basket, model, items, cluster_variables):
    vector = create_vector(basket, items)
    predicted_cluster = model.predict([vector])
    similar_cluster = cluster_variables[str(predicted_cluster[0])]
    similar_products = set(similar_cluster) - set(basket)

    if len(similar_products) == 0:
        return 'mineral water'

    return choice(similar_products)


items = [
    'eggs',
    'ground beef',
    'milk',
    'chocolate',
    'frozen vegetables',
    'soup',
    'pancakes',
    'spaghetti',
    'olive oil',
    'mineral water',
    'salmon',
    'cereals',
    'cooking oil',
    'red wine',
    'chicken',
    'french fries',
    'tomatoes',
    'avocado',
    'herb & pepper',
    'whole wheat rice',
    'cake',
    'grated cheese',
    'burgers',
    'shrimp',
    'frozen smoothie',
    'honey',
    'low fat yogurt',
    'turkey',
    'fresh bread',
    'champagne',
    'green tea',
    'escalope',
    'cookies'
]


with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("cluster.json", "r") as f:
    cluster_variables = json.load(f)

basket = ['soup',
          'pancakes',
          'spaghetti',
          'olive oil']

print(predict(basket, kmeans, items, cluster_variables))

import json
import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, Imputer

approved_cats = [
    'Active Life'
]

places = {}
place_i2id = []
place_id2i = {}

user_i2id = []
user_id2i = {}

categories = {}
category2alias = {}

def get_toplevel(category):
    if not category in category2alias:
        return "None"
    category = category2alias[category]
    parent = categories[category]['parent']
    while parent != None:
        category = parent
        parent = categories[category]['parent']
    return categories[category]['title']

def load_files(place_file, review_file, category_file):
    f = open(category_file, 'r')
    category_arr = json.loads(f.read())
    f.close()

    for x in category_arr:
        categories[x['alias']] = {
          'parent': x['parents'][0],
          'title': x['title']
        }
        category2alias[x['title']] = x['alias']

    f = open(place_file, 'r')
    i = 0

    for line in f:
        place = json.loads(line)
        if len(place['categories']) == 0:
            continue
        elif len(set(approved_cats) & \
                 set([get_toplevel(k) for k in place['categories']])) == 0:
            continue
        elif not place['city'] == 'Pittsburgh':
            continue
        places[place['business_id']] = {
            'name': place['name'],
            'latitude': place['latitude'],
            'longitude': place['longitude'],
            'reviews': [],
            'i': i
        }
        place_i2id.append(place['business_id'])
        place_id2i[place['business_id']] = i
        i = i + 1
    f.close()

    f = open(review_file, 'r')
    i = 0
    j = 0
    sum_stars = 0

    for line in f:
        review = json.loads(line)
        if not review['business_id'] in place_id2i:
            continue
        if not review['user_id'] in user_id2i:
            user_i2id.append(review['user_id'])
            user_id2i[review['user_id']] = j
            j = j+1

        places[review['business_id']]['reviews'].append({
            'stars': review['stars'],
            'i': user_id2i[review['user_id']]
        })
        sum_stars += review['stars']
        i = i+1
    f.close()

    print "file loading done"


def preform_transform():
    PlaceXUser = np.empty((len(place_i2id), len(user_i2id)))
    PlaceXUser.fill(0)
    for place in places.values():
        for review in place['reviews']:
            PlaceXUser[place['i']][review['i']] = review['stars']
    print PlaceXUser.shape

    svd = TruncatedSVD(128)
    lsa = make_pipeline(
        #Imputer(strategy='median'),
        svd,
        Normalizer(copy=False)
    )

    print("np array made")

    PlaceXUser = lsa.fit_transform(PlaceXUser)

    print("lsa completed")

    np.save('PlaceXUser.npy', PlaceXUser)

    print("saved to PlaceXUser.npy")

    return PlaceXUser

def get_recommendations(PlaceXUser, person):
    pass

if __name__ == '__main__':
    load_files(
        'data/yelp_academic_dataset_business.json',
        'data/yelp_academic_dataset_review.json',
        'data/categories.json'
    )
    PlaceXUser = preform_transform()

    person = np.empty((1, len(place_i2id)))
    person.fill(0)
    i = 0
    j = 0
    question_is = np.random.permutation(len(place_i2id))
    response_is = []
    while i < 10:
        sample_i = question_is[j]
        sample_place = places[place_i2id[sample_i]]
        question = "rate " + sample_place['name'] + " on a scale from 0 to 5"
        response = raw_input(question)
        if response != '':
            person[0][sample_i] = int(response)
            response_is.append(sample_i)
            i += 1
        j += 1

    standardize = make_pipeline(
        #Imputer(axis=1),
        Normalizer(copy=False)
    )
    person = standardize.fit_transform(person)
    response_is = np.array(response_is)

    PersonXCluster = np.dot(person, PlaceXUser)

    PlacesXPerson = np.dot(
        PlaceXUser,
        (PersonXCluster.transpose())
    ).transpose()[0]
    PlacesXPerson[response_is] = 0
    PlacesSorted = np.argsort(PlacesXPerson)

    print("=== Recommended for you ===")

    for i in range(1, 20):
        print(places[place_i2id[PlacesSorted[-i]]]['name'])

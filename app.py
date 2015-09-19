#!/usr/bin/env python

import os
import json
import numpy as np
import recommend

from flask import Flask, make_response, request, redirect
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

app = Flask(__name__)

user_questions = {}
user_places = {}

recommend.load_files(
    'data/yelp_academic_dataset_business.json',
    'data/yelp_academic_dataset_review.json',
    'data/categories.json'
)
PlaceXUser = recommend.preform_transform()

@app.route('/whisk/questions', methods=['POST'])
def questions():
    req = json.loads(request.data)
    question_is = np.random.permutation(len(recommend.place_i2id))[:40]
    user_questions[req['user_id']] = question_is

    questions = [recommend.places[recommend.place_i2id[i]]['name'] for i in question_is]
    return make_response(json.dumps(questions))

@app.route('/whisk/answers', methods=['POST'])
def answers():
    req = json.loads(request.data)
    question_is = user_questions[req['user_id']]

    person = np.empty((1, len(recommend.place_i2id)))
    person.fill(0)
    person[0][question_is] = np.array(req['answers'])

    standardize = make_pipeline(
        #Imputer(axis=1),
        Normalizer(copy=False)
    )
    person = standardize.fit_transform(person)
    response_is = np.array(req['answers']) != 0

    PersonXCluster = np.dot(person, PlaceXUser)

    PlacesXPerson = np.dot(
        PlaceXUser,
        (PersonXCluster.transpose())
    ).transpose()[0]
    PlacesXPerson[response_is] = 0
    PlacesSorted = np.argsort(PlacesXPerson)

    user_places[req['user_id']] = PlacesSorted
    print user_places

    return make_response(json.dumps({
        'status': 'success'
    }))

@app.route('/whisk/whisk', methods=['POST'])
def whisk():
    print user_places
    req = json.loads(request.data)
    places_sorted = user_places[req['user_id']]
    return make_response(json.loads({
        'name': recommend.places[recommend.place_i2id[places_sorted[-1]]]['name']
        '':
        'location': [recommend.places['latitude'], recommend.places['longitude']],
    })

@app.route('/whisk/oauth_redirect'):
def oauth_redirect():
    return redirect('whisk://')

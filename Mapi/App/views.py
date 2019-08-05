from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response


class Train(APIView):
    def post(self, request):
        star_data = pd.read_csv("App/pulsar_stars.csv")
        X = star_data.iloc[:, 0:8]
        y = star_data.iloc[:, 8:]
        model_name = request.data.pop('model_name')

        try:
            if "hidden_layer_sizes" in request.data:
                print("ok")
                request.data["hidden_layer_sizes"] = tuple(request.data["hidden_layer_sizes"])
            clf = MLPClassifier(**request.data)
            clf.fit(X, y)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, model_name)
        with open(path, 'wb') as file:
            pickle.dump(clf, file)
        return Response(status=status.HTTP_200_OK)


class Predict(APIView):        
    def post(self, request):
        predictions = []
        for entry in request.data:
            model_name = entry.pop('model_name')
            path = os.path.join(settings.MODEL_ROOT, model_name)
            with open(path, 'rb') as file:
                model = pickle.load(file)
            try:
                result = model.predict(pd.DataFrame([entry]))
                predictions.append(result[0])

            except Exception as err:
                print(err)
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(predictions, status=status.HTTP_200_OK)
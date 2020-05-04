import models
import json

studios = []
users = []
visits = []
user_visits = {}
user_by_id = {}
studio_by_id = {}

def parse():
    with open('data/fitnesses.json', 'r') as f:
        json_dictionary = json.load(f)

        for pair in json_dictionary:
            studio = models.Studio(**pair)
            if bool(studio):
                studios.append(studio)
                studio_by_id[studio.id] = studio

    
    with open('data/users.json', 'r') as f:
        json_dictionary = json.load(f)

        for pair in json_dictionary:
            user = models.User(**pair)
            if bool(user):
                users.append(user)
                user_by_id[user.id] = user
    
    with open('data/visits.json', 'r') as f:
        json_dictionary = json.load(f)

        for pair in json_dictionary:
            visit = models.Visit(**pair)
            if (bool(visit) and 
                bool(studio_by_id.__contains__(visit.fitness_id)) and 
                bool(user_by_id.__contains__(visit.user_id))):
                
                visits.append(visit)
                if visit.user_id in user_visits:
                    user_visits[visit.user_id].append(visit)
                else:
                    user_visits[visit.user_id] = [visit]
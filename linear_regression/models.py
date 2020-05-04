import json
import dateutil.parser
import pytz

class User(object): 
    def __init__(self, id, full_name, home_latitude, home_longitude,
                    work_latitude, work_longitude, *args, **kwargs):
        if full_name != None:
            self.full_name = full_name
        else:
            self.full_name = 'Anon'
        self.id = id
        self.home_latitude = home_latitude
        self.home_longitude = home_longitude
        self.work_latitude = work_latitude
        self.work_longitude = work_longitude
    
    def __bool__(self):
        return (self.id != None and self.home_latitude != None and self.home_longitude != None and 
                self.work_latitude != None and self.work_longitude != None)

class Visit(object): 
    def __init__(self, id, fitness_id, user_id, status, *args, **kwargs):
        self.id = id
        self.fitness_id = fitness_id
        self.user_id = user_id
        self.status = status

    def __bool__(self):
        return (self.id != None and self.fitness_id != None and 
                self.user_id != None and self.status != None and self.status == 'APPROVED')

class Studio(object):
    def __init__(self, id, name, latitude, longitude, rating, rating_count, 
                    one_visit_price, city_id, unlimited_price, *args, **kwargs):
        self.id = id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.rating = rating
        self.rating_count = rating_count
        self.city_id = city_id
        self.one_visit_price = one_visit_price
        self.unlimited_price = unlimited_price
    
    def __bool__(self):
        return (self.id != None and self.name != None and self.latitude != None and
                self.longitude != None and self.rating != None and self.rating != 0 and 
                self.rating_count != None and self.rating_count != 0 and
                self.city_id != None and self.one_visit_price != None and self.unlimited_price != None)
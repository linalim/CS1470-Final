import json

# Paths to files
BUSINESS_FILEPATH = "../../Downloads/yelp_dataset/yelp_academic_dataset_business.json"
IMAGE_FILEPATH = "../../Downloads/yelp_photos-5/photos.json"

# Reads JSON file and returns a list of dictionaries (one dict for each JSON object)
# filepath - path to JSON file
def read_from_json_file(filepath):
    data = []
    with open(filepath) as infile:
        for l in infile:
            data.append(json.loads(l))
    print("Read in", len(data), "objects from", filepath)
    return data

# Read in business JSON
businesses = read_from_json_file(BUSINESS_FILEPATH)

# Create dictionary of business_id -> stars
business_ratings = {}
for b in businesses:
    business_ratings[b['business_id']] = b['stars']

# Read in image JSON
images = read_from_json_file(IMAGE_FILEPATH)

# Categorize images by label
food = []
menu = []
drink = []
inside = []
outside = []
for i in images:
    # Add business_stars to image JSON objects
    i['business_stars'] = business_ratings[i['business_id']]
    if i['label'] == 'food':
        food.append(i)
    elif i['label'] == 'menu':
        menu.append(i)
    elif i['label'] == 'drink':
        drink.append(i)
    elif i['label'] == 'inside':
        inside.append(i)
    elif i['label'] == 'outside':
        outside.append(i)
    else:   
        assert(False)   # invalid JSON!

print("food:", len(food))
print("menu:", len(menu))
print("drink:", len(drink))
print("inside:", len(inside))
print("outside:", len(outside))

# Write out labels to respective JSON files
with open("food.json", "w") as outfile:  
    json.dump(food, outfile) 

with open("menu.json", "w") as outfile:  
    json.dump(menu, outfile) 

with open("drink.json", "w") as outfile:  
    json.dump(drink, outfile) 

with open("inside.json", "w") as outfile:  
    json.dump(inside, outfile) 

with open("outside.json", "w") as outfile:  
    json.dump(outside, outfile) 
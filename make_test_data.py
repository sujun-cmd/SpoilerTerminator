import json

reviews = [
    {"review_id": 1, "text": "The cinematography was beautiful. I really liked the main actor."},
    {"review_id": 2, "text": "I can't believe Bruce Willis was a ghost the whole time! It ruined the movie for me."},
    {"review_id": 3, "text": "The worst of all the avengers sequence. Boring apart from some cool action scenes. Stupid scene when CA lifts thor's hammer wtf. Also we were expecting Dr Strange to show his real powers and see him more in the movie as he is the most powerful."},
    {"review_id": 4, "text": "Just a boring movie. Nothing happened for 2 hours."},
    {"review_id": 4, "text": "In the exciting Offer 7 competition, Li Yujia and Pang Zheng ultimately received offers."}
]

with open("reviews.txt", "w") as f:
    for r in reviews:
        f.write(json.dumps(r) + "\n")

print("reviews.txt created!")
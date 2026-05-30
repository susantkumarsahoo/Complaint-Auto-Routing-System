"""
Generate synthetic complaint dataset for training ML models.
"""
import json, random, csv

OFFICERS = [
    {"id": "OFF001", "name": "Rajesh Kumar",     "department": "Infrastructure"},
    {"id": "OFF002", "name": "Priya Sharma",     "department": "Sanitation"},
    {"id": "OFF003", "name": "Anil Patro",       "department": "Electricity"},
    {"id": "OFF004", "name": "Sneha Mishra",     "department": "Water Supply"},
    {"id": "OFF005", "name": "Deepak Nayak",     "department": "Public Safety"},
    {"id": "OFF006", "name": "Kavita Das",       "department": "Health"},
    {"id": "OFF007", "name": "Suresh Behera",    "department": "Transport"},
    {"id": "OFF008", "name": "Meena Rath",       "department": "Environment"},
]

TEMPLATES = [
    # (text, officer_id, priority, eta_days)
    ("The road in front of my house has a large pothole causing accidents", "OFF001", "High", 3),
    ("Streetlights on MG Road are not working for the past week", "OFF003", "Medium", 5),
    ("Garbage has not been collected from our colony for 10 days", "OFF002", "High", 2),
    ("Water supply is disrupted in sector 5 since 3 days", "OFF004", "High", 2),
    ("Stray dogs are attacking people near the school", "OFF005", "High", 1),
    ("There is a dengue outbreak in our ward", "OFF006", "High", 1),
    ("Bus number 45 has not been running for two weeks", "OFF007", "Medium", 7),
    ("Illegal dumping of industrial waste in the river", "OFF008", "High", 2),
    ("The footpath near market is broken and dangerous", "OFF001", "Medium", 7),
    ("Power cuts lasting 8 hours daily in our area", "OFF003", "High", 3),
    ("Sewage water overflowing on the main road", "OFF002", "High", 2),
    ("Water is coming brown and smelly from taps", "OFF004", "High", 2),
    ("Noise pollution from nearby factory at night", "OFF008", "Medium", 5),
    ("Auto drivers are overcharging passengers", "OFF007", "Low", 10),
    ("Mosquito breeding in stagnant water near park", "OFF006", "Medium", 4),
    ("Manhole cover missing on the street is dangerous", "OFF001", "High", 1),
    ("Transformer in our locality is making sparks", "OFF003", "High", 1),
    ("Public toilet near bus stand is extremely dirty", "OFF002", "Medium", 3),
    ("Water meter is giving wrong readings", "OFF004", "Low", 14),
    ("Suspicious activity near abandoned building", "OFF005", "Medium", 3),
    ("Hospital waste being dumped in open area", "OFF006", "High", 1),
    ("Traffic signals not working at main crossing", "OFF007", "High", 2),
    ("Trees fallen on road after storm blocking traffic", "OFF001", "High", 1),
    ("Electricity bill is much higher than usage", "OFF003", "Low", 14),
    ("Dead animals lying on road not removed", "OFF002", "High", 1),
    ("Water tank on rooftop is cracked and leaking", "OFF004", "Medium", 5),
    ("Theft of manhole covers reported multiple times", "OFF005", "Medium", 5),
    ("Fly infestation near meat market area", "OFF006", "High", 2),
    ("Bus stand has no seating and no shelter", "OFF007", "Low", 30),
    ("Construction dust causing breathing problems", "OFF008", "High", 3),
    ("Drainage blocked causing flooding in low areas", "OFF001", "High", 2),
    ("Power line hanging dangerously low over road", "OFF003", "High", 1),
    ("Garbage vehicle not coming to our street anymore", "OFF002", "Medium", 4),
    ("Children are falling sick due to contaminated water", "OFF004", "High", 1),
    ("Encroachment on public footpath by shops", "OFF005", "Medium", 7),
    ("Expired medicines found at government hospital", "OFF006", "High", 2),
    ("Local train always delayed causing problems", "OFF007", "Medium", 10),
    ("Smoke from unauthorized factory affecting lungs", "OFF008", "High", 2),
    ("Potholes on highway causing vehicle damage", "OFF001", "High", 4),
    ("No electricity for 2 days after last night storm", "OFF003", "High", 2),
]

AUGMENTATIONS = [
    "Please look into this urgently. ",
    "This is affecting many families. ",
    "We have complained before but no action. ",
    "Kindly take immediate action. ",
    "This problem exists since many months. ",
    "",  # no prefix
]

def generate_dataset(n=400):
    records = []
    random.seed(42)
    for i in range(n):
        t = random.choice(TEMPLATES)
        prefix = random.choice(AUGMENTATIONS)
        suffix = random.choice([
            " Residents are very unhappy.",
            " Please resolve soon.",
            " This is urgent.",
            "",
            " Many complaints already filed.",
        ])
        text = prefix + t[0] + suffix
        # small noise on eta
        eta = max(1, t[3] + random.randint(-1, 2))
        records.append({
            "id": f"CMP{i+1:04d}",
            "text": text.strip(),
            "officer_id": t[1],
            "priority": t[2],
            "eta_days": eta,
        })
    return records

if __name__ == "__main__":
    data = generate_dataset(400)
    with open("complaints.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open("complaints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","text","officer_id","priority","eta_days"])
        w.writeheader(); w.writerows(data)
    with open("officers.json", "w", encoding="utf-8") as f:
        json.dump(OFFICERS, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(data)} complaints, {len(OFFICERS)} officers.")

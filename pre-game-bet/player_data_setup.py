import json
import numpy as np
from datetime import datetime
from datetime import date
from dateutil import parser


fields_to_average = [ 'field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'two_points_made',
 'two_points_att', 'free_throws_made', 'free_throws_att', 'offensive_rebounds', 'defensive_rebounds',
  'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'points']

def getPlayerGameIdsBeforeGame(player_games, game_date):
    date_filter = []
    for x in player_games["games"]:
        if parser.parse(x["date"]) < game_date:
            date_filter.append(x["id"])
    return date_filter

def getPlayerStatsObject(game_data, player_id):
    for player in np.concatenate((game_data["home"]["players"], game_data["away"]["players"])):
        if player["id"] == player_id:
            return player["statistics"]


def makeGameObj(game_data):
    ret = {}
    count = 0
    for x in game_data:
        count += 1
        ret[x["id"]] = x
    return ret

def getPlayerAverageStats(game_ids, player_id, game_data):
    player_game_data = {"minutes":0, "pls_min":0}

    for field in fields_to_average:
        player_game_data[field] = 0
    
    player_game_data["games_played"] = len(game_ids)
    for game in game_ids:
        player_stats = getPlayerStatsObject(game_data[game], player_id)
        for field in fields_to_average:
            player_game_data[field] += player_stats[field]

        if player_stats.get("minutes"):
            player_game_data["minutes"] += 60*int(player_stats["minutes"].split(":")[0]) + int(player_stats["minutes"].split(":")[1])
            player_game_data["pls_min"] += player_stats.get("pls_min") or 0


    player_game_data["minutes"] = player_game_data["minutes"]/60
    player_game_data["field_goals_pct"] = player_game_data["field_goals_made"]/player_game_data["field_goals_att"] if player_game_data["field_goals_att"] else 0
    player_game_data["three_points_pct"] = player_game_data["three_points_made"]/player_game_data["three_points_att"] if player_game_data["three_points_att"] else 0
    player_game_data["two_points_pct"] = player_game_data["two_points_made"]/player_game_data["two_points_att"] if player_game_data["two_points_att"] else 0
    player_game_data["free_throws_pct"] = player_game_data["free_throws_made"]/player_game_data["free_throws_att"] if player_game_data["free_throws_att"] else 0
    player_game_data["assist_turnovers_ratio"] = player_game_data["assists"]/player_game_data["turnovers"] if player_game_data["turnovers"] else 0

    for field in fields_to_average:
        if len(game_ids) >0:
            player_game_data[field] = player_game_data[field] / len(game_ids)

    if len(game_ids) >0:
        player_game_data["minutes"] = player_game_data["minutes"]/len(game_ids)
    return player_game_data


def getPlayerSeasonAveragesBeforeGame(player_games, game_data, player_id, game_date):
    if player_games.get(player_id):
        player_game_list = getPlayerGameIdsBeforeGame(player_games[player_id], game_date)
    else:
        player_game_list = []
    return getPlayerAverageStats(player_game_list, player_id, game_data)


def getActivePlayers(player_list):
    return [p for p in player_list if p.get("active")]

def getPlayerCompositeDataForYear(year):
    player_composite_data = {"h":1}
    f1 = open("data/game_data_"+str(year)+".json")
    data1 = json.load(f1)
    #print(data1[0])

    game_data = makeGameObj(data1)

    f2 = open("data/player_data/player_ref_data_"+str(year)+".json")
    data2 = json.load(f2)
    count = 0
    for game in data1:
        count+=1
        print("Game "+ str(count))
        player_composite_data[game["id"]] = {"reference" : {"id": game["id"]}}
        player_composite_data[game["id"]]["inputs"] = {"home_players": [], "away_players":[]}

        for player in getActivePlayers(game["home"]["players"]):
            p_arr = {"reference": {"full_name": player["full_name"], "id": player["id"]}, "inputs": getPlayerSeasonAveragesBeforeGame(data2, game_data, player["id"], parser.parse(game["scheduled"]))}
            player_composite_data[game["id"]]["inputs"]["home_players"].append(p_arr)

        for player in getActivePlayers(game["away"]["players"]):
            p_arr = {"reference": {"full_name": player["full_name"], "id": player["id"]}, "inputs": getPlayerSeasonAveragesBeforeGame(data2, game_data, player["id"], parser.parse(game["scheduled"]))}
            player_composite_data[game["id"]]["inputs"]["away_players"].append(p_arr)

    return player_composite_data


for year in range(2013, 2021):
    print("***************** "+ str(year)+ "***********************")
    pdata = getPlayerCompositeDataForYear(year)
    with open("data/player_data/player_composite_"+str(year)+".json", "w") as out:
        json.dump(pdata, out)

"""
year = 2014
f1 = open("data/game_data_"+str(year)+".json")
data1 = json.load(f1)

game_data = makeGameObj(data1)

f2 = open("data/player_data/player_ref_data_"+str(year)+".json")
data2 = json.load(f2)


x= getPlayerSeasonAveragesBeforeGame(data2, game_data, "9cf99a61-6b51-4aed-8940-0480dc512b36", parser.parse("2015-05-04T03:00:00+00:00"))

print(x)
"""
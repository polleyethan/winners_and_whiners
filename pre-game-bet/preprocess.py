import json
import numpy as np

stat_fields = ["field_goals_made", "field_goals_att", "three_points_made", "three_points_att", "two_points_made", "two_points_att", "free_throws_made", "free_throws_att", "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "turnovers", "steals", "blocks", "points", "field_goals_pct", "three_points_pct", "two_points_pct", "free_throws_pct", "assists_turnover_ratio"]
player_stat_fields = [ 'field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'two_points_made',
 'two_points_att', 'free_throws_made', 'free_throws_att', 'offensive_rebounds', 'defensive_rebounds',
  'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'points', "minutes", "field_goals_pct", "three_points_pct", "two_points_pct", "pls_min", "free_throws_pct", "assist_turnovers_ratio"]


def getNonePlayerObj():
    p = {}
    p["reference"] = {"id": "None", "full_name": "None"}
    p["inputs"] = {}
    for f in player_stat_fields:
        p["inputs"][f] = 0

    return p
def getTeamDataForHomeWin(year):
    f = open("data/composite/composite_"+str(year)+".json")
    games = json.load(f)

    data = []

    outs = []

    for idx, game in enumerate(games):
        inputs = []
        if game["inputs"]["home_points"] > 0 and game["inputs"]["away_points"] > 0:
            for field in stat_fields:
                inputs.append(game["inputs"]["home_"+field])
                inputs.append(game["inputs"]["home_"+field+"_against"])
                inputs.append(game["inputs"]["away_"+field])
                inputs.append(game["inputs"]["away_"+field+"_against"])
            
            outs.append([game["ouputs"]["home_win"],game["ouputs"]["away_win"]])
            data.append(inputs)

    return (data, outs)


def getTeamDataForPoints(year):
    f = open("data/composite/composite_"+str(year)+".json")
    games = json.load(f)

    data = []

    outs = []

    for idx, game in enumerate(games):
        inputs = []
        if game["inputs"]["home_points"] > 0 and game["inputs"]["away_points"] > 0:
            for field in stat_fields:
                inputs.append(game["inputs"]["home_"+field])
                inputs.append(game["inputs"]["home_"+field+"_against"])
                inputs.append(game["inputs"]["away_"+field])
                inputs.append(game["inputs"]["away_"+field+"_against"])
            
            outs.append([game["ouputs"]["home_points"], game["ouputs"]["away_points"]])
            data.append(inputs)

    return (data, outs)


def get_win_data():
    data, outs = getTeamDataForHomeWin(2013)

    for year in range(2014, 2021):
        print(year)
        (yr_d, yr_o) = getTeamDataForHomeWin(year)
        data = np.concatenate((data, yr_d))
        outs = np.concatenate((outs, yr_o))

    game_cut = 4*len(data)//5

    return (data[:game_cut], outs[:game_cut], data[game_cut:], outs[game_cut:])

def get_points_data():
    data, outs = getTeamDataForPoints(2013)

    for year in range(2014, 2021):
        print(year)
        (yr_d, yr_o) = getTeamDataForPoints(year)
        data = np.concatenate((data, yr_d))
        outs = np.concatenate((outs, yr_o))

    game_cut = 4*len(data)//5

    return (data[:game_cut], outs[:game_cut], data[game_cut:], outs[game_cut:])



def getPlayersDataForGame(game_data):
    data = []
    game_data["inputs"]["home_players"].sort(key = lambda x: x["inputs"]["minutes"], reverse=True)
    game_data["inputs"]["away_players"].sort(key = lambda x: x["inputs"]["minutes"], reverse=True)

    if len(game_data["inputs"]["home_players"]) < 12:
        for x in range(12 - len(game_data["inputs"]["home_players"])):
            game_data["inputs"]["home_players"].append(getNonePlayerObj())

    if len(game_data["inputs"]["away_players"]) < 12:
        for x in range(12 - len(game_data["inputs"]["away_players"])):
            game_data["inputs"]["away_players"].append(getNonePlayerObj())
    
    for player in game_data["inputs"]["home_players"][:12]:
        for field in player_stat_fields:
            data.append(player["inputs"][field])

    for player in game_data["inputs"]["away_players"][:12]:
        for field in player_stat_fields:
            data.append(player["inputs"][field])
    
    return data


def getTeamDataForHomeWinWithPlayers(year):
    f = open("data/composite/composite_"+str(year)+".json")
    games = json.load(f)
    f2 = open("data/player_data/player_composite_"+str(year)+".json")
    player_games = json.load(f2)

    data = []

    outs = []

    for idx, game in enumerate(games):
        inputs = []
        if game["inputs"]["home_points"] > 0 and game["inputs"]["away_points"] > 0:
            for field in stat_fields:
                inputs.append(game["inputs"]["home_"+field])
                inputs.append(game["inputs"]["home_"+field+"_against"])
                inputs.append(game["inputs"]["away_"+field])
                inputs.append(game["inputs"]["away_"+field+"_against"])
            
            outs.append([game["ouputs"]["home_win"],game["ouputs"]["away_win"]])
            data.append(np.concatenate((inputs, getPlayersDataForGame(player_games[game["reference"]["sportsradar_id"]]))))

    return (data, outs)

def getTeamDataForPointsWithPlayers(year):
    f = open("data/composite/composite_"+str(year)+".json")
    games = json.load(f)
    f2 = open("data/player_data/player_composite_"+str(year)+".json")
    player_games = json.load(f2)
    data = []

    outs = []

    for idx, game in enumerate(games):
        inputs = []
        if game["inputs"]["home_points"] > 0 and game["inputs"]["away_points"] > 0:
            for field in stat_fields:
                inputs.append(game["inputs"]["home_"+field])
                inputs.append(game["inputs"]["home_"+field+"_against"])
                inputs.append(game["inputs"]["away_"+field])
                inputs.append(game["inputs"]["away_"+field+"_against"])
            
            outs.append([game["ouputs"]["home_points"], game["ouputs"]["away_points"]])
            data.append(np.concatenate((inputs, getPlayersDataForGame(player_games[game["reference"]["sportsradar_id"]]))))

    return (data, outs)


def get_win_data_with_players():
    data, outs = getTeamDataForHomeWinWithPlayers(2013)

    for year in range(2014, 2021):
        print(year)
        (yr_d, yr_o) = getTeamDataForHomeWinWithPlayers(year)
        data = np.concatenate((data, yr_d))
        outs = np.concatenate((outs, yr_o))

    game_cut = 4*len(data)//5

    return (data[:game_cut], outs[:game_cut], data[game_cut:], outs[game_cut:])


def get_points_data_with_players():
    data, outs = getTeamDataForPointsWithPlayers(2013)

    for year in range(2014, 2021):
        print(year)
        (yr_d, yr_o) = getTeamDataForPointsWithPlayers(year)
        data = np.concatenate((data, yr_d))
        outs = np.concatenate((outs, yr_o))

    game_cut = 4*len(data)//5

    return (data[:game_cut], outs[:game_cut], data[game_cut:], outs[game_cut:])

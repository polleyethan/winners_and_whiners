from datetime import date
from os import stat
import tensorflow as tf
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from nba_api.stats.endpoints import leaguegamefinder
from preprocess import getNonePlayerObj

stat_fields = ["field_goals_made", "field_goals_att", "three_points_made", "three_points_att", 
"two_points_made", "two_points_att", "free_throws_made", "free_throws_att", "offensive_rebounds", 
"defensive_rebounds", "rebounds", "assists", "turnovers", "blocks", "steals", "points", "field_goals_pct", 
"three_points_pct", "two_points_pct", "free_throws_pct", "assists_turnover_ratio"]

stat_fields_mapped = {"field_goals_made":7, "field_goals_att":8, "three_points_made":10, "three_points_att":11, 
 "free_throws_made":13, "free_throws_att":14, "offensive_rebounds":16, 
"defensive_rebounds":17, "rebounds":18, "assists":19, "turnovers":20, "steals":21, "blocks":22, "points":26, "field_goals_pct":9, 
"three_points_pct":12, "free_throws_pct":15}


player_stat_fields = [ 'field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'two_points_made', 'two_points_att', 'free_throws_made', 'free_throws_att', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'points', "minutes", "field_goals_pct", "three_points_pct", "two_points_pct", "pls_min", "free_throws_pct", "assist_turnovers_ratio"]



def getPlayerStatsForTeam(team_abbrev, exclude_dtd):
    data = []
    URL = "https://www.espn.com/nba/team/stats/_/name/"+team_abbrev
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    name_table = soup.find_all('table')[0]
    names = name_table.find_all('tr')

    for idx, x in enumerate(names[1:len(names)-1]):
        id = x.find('a').attrs['data-player-uid'].split(":")[3]
        name = x.find('a').text
        data.append({"ref":{"name": name, "id":id}})

    stats_table_1 = soup.find_all('table')[1]
    tr = stats_table_1.find_all('tr')
    for idx, row in enumerate(tr[1:len(tr)-1]):
        data[idx]["inputs"] = {}
        t_data = row.find_all('td')
        data[idx]["inputs"]["minutes"] = float(t_data[2].text)
        data[idx]["inputs"]["points"] = float(t_data[3].text)
        data[idx]["inputs"]["offensive_rebounds"] = float(t_data[4].text)
        data[idx]["inputs"]["defensive_rebounds"] = float(t_data[5].text)
        data[idx]["inputs"]["rebounds"] = float(t_data[6].text)
        data[idx]["inputs"]["assists"] = float(t_data[7].text)
        data[idx]["inputs"]["steals"] = float(t_data[8].text)
        data[idx]["inputs"]["blocks"] = float(t_data[9].text)
        data[idx]["inputs"]["turnovers"] = float(t_data[10].text)
        data[idx]["inputs"]["assist_turnovers_ratio"] = float(t_data[12].text)
        

    stats_table_2 = soup.find_all('table')[3]

    tr2 = stats_table_2.find_all('tr')

    for idx, row in enumerate(tr2[1:len(tr2)-1]):
        t_data = row.find_all('td')
        data[idx]["inputs"]["field_goals_made"] = float(t_data[0].text)
        data[idx]["inputs"]["field_goals_att"] = float(t_data[1].text)
        data[idx]["inputs"]["field_goals_pct"] = float(t_data[2].text)/100
        data[idx]["inputs"]["three_points_made"] = float(t_data[3].text)
        data[idx]["inputs"]["three_points_att"] = float(t_data[4].text)
        data[idx]["inputs"]["three_points_pct"] = float(t_data[5].text)/100
        data[idx]["inputs"]["free_throws_made"] = float(t_data[6].text)
        data[idx]["inputs"]["free_throws_att"] = float(t_data[7].text)
        data[idx]["inputs"]["free_throws_pct"] = float(t_data[8].text)/100
        data[idx]["inputs"]["two_points_made"] = float(t_data[9].text)
        data[idx]["inputs"]["two_points_att"] = float(t_data[10].text)
        data[idx]["inputs"]["two_points_pct"] = float(t_data[11].text)/100
        
    
    p_d = leaguedashplayerstats.LeagueDashPlayerStats().get_dict()
    for x in data:
        name = x["ref"]["name"]
        p = [p for p in p_d['resultSets'][0]["rowSet"] if p[1] == name]
        x["inputs"]["pls_min"] = p[0][31]

    data_new = []
    for g in data:
        URL2 = "https://www.espn.com/nba/player/_/id/"+g["ref"]["id"]
        page2 = requests.get(URL2)
        soup2 = BeautifulSoup(page2.content, "html.parser")
        status1 = soup2.find("span", {"class":"TextStatus"})
        if not status1 == None:
            status = status1.text
            if not status == "Out" and not (status == "Day-To-Day" and exclude_dtd):
                data_new.append(g)
        else:
            data_new.append(g)
    print(len(data_new))
    return data_new


#x = getPlayerStatsForTeam("ny")


def getTeamStatsForSeason(home_team_id, away_team_id):
    data = {"ref":{"home_id": home_team_id, "away_id": away_team_id}, "inputs":{}}
    home_league_avgs_ret = leaguedashteamstats.LeagueDashTeamStats(team_id_nullable=home_team_id, per_mode_detailed="PerGame").get_dict()["resultSets"][0]["rowSet"][0]
    away_league_avgs_ret = leaguedashteamstats.LeagueDashTeamStats(team_id_nullable=away_team_id, per_mode_detailed="PerGame").get_dict()["resultSets"][0]["rowSet"][0]
    home_opponent_avgs_ret = leaguedashteamstats.LeagueDashTeamStats(team_id_nullable= home_team_id, per_mode_detailed="PerGame", measure_type_detailed_defense="Opponent").get_dict()["resultSets"][0]["rowSet"][0]
    away_opponent_avgs_ret = leaguedashteamstats.LeagueDashTeamStats(team_id_nullable= away_team_id, per_mode_detailed="PerGame", measure_type_detailed_defense="Opponent").get_dict()["resultSets"][0]["rowSet"][0]


    for f in stat_fields_mapped.keys():
        data["inputs"]["home_"+f] = home_league_avgs_ret[stat_fields_mapped[f]]
        data["inputs"]["away_"+f] = away_league_avgs_ret[stat_fields_mapped[f]]
        data["inputs"]["home_"+f+"_against"] = home_opponent_avgs_ret[stat_fields_mapped[f]]
        data["inputs"]["away_"+f+"_against"] = away_opponent_avgs_ret[stat_fields_mapped[f]]

    for x in ["home", "away"]:
            for y in ["","_against"]:
                data["inputs"][x+"_two_points_made"+y] = data["inputs"][x+"_field_goals_made"+y] - data["inputs"][x+"_three_points_made"+y]
                data["inputs"][x+"_two_points_att"+y] = data["inputs"][x+"_field_goals_att"+y] - data["inputs"][x+"_three_points_att"+y]
                data["inputs"][x+"_two_points_pct"+y] = data["inputs"][x+"_two_points_made"+y] - data["inputs"][x+"_two_points_att"+y]
                data["inputs"][x+"_assists_turnover_ratio"+y] = data["inputs"][x+"_assists"+y] / data["inputs"][x+"_turnovers"+y]
                data["inputs"][x+"_two_points_pct"+y]= data["inputs"][x+"_two_points_pct"+y]/100
                data["inputs"][x+"_three_points_pct"+y]= data["inputs"][x+"_three_points_pct"+y]/100
                data["inputs"][x+"_field_goals_pct"+y]= data["inputs"][x+"_field_goals_pct"+y]/100
                data["inputs"][x+"_free_throws_pct"+y]= data["inputs"][x+"_free_throws_pct"+y]/100


    return data

def getPlayersDataForGame(game_data):
    data = []
    game_data["inputs"]["home_players"].sort(key = lambda x: x["inputs"]["minutes"], reverse=True)
    game_data["inputs"]["away_players"].sort(key = lambda x: x["inputs"]["minutes"], reverse=True)

    if len(game_data["inputs"]["home_players"]) < 12:
        for x in range(13 - len(game_data["inputs"]["home_players"])):
            game_data["inputs"]["home_players"].append(getNonePlayerObj())

    if len(game_data["inputs"]["away_players"]) < 12:
        for x in range(13 - len(game_data["inputs"]["away_players"])):
            game_data["inputs"]["away_players"].append(getNonePlayerObj())
    
    for player in game_data["inputs"]["home_players"][:12]:
        for field in player_stat_fields:
            data.append(player["inputs"][field])

    for player in game_data["inputs"]["away_players"][:12]:
        for field in player_stat_fields:
            data.append(player["inputs"][field])
    
    print(data)
    return data

def getPassableStatArray(home_team_id, away_team_id, abbrev_home, abbrev_away, exclude_dtd):
    data = []
    game_data = {"inputs": {"home_players": getPlayerStatsForTeam(abbrev_home, exclude_dtd), "away_players":getPlayerStatsForTeam(abbrev_away, exclude_dtd)}}
    game = getTeamStatsForSeason(home_team_id, away_team_id)
    for field in stat_fields:
        data.append(game["inputs"]["home_"+field])
        data.append(game["inputs"]["home_"+field+"_against"])
        data.append(game["inputs"]["away_"+field])
        data.append(game["inputs"]["away_"+field+"_against"])
            
    
    return np.concatenate((data, getPlayersDataForGame(game_data)))